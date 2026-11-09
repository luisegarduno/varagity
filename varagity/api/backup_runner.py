"""Background backup runs with a replayable live event feed (ADR-019).

``POST /api/backups/{corpus}/export`` must answer at once while the bundle
is written, and ``GET /api/backups/{corpus}/status`` must stream where the
run is — the same contract ingest and the graph build already keep, so the
``/corpus`` panels fold all three with the same kind of pure reducer:

* **One writer per corpus.** :meth:`BackupRunner.start` claims the corpus's
  :class:`~varagity.api.corpus_gate.CorpusGate` and holds it for the whole
  run; a second export or restore — or an ingest, which claims the same
  gate — is refused with :class:`~varagity.api.corpus_gate.CorpusBusy`, the
  route's ``409 corpus_busy``. There is no separate single-flight flag: the
  gate *is* the one-run-at-a-time rule.
* **A tracked flow.** The run executes the ``rag-export``, ``rag-restore``,
  ``graph-export`` or ``graph-restore`` Prefect flow on a daemon thread,
  in-process, exactly as the CLI and the ingest runner run theirs; stage
  progress rides an ``on_progress`` callback and the ``varagity.backup`` log
  records are relayed as ``log`` events. Each corpus has its own runner (and
  gate), so a graph export and a RAG restore run side by side.
* **Replay + live fan-out.** Events accumulate per run; a subscriber gets
  the full backlog first, then live events until the terminal ``status``
  frame — connecting mid-run or after completion renders the same picture.

The terminal transition is one critical section: the run turns terminal,
its last frame joins the backlog, live subscribers are detached, and the
gate is released together — so a subscriber can never see a finished run
without its final frame, and a client reacting to that frame can always
start the next run.
"""

import asyncio
import contextlib
import dataclasses
import logging
import threading
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from varagity.api.corpus_gate import CorpusGate, get_corpus_gates
from varagity.api.runtime_settings import REINGEST_AFFECTING, to_env_value
from varagity.api.schemas import (
    BackupLogEvent,
    BackupProgressEvent,
    BackupRunOut,
    BackupStatusEvent,
    BackupSummaryOut,
)
from varagity.backup.bundle import Corpus, display_path, exports_dir
from varagity.backup.graph import (
    GraphExportReport,
    GraphExportStage,
    GraphRestore,
    GraphRestoreStage,
)
from varagity.backup.rag import ExportStage, RagExport, RagRestore, RestoreStage
from varagity.config import get_settings
from varagity.pipeline.backup_flow import (
    graph_export_flow,
    graph_restore_flow,
    rag_export_flow,
    rag_restore_flow,
)

logger = logging.getLogger(__name__)

EVENT_STATUS = "status"
EVENT_PROGRESS = "progress"
EVENT_LOG = "log"

# The logger namespace whose records are relayed into the event feed — the
# backup package's own lines (what was exported, a swept partial, drift).
_RELAY_LOGGER = "varagity.backup"

BackupOperation = Literal["export", "restore"]
"""What a backup run can do."""

RunState = Literal["running", "completed", "failed"]

BackupStage = ExportStage | RestoreStage | GraphExportStage | GraphRestoreStage
"""Every stage a backup run's ``progress`` frames can name."""

ExportResult = RagExport | GraphExportReport
"""What an export flow returns — either corpus's published bundle and its timing."""

RestoreResult = RagRestore | GraphRestore
"""What a restore flow returns — what either corpus's restore wrote, and its stale flag."""

Feed = tuple[str, BaseModel]
"""One feed item: ``(sse_event_name, payload_model)``."""

# The flows each corpus backs up and restores with when none is injected.
_EXPORT_FLOWS: dict[str, Callable[..., ExportResult]] = {
    "rag": rag_export_flow,
    "graph": graph_export_flow,
}
_RESTORE_FLOWS: dict[str, Callable[..., RestoreResult]] = {
    "rag": rag_restore_flow,
    "graph": graph_restore_flow,
}


def current_ingest_settings() -> dict[str, str]:
    """Snapshot the reingest-affecting settings, in env-string form.

    What a RAG bundle records as ``ingest_settings``: the values whose
    change marks the corpus stale (``CHUNKING_STRATEGY``, ``CHUNK_SIZE``,
    ``CHUNK_OVERLAP``, ``CONTEXTUALIZE``, ``OCR_ENGINE``), rendered exactly
    as ``app_settings`` stores an override — so a restore compares like with
    like. Read here, in the API layer, so the backup package never imports
    it.

    Returns:
        Setting name → env-string value, sorted by name.
    """
    settings = get_settings()
    return {name: to_env_value(getattr(settings, name)) for name in sorted(REINGEST_AFFECTING)}


@dataclasses.dataclass
class _RunState:
    """Mutable state of one backup run (guarded by the runner's lock).

    Attributes:
        run_id: The run handle.
        corpus: The corpus being backed up.
        operation: What the run does.
        ingest_settings: The reingest-affecting settings captured at start
            (what a RAG export's bundle records); ``None`` for the graph,
            which has none.
        state: ``running`` | ``completed`` | ``failed``.
        started_at: Start timestamp.
        finished_at: Terminal timestamp (``None`` while running).
        bundle: An export's published bundle (``completed`` only), or the
            bundle a restore restores (from the start).
        saved_to: An export's checkout-relative path (``completed`` only).
        error: The run-level failure (``failed`` only).
        summary: The final counters (``completed`` only).
    """

    run_id: str
    corpus: Corpus
    operation: BackupOperation
    ingest_settings: dict[str, str] | None
    state: RunState = "running"
    started_at: datetime = dataclasses.field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    bundle: str | None = None
    saved_to: str | None = None
    error: str | None = None
    summary: BackupSummaryOut | None = None

    def to_out(self) -> BackupRunOut:
        """Render the wire form of this run.

        Returns:
            The :class:`~varagity.api.schemas.BackupRunOut` snapshot.
        """
        return BackupRunOut(
            run_id=self.run_id,
            corpus=self.corpus,
            operation=self.operation,
            state=self.state,
            bundle=self.bundle,
            saved_to=self.saved_to,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            summary=self.summary,
        )


class _RelayHandler(logging.Handler):
    """Forward ``varagity.backup`` log records into the event feed.

    Args:
        emit: The runner's emit function.
    """

    def __init__(self, emit: Callable[[str, BaseModel], None]) -> None:
        super().__init__(level=logging.INFO)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        """Relay one record as a ``log`` event.

        Args:
            record: The log record (its formatted message only — tracebacks
                stay in the server log).
        """
        try:
            self._emit(
                EVENT_LOG, BackupLogEvent(level=record.levelname, message=record.getMessage())
            )
        except Exception:  # logging must never raise into the export
            self.handleError(record)


def _deliver(
    subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Feed | None]]],
    item: Feed | None,
) -> None:
    """Hand one feed item to every live subscriber's loop (any thread).

    Args:
        subscribers: ``(loop, queue)`` pairs.
        item: The item, or ``None`` (the end-of-stream sentinel).
    """
    for loop, queue in subscribers:
        # RuntimeError: that subscriber's loop already closed — it is gone.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, item)


class BackupRunner:
    """Owner of one corpus's (single) background backup run and its event feed.

    Attributes:
        corpus: The corpus this runner backs up.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        gate: CorpusGate | None = None,
        export_flow: Callable[..., ExportResult] | None = None,
        restore_flow: Callable[..., RestoreResult] | None = None,
        ingest_settings: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        """Wire the runner's collaborators (all injectable for tests).

        Args:
            corpus: The corpus to back up.
            gate: The corpus's write gate; the process-wide one when omitted.
            export_flow: The export callable — ``rag_export_flow``-compatible
                for ``rag``, ``graph_export_flow``-compatible for ``graph``;
                the corpus's own when omitted (injectable so tests run
                without stores, an engine, or Prefect).
            restore_flow: The restore callable — ``rag_restore_flow``-
                compatible for ``rag``, ``graph_restore_flow``-compatible for
                ``graph``; the corpus's own when omitted.
            ingest_settings: Provider of the reingest-affecting settings —
                what a RAG export's bundle records and what a RAG restore's
                stale flag is judged against; :func:`current_ingest_settings`
                by default (never called for the graph, which has none).

        Raises:
            KeyError: If no export or restore flow exists for ``corpus`` and
                none was injected.
        """
        self.corpus: Corpus = corpus
        self._gate = gate if gate is not None else get_corpus_gates()[corpus]
        self._export_flow = export_flow if export_flow is not None else _EXPORT_FLOWS[corpus]
        self._restore_flow = restore_flow if restore_flow is not None else _RESTORE_FLOWS[corpus]
        self._ingest_settings = (
            ingest_settings if ingest_settings is not None else current_ingest_settings
        )
        self._lock = threading.Lock()
        self._run: _RunState | None = None
        self._events: list[Feed] = []
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Feed | None]]] = []

    # ── run control ────────────────────────────────────────────────────

    def start(self, operation: BackupOperation, bundle: str | None = None) -> BackupRunOut:
        """Start a background backup run.

        Args:
            operation: What to run (``export`` | ``restore``).
            bundle: The bundle a restore restores — a file name in the
                corpus's exports folder (the route has already resolved and
                verified it). Not taken by an export.

        Returns:
            The new run's handle/state snapshot (state ``running``).

        Raises:
            CorpusBusy: If any operation already holds the corpus — a second
                export or restore included.
            ValueError: If a restore names no bundle, or an export names one.
        """
        if (operation == "restore") != (bundle is not None):
            raise ValueError("a restore takes the bundle to restore; an export takes none")
        ingest_settings = self._ingest_settings() if self.corpus == "rag" else None
        with self._lock:
            self._gate.claim(operation)
            run = _RunState(
                run_id=uuid4().hex[:12],
                corpus=self.corpus,
                operation=operation,
                ingest_settings=ingest_settings,
                bundle=bundle,
            )
            started = run.to_out()
            self._run = run
            # The first frame joins the backlog in the same critical section
            # that makes the run current: a subscriber can never find a live
            # run with an empty feed.
            self._events = [(EVENT_STATUS, BackupStatusEvent(run=started))]
        thread = threading.Thread(
            target=self._execute,
            args=(run,),
            name=f"backup-{self.corpus}-{run.run_id}",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError as error:  # no thread to run it: give the corpus back
            self._finish(run, error=f"{type(error).__name__}: {error}")
            raise
        return started

    def snapshot(self) -> BackupRunOut | None:
        """Report the current (or last) run without subscribing.

        Returns:
            The run snapshot, or ``None`` when nothing ever ran.
        """
        with self._lock:
            return self._run.to_out() if self._run is not None else None

    @property
    def running(self) -> bool:
        """Whether a run is in flight.

        Returns:
            ``True`` between a start and its terminal frame.
        """
        with self._lock:
            return self._run is not None and self._run.state == "running"

    def ensure_free(self) -> None:
        """Refuse at once if any operation holds the corpus (without claiming it).

        The restore route's cheap pre-check: a busy corpus answers ``409``
        before the bundle is hashed.

        Raises:
            CorpusBusy: If the corpus is held.
        """
        self._gate.ensure_free()

    # ── the event feed ─────────────────────────────────────────────────

    async def subscribe(self) -> AsyncIterator[Feed]:
        """Stream this runner's events: full backlog first, then live.

        The stream ends after the terminal ``status`` frame (or immediately
        after an idle/terminal snapshot when nothing is running), so a
        client connecting at any point renders the same picture.

        Yields:
            ``(event_name, payload)`` feed items in emission order.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Feed | None] = asyncio.Queue()
        with self._lock:
            backlog = list(self._events)
            live = self._run is not None and self._run.state == "running"
            if live:
                self._subscribers.append((loop, queue))
        if not backlog:
            # Nothing ever ran for this corpus in this API process: say so.
            yield (EVENT_STATUS, BackupStatusEvent(run=None))
            return
        try:
            for item in backlog:
                yield item
            if not live:
                return
            while (live_item := await queue.get()) is not None:
                yield live_item
        finally:
            with self._lock:
                self._subscribers = [entry for entry in self._subscribers if entry[1] is not queue]

    def _emit(self, event: str, payload: BaseModel) -> None:
        """Append one event and fan it out to live subscribers (any thread).

        Args:
            event: The SSE event name.
            payload: Its typed payload.
        """
        item: Feed = (event, payload)
        with self._lock:
            self._events.append(item)
            subscribers = list(self._subscribers)
        _deliver(subscribers, item)

    def _report_progress(self, stage: BackupStage, current: int | None, total: int | None) -> None:
        """Turn the flow's progress callback into ``progress`` frames.

        Args:
            stage: The export or restore stage.
            current: Rows (a graph export's ``pack``: bytes) done so far in a
                counted stage.
            total: The stage's row (byte) count.
        """
        self._emit(EVENT_PROGRESS, BackupProgressEvent(stage=stage, current=current, total=total))

    # ── the worker thread ──────────────────────────────────────────────

    def _execute(self, run: _RunState) -> None:
        """Run the export or restore flow; publish the terminal state.

        Args:
            run: The run this thread owns.
        """
        relay = _RelayHandler(self._emit)
        backup_logger = logging.getLogger(_RELAY_LOGGER)
        # The backup's own lines log at INFO; the feed must carry them
        # whatever the process LOG_LEVEL (root handlers still apply their
        # own level filters, so the console stays as quiet as configured).
        previous_level = backup_logger.level
        backup_logger.setLevel(logging.INFO)
        backup_logger.addHandler(relay)
        result: ExportResult | RestoreResult | None = None
        failure: str | None = None
        try:
            result = self._restore(run) if run.operation == "restore" else self._export(run)
        except Exception as error:
            logger.exception("backup run %s (%s %s) failed", run.run_id, self.corpus, run.operation)
            failure = f"{type(error).__name__}: {error}"
        finally:
            backup_logger.removeHandler(relay)
            backup_logger.setLevel(previous_level)
            self._finish(run, result=result, error=failure)

    def _export(self, run: _RunState) -> ExportResult:
        """Run the corpus's export flow into its exports folder.

        A RAG export records the reingest-affecting settings captured at
        start — read here, in the API layer, so the backup package never
        imports it. A graph export needs nothing from this layer: its flow
        reads the graph-stale flag and finds the process's graph service
        itself.

        Args:
            run: The export run this thread owns.

        Returns:
            The flow's published bundle and its timing.
        """
        dest = exports_dir(self.corpus)
        if self.corpus == "rag":
            assert run.ingest_settings is not None  # start() captures them for every RAG run
            return self._export_flow(
                dest, ingest_settings=run.ingest_settings, on_progress=self._report_progress
            )
        return self._export_flow(dest, on_progress=self._report_progress)

    def _restore(self, run: _RunState) -> RestoreResult:
        """Run the corpus's restore flow over the run's bundle.

        A RAG restore gets the *provider* of the reingest-affecting settings,
        not a snapshot: its stale flag is judged against the settings current
        when it is written. A graph restore gets the run's id instead, which
        names the folders it parks beside the workdir while it swaps.

        Args:
            run: The restore run this thread owns.

        Returns:
            What the flow restored, with the stale flag it left.
        """
        assert run.bundle is not None  # start() requires one for a restore
        bundle = exports_dir(self.corpus) / run.bundle
        if self.corpus == "rag":
            return self._restore_flow(
                bundle, ingest_settings=self._ingest_settings, on_progress=self._report_progress
            )
        return self._restore_flow(bundle, run_id=run.run_id, on_progress=self._report_progress)

    def _finish(
        self,
        run: _RunState,
        *,
        result: ExportResult | RestoreResult | None = None,
        error: str | None = None,
    ) -> None:
        """Make the run terminal, publish its last frame, and free the corpus.

        One critical section (see the module docstring): terminal state,
        terminal frame, subscriber detach, and gate release land together.

        Args:
            run: The run ending.
            result: The flow's outcome, when it completed.
            error: The failure, when it did not.
        """
        with self._lock:
            run.finished_at = datetime.now(UTC)
            if isinstance(result, RagExport | GraphExportReport) and error is None:
                run.state = "completed"
                run.bundle = result.bundle.name
                run.saved_to = display_path(self.corpus, result.bundle.name)
                run.summary = BackupSummaryOut(
                    counts=result.counts,
                    size_bytes=result.bundle.size_bytes,
                    wall_clock_s=round(result.wall_clock_s, 3),
                )
            elif isinstance(result, RagRestore | GraphRestore) and error is None:
                run.state = "completed"
                run.summary = BackupSummaryOut(
                    counts=result.counts,
                    size_bytes=result.size_bytes,
                    wall_clock_s=round(result.wall_clock_s, 3),
                    stale=result.stale,
                )
            else:
                run.state = "failed"
                run.error = error or (
                    "the export returned no bundle"
                    if run.operation == "export"
                    else "the restore returned no result"
                )
            item: Feed = (EVENT_STATUS, BackupStatusEvent(run=run.to_out()))
            self._events.append(item)
            subscribers, self._subscribers = self._subscribers, []
            self._gate.release()
        _deliver(subscribers, item)
        _deliver(subscribers, None)


_default_runners: dict[str, BackupRunner] | None = None
_default_runners_lock = threading.Lock()


def get_backup_runners() -> dict[str, BackupRunner]:
    """Provide the process-wide backup runners, keyed by corpus (dependency, override seam).

    Returns:
        The lazily created runners, one per corpus (``rag``, ``graph``) —
        each on its corpus's process-wide gate, so a backup and that
        corpus's ingest or build always exclude each other. Constructing one
        resolves nothing: the graph's flow finds the graph service only when
        an export runs.
    """
    global _default_runners
    with _default_runners_lock:
        if _default_runners is None:
            _default_runners = {"rag": BackupRunner("rag"), "graph": BackupRunner("graph")}
        return _default_runners
