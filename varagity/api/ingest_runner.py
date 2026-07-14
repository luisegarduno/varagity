"""Background ingest runs with a replayable live event feed (spec_v2 §4.2).

``POST /api/ingest`` must return a run handle immediately while the corpus
ingests (minutes of contextualization on a real corpus), and ``GET
/api/ingest/status`` must stream per-stage/per-file progress that mirrors
the CLI's ``rich`` display. This module owns both halves:

* **One run at a time.** :class:`IngestRunner` executes
  :func:`~varagity.pipeline.ingest_flow.ingest_flow` on a daemon thread —
  the same tracked Prefect flow the CLI runs (and, since it runs in the API
  process, the ingest Prometheus counters finally reach the scrape — the
  Phase 7 note #4 gap). A second start while one is running raises
  :class:`IngestAlreadyRunning` (the route's structured ``409``).
* **Progress without pipeline edits.** The runner wraps the flow's
  ``@task``-wrapped stages (:data:`~varagity.pipeline.ingest_flow.TASK_STAGES`)
  with event emitters — tracking and streaming compose — and passes the
  loader's ``on_file`` observer for exact per-file outcomes. The
  contextualize stage's per-chunk ticks ride a proxy around the ``rich``
  progress handle it already advances. ``varagity.ingest`` log records
  (skips, failures, no-text warnings) are relayed as ``log`` events, so the
  browser sees what the terminal would.
* **Replay + live fan-out.** Events accumulate per run; a subscriber gets
  the full backlog first (connecting mid-run or after completion both
  render correctly), then live events until the terminal ``status`` frame.
"""

import asyncio
import dataclasses
import logging
import threading
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from langchain_core.documents import Document
from pydantic import BaseModel
from rich.progress import Progress, TaskID

from varagity.api.schemas import (
    IngestLogEvent,
    IngestProgressEvent,
    IngestRunOut,
    IngestStatusEvent,
    IngestSummaryOut,
)
from varagity.ingest.loader import IngestStages, IngestSummary
from varagity.ingest.parsers import Parser, RawDocument
from varagity.models import EmbeddingsClient, LLMClient
from varagity.pipeline.ingest_flow import TASK_STAGES, ingest_flow
from varagity.stores import ChunkRecord, ContextualVectorDB, ElasticsearchBM25
from varagity.stores.app_settings_store import AppSettingsStore

logger = logging.getLogger(__name__)

EVENT_STATUS = "status"
EVENT_PROGRESS = "progress"
EVENT_LOG = "log"

# The logger namespace whose records are relayed into the event feed —
# loader outcomes, discovery, and parser warnings all live under it.
_RELAY_LOGGER = "varagity.ingest"

Feed = tuple[str, BaseModel]
"""One feed item: ``(sse_event_name, payload_model)``."""


class IngestAlreadyRunning(RuntimeError):
    """Raised by :meth:`IngestRunner.start` while a run is in flight."""


@dataclasses.dataclass
class _RunState:
    """Mutable state of one ingest run (guarded by the runner's lock).

    Attributes:
        run_id: The run handle.
        reingest: Whether the run re-processes unchanged files.
        state: ``running`` | ``completed`` | ``failed``.
        started_at: Start timestamp.
        finished_at: Terminal timestamp (``None`` while running).
        summary: Final counters (terminal only).
        error: Flow-level failure (``failed`` only).
        files_total: Files discovered (``None`` until discovery lands).
        files_done: Files finished so far.
        current_file: The file the pipeline is on (labels ``embed``).
    """

    run_id: str
    reingest: bool
    state: str = "running"
    started_at: datetime = dataclasses.field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    summary: IngestSummary | None = None
    error: str | None = None
    files_total: int | None = None
    files_done: int = 0
    current_file: str | None = None

    def to_out(self) -> IngestRunOut:
        """Render the wire form of this run.

        Returns:
            The :class:`~varagity.api.schemas.IngestRunOut` snapshot.
        """
        return IngestRunOut(
            run_id=self.run_id,
            state=self.state,
            reingest=self.reingest,
            started_at=self.started_at,
            finished_at=self.finished_at,
            summary=(
                IngestSummaryOut(**dataclasses.asdict(self.summary))
                if self.summary is not None
                else None
            ),
            error=self.error,
        )


class _RelayHandler(logging.Handler):
    """Forward ``varagity.ingest`` log records into the event feed.

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
                EVENT_LOG, IngestLogEvent(level=record.levelname, message=record.getMessage())
            )
        except Exception:  # logging must never raise into the pipeline
            self.handleError(record)


class _EmittingProgress:
    """Proxy over the loader's ``rich`` progress that also emits tick events.

    The contextualize stage advances a per-chunk sub-bar on the ``Progress``
    handle it receives; routing the handle through this proxy turns each
    ``advance`` into an SSE ``progress`` frame — per-chunk granularity on
    the ingest's long pole (~12 s/chunk blurbs) with zero loader edits.

    Args:
        inner: The real progress display (delegated to for the terminal).
        emit: The runner's emit function.
        file_name: The file being contextualized.
        total: Its chunk count.
    """

    def __init__(
        self,
        inner: Progress,
        emit: Callable[[str, BaseModel], None],
        file_name: str,
        total: int,
    ) -> None:
        self._inner = inner
        self._emit = emit
        self._file_name = file_name
        self._total = total
        self._done = 0

    def add_task(self, description: str, total: float | None = None, **kwargs: Any) -> TaskID:
        """Delegate sub-bar creation.

        Args:
            description: The sub-bar label.
            total: The sub-bar denominator.
            **kwargs: Further ``rich`` options, passed through.

        Returns:
            The real task id.
        """
        return self._inner.add_task(description, total=total, **kwargs)

    def advance(self, task_id: TaskID, advance: float = 1) -> None:
        """Delegate one tick and emit it as a ``progress`` frame.

        Args:
            task_id: The sub-bar to advance.
            advance: Tick size.
        """
        self._inner.advance(task_id, advance)
        self._done += int(advance)
        self._emit(
            EVENT_PROGRESS,
            IngestProgressEvent(
                stage="contextualize",
                file=self._file_name,
                current=self._done,
                total=self._total,
            ),
        )

    def remove_task(self, task_id: TaskID) -> None:
        """Delegate sub-bar removal.

        Args:
            task_id: The sub-bar to remove.
        """
        self._inner.remove_task(task_id)

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the real progress display.

        Args:
            name: The attribute being looked up.

        Returns:
            The delegated attribute.
        """
        return getattr(self._inner, name)


def _clear_corpus_stale() -> None:
    """Clear the persisted corpus-stale flag (post-reingest, best-effort).

    A completed ``reingest=true`` run re-processed every discovered file
    under the current settings, so the "Re-ingest to apply" affordance can
    retire. Unreachability is logged, never raised — the run itself
    succeeded.
    """
    try:
        with AppSettingsStore() as store:
            store.set_corpus_stale(False)
    except psycopg.OperationalError:
        logger.warning("could not clear the corpus-stale flag — postgres unreachable")


class IngestRunner:
    """Owner of the (single) background ingest run and its event feed."""

    def __init__(
        self,
        flow: Callable[..., IngestSummary] | None = None,
        *,
        base_stages: IngestStages | None = None,
        on_reingest_complete: Callable[[], None] | None = None,
    ) -> None:
        """Wire the runner's collaborators (all injectable for tests).

        Args:
            flow: The ingest callable, ``ingest_flow``-compatible
                (injectable so tests run without stores/models/Prefect).
            base_stages: The stage bundle to wrap with emitters; the flow's
                Prefect task stages when omitted.
            on_reingest_complete: Hook fired after a ``reingest=true`` run
                completes; defaults to clearing the corpus-stale flag.
        """
        self._flow = flow if flow is not None else ingest_flow
        self._base_stages = base_stages if base_stages is not None else TASK_STAGES
        self._on_reingest_complete = (
            on_reingest_complete if on_reingest_complete is not None else _clear_corpus_stale
        )
        self._lock = threading.Lock()
        self._run: _RunState | None = None
        self._events: list[Feed] = []
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Feed | None]]] = []

    # ── run control ────────────────────────────────────────────────────

    def start(self, *, reingest: bool) -> IngestRunOut:
        """Start a background ingest run.

        Args:
            reingest: Delete each discovered document's previous ingest and
                re-process it (the stale-corpus action).

        Returns:
            The new run's handle/state snapshot.

        Raises:
            IngestAlreadyRunning: If a run is already in flight.
        """
        with self._lock:
            if self._run is not None and self._run.state == "running":
                raise IngestAlreadyRunning(f"run {self._run.run_id} is still running")
            run = _RunState(run_id=uuid4().hex[:12], reingest=reingest)
            self._run = run
            self._events = []
        # Snapshot before the thread starts: the caller's handle (and the
        # feed's first frame) deterministically says "running" even if the
        # run finishes faster than this function returns.
        started = run.to_out()
        self._emit(EVENT_STATUS, IngestStatusEvent(run=started))
        thread = threading.Thread(
            target=self._execute, args=(run,), name=f"ingest-{run.run_id}", daemon=True
        )
        thread.start()
        return started

    def snapshot(self) -> IngestRunOut | None:
        """Report the current (or last) run without subscribing.

        Returns:
            The run snapshot, or ``None`` when nothing ever ran.
        """
        with self._lock:
            return self._run.to_out() if self._run is not None else None

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
            # Nothing ever ran in this API process: say so and close.
            yield (EVENT_STATUS, IngestStatusEvent(run=self.snapshot()))
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
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def _close_feed(self) -> None:
        """Signal end-of-stream to every live subscriber."""
        with self._lock:
            subscribers, self._subscribers = self._subscribers, []
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # ── the worker thread ──────────────────────────────────────────────

    def _execute(self, run: _RunState) -> None:
        """Run the flow with emitting stages; publish the terminal state.

        Args:
            run: The run this thread owns.
        """
        relay = _RelayHandler(self._emit)
        ingest_logger = logging.getLogger(_RELAY_LOGGER)
        # The per-file outcome lines (skips, no-text, failures) log at INFO;
        # the feed must carry them regardless of the process LOG_LEVEL, so
        # the namespace is pinned to INFO for the run (records still pass
        # root handlers' own level filters, so the console stays quiet).
        previous_level = ingest_logger.level
        ingest_logger.setLevel(logging.INFO)
        ingest_logger.addHandler(relay)
        try:
            summary = self._flow(
                reingest=run.reingest,
                verbose=0,
                stages=self._wrap_stages(run),
                on_file=self._make_on_file(run),
            )
        except Exception as error:
            logger.exception("ingest run %s failed", run.run_id)
            with self._lock:
                run.state = "failed"
                run.error = f"{type(error).__name__}: {error}"
                run.finished_at = datetime.now(UTC)
        else:
            with self._lock:
                run.state = "completed"
                run.summary = summary
                run.finished_at = datetime.now(UTC)
        finally:
            ingest_logger.removeHandler(relay)
            ingest_logger.setLevel(previous_level)
            self._emit(EVENT_STATUS, IngestStatusEvent(run=run.to_out()))
            self._close_feed()
            if run.state == "completed" and run.reingest:
                self._on_reingest_complete()

    def _make_on_file(self, run: _RunState) -> Callable[[Path, str, int], None]:
        """Build the loader's per-file observer for this run.

        Args:
            run: The run to update.

        Returns:
            The observer (emits ``file_done`` progress frames).
        """

        def on_file(path: Path, outcome: str, n_chunks: int) -> None:
            with self._lock:
                run.files_done += 1
                done, total = run.files_done, run.files_total
            self._emit(
                EVENT_PROGRESS,
                IngestProgressEvent(
                    stage="file_done",
                    file=path.name,
                    outcome=outcome,
                    total=n_chunks or None,
                    files_done=done,
                    files_total=total,
                ),
            )

        return on_file

    def _wrap_stages(self, run: _RunState) -> IngestStages:
        """Wrap the base stages with progress emitters.

        The wrappers call straight through to the base stages (the Prefect
        tasks — their tracking, retries, and metrics are untouched), adding
        one ``progress`` frame per stage transition and the contextualize
        per-chunk proxy.

        Args:
            run: The run receiving the events.

        Returns:
            The emitting stage bundle.
        """
        base = self._base_stages

        def discover(docs_path: str, verbose: int | None = None) -> Any:
            buckets = base.discover(docs_path, verbose=verbose)
            with self._lock:
                run.files_total = buckets.total
            self._emit(EVENT_PROGRESS, IngestProgressEvent(stage="discover", total=buckets.total))
            return buckets

        def parse(parser: Parser, path: Path, *, verbose: int) -> RawDocument:
            with self._lock:
                run.current_file = path.name
            self._emit(EVENT_PROGRESS, IngestProgressEvent(stage="parse", file=path.name))
            return base.parse(parser, path, verbose=verbose)

        def chunk(raw: RawDocument, *, verbose: int) -> list[Document]:
            chunks = base.chunk(raw, verbose=verbose)
            file_name = raw.source_meta.get("file_name")
            self._emit(
                EVENT_PROGRESS,
                IngestProgressEvent(stage="chunk", file=file_name, total=len(chunks)),
            )
            return chunks

        def contextualize(
            *,
            document_text: str,
            chunk_texts: list[str],
            llm: LLMClient | None,
            file_name: str,
            progress: Progress,
            verbose: int,
        ) -> list[str | None]:
            self._emit(
                EVENT_PROGRESS,
                IngestProgressEvent(
                    stage="contextualize", file=file_name, current=0, total=len(chunk_texts)
                ),
            )
            proxy = _EmittingProgress(progress, self._emit, file_name, len(chunk_texts))
            return base.contextualize(
                document_text=document_text,
                chunk_texts=chunk_texts,
                llm=llm,
                file_name=file_name,
                progress=proxy,
                verbose=verbose,
            )

        def embed(
            texts: list[str], *, embeddings: EmbeddingsClient, verbose: int
        ) -> list[list[float]]:
            with self._lock:
                file_name = run.current_file
            self._emit(
                EVENT_PROGRESS,
                IngestProgressEvent(stage="embed", file=file_name, total=len(texts)),
            )
            return base.embed(texts, embeddings=embeddings, verbose=verbose)

        def store(
            records: list[ChunkRecord],
            vectors: list[list[float]],
            *,
            store: ContextualVectorDB,
            bm25: ElasticsearchBM25,
        ) -> Any:
            result = base.store(records, vectors, store=store, bm25=bm25)
            self._emit(
                EVENT_PROGRESS,
                IngestProgressEvent(stage="store", file=records[0].file_name, total=len(records)),
            )
            return result

        return IngestStages(
            discover=discover,
            parse=parse,
            chunk=chunk,
            contextualize=contextualize,
            embed=embed,
            store=store,
        )


_default_runner: IngestRunner | None = None


def get_ingest_runner() -> IngestRunner:
    """Provide the process-wide runner (FastAPI dependency, override seam).

    Returns:
        The lazily created singleton.
    """
    global _default_runner
    if _default_runner is None:
        _default_runner = IngestRunner()
    return _default_runner
