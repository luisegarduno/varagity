"""Background graph builds with a replayable live event feed (spec_graphrag §5.2).

``POST /api/graph/build`` must return a run handle immediately: a real
archive indexes at ADR-017's measured 7.17 s/message, so a full backfill is
*hours to days* of extraction on the single llama.cpp slot. This module is
:mod:`varagity.api.ingest_runner`'s shape applied to that job — one run at a
time on a daemon thread, an event feed with full backlog replay, log records
relayed from the ``varagity.graph`` namespace — with the three differences a
multi-day job forces:

* **Resume is the normal case, not an error path.** The engine enqueues
  documents into a durable status store and re-selects every in-flight or
  failed one at the top of each batch, so a killed build is finished by
  starting another (the runner reports what it inherited as ``resumed``).
  Nothing here needs to checkpoint; the discipline is simply not to wipe
  unless asked.
* **Progress comes from two clocks.** Scanning and parsing the corpus
  directory takes seconds and emits per-file frames; the engine's own
  extraction takes hours and is sampled — :data:`_SAMPLE_INTERVAL_S` apart —
  from its document-status counts, on a second thread, while the build call
  is still blocked inside the flow.
* **Bounds are applied before the engine sees anything.** ``message_limit``
  and ``since`` narrow the merged message stream so a spot check costs
  minutes; a bounded run then tells the session **not** to prune, because
  its render is partial by construction and pruning on its say-so would
  delete the rest of the archive (stage-2 decision #9).

The build itself runs through
:func:`varagity.pipeline.graph_flow.graph_build_flow`, so every attempt is a
tracked Prefect flow run, and through
:class:`~varagity.graph.service.GraphService`, so the API stays the single
writer of the graph working directory.
"""

import asyncio
import dataclasses
import logging
import shutil
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
from pydantic import BaseModel

from varagity.api.schemas import (
    GraphBuildLogEvent,
    GraphBuildProgressEvent,
    GraphBuildRunOut,
    GraphBuildStatusEvent,
    GraphBuildSummaryOut,
    GraphParseSummary,
)
from varagity.config import get_settings
from varagity.graph.records import BuildReport
from varagity.graph.service import GraphService, get_graph_service
from varagity.graph.sources.base import (
    MessageBatch,
    SourceMessage,
    batch_for_path,
    find_message_source,
)
from varagity.pipeline.graph_flow import graph_build_flow
from varagity.stores.app_settings_store import AppSettingsStore

logger = logging.getLogger(__name__)

EVENT_STATUS = "status"
EVENT_PROGRESS = "progress"
EVENT_LOG = "log"

# The logger namespace whose records are relayed into the event feed — the
# parser's per-file summaries, the session's build diff, and the adapter's
# per-document failures all live under it.
_RELAY_LOGGER = "varagity.graph"

# How often the engine's document statuses are sampled while a build is in
# flight. The counts move on the order of minutes per document, so this is
# about the browser feeling alive, not about resolution.
_SAMPLE_INTERVAL_S = 2.0

# Document statuses that mean "this document still owes work" — what a
# killed build left behind and a new one inherits (the engine's own
# vocabulary, matched case-insensitively).
_INFLIGHT_STATUSES = frozenset({"pending", "processing", "parsing", "analyzing", "failed"})

Feed = tuple[str, BaseModel]
"""One feed item: ``(sse_event_name, payload_model)``."""


class GraphBuildAlreadyRunning(RuntimeError):
    """Raised by :meth:`GraphBuildRunner.start` while a build is in flight."""


class ScannedDocument(BaseModel):
    """One graph source file as the last build's scan found it.

    Attributes:
        relative_path: The file's POSIX path under ``GRAPH_DOCS_PATH``.
        doc_id: The stable id derived for it (relative path + byte hash).
        parse: What the scan parsed out of it.
    """

    relative_path: str
    doc_id: str
    parse: GraphParseSummary


def scan_graph_corpus(root: Path) -> list[Path]:
    """List every file under the graph corpus root, in stable path order.

    Args:
        root: ``GRAPH_DOCS_PATH``.

    Returns:
        Every regular file under ``root``, sorted by relative path — files
        no registered source claims included (a contacts list, a
        ``chat.db``'s ``-wal`` sidecar). The caller decides what to do with
        them; the corpus listing shows the user everything they dropped in.
    """
    if not root.is_dir():
        return []
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.as_posix())


def parse_summary(batch: MessageBatch) -> GraphParseSummary:
    """Summarize one parsed source file for the corpus listing.

    Args:
        batch: The parsed batch.

    Returns:
        Its message/thread counts and time span (both timestamps ``None``
        for a file that parsed to nothing).
    """
    timestamps = [message.timestamp for message in batch.messages]
    return GraphParseSummary(
        messages=len(batch.messages),
        threads=len({message.thread_id for message in batch.messages}),
        first=min(timestamps, default=None),
        last=max(timestamps, default=None),
    )


def apply_bounds(
    batches: Sequence[MessageBatch],
    *,
    message_limit: int | None,
    since: date | None,
) -> list[MessageBatch]:
    """Narrow parsed batches to the messages a bounded build should index.

    ``since`` is a date floor and ``message_limit`` keeps the **newest**
    messages — both bound toward the recent end, which is what a spot check
    on a decade-long archive wants to look at. The cap applies to the
    batches' *merged* timestamp order, so it means "N messages", not "N per
    file"; batch identity survives, because a build's document ids and the
    manifest diff are derived from it.

    Args:
        batches: Parsed source files.
        message_limit: Keep at most this many messages; ``None`` keeps all.
        since: Drop messages older than this date; ``None`` keeps all.

    Returns:
        The narrowed batches, in input order, each keeping only its
        surviving messages. A batch left with none is dropped: handing the
        engine an empty document would claim the conversation is empty.
    """
    if message_limit is None and since is None:
        return list(batches)

    def in_range(message: SourceMessage) -> bool:
        return since is None or message.timestamp.date() >= since

    kept: set[str] | None = None
    if message_limit is not None:
        ordered = sorted(
            (message for batch in batches for message in batch.messages if in_range(message)),
            key=lambda message: (message.timestamp, message.guid),
        )
        kept = {message.guid for message in ordered[-message_limit:]}
    narrowed: list[MessageBatch] = []
    for batch in batches:
        messages = [
            message
            for message in batch.messages
            if in_range(message) and (kept is None or message.guid in kept)
        ]
        if messages:
            narrowed.append(batch.model_copy(update={"messages": messages}))
    return narrowed


@dataclasses.dataclass
class _RunState:
    """Mutable state of one graph build run (guarded by the runner's lock).

    Attributes:
        run_id: The run handle.
        reingest: Whether the run wiped the working directory first.
        message_limit: The requested message cap, if any.
        since: The requested date floor, if any.
        state: ``running`` | ``completed`` | ``failed``.
        started_at: Start timestamp.
        finished_at: Terminal timestamp (``None`` while running).
        summary: Final counters (terminal only).
        error: Run-level failure (``failed`` only).
        files_scanned: Files found under ``GRAPH_DOCS_PATH``.
        files_parsed: Files a registered message source claimed and parsed.
        files_failed: Files that raised while parsing.
        messages_parsed: Messages recovered before the bounds were applied.
        messages_indexed: Messages left after them.
        resumed: Documents already owing work when this run started.
    """

    run_id: str
    reingest: bool
    message_limit: int | None = None
    since: date | None = None
    state: str = "running"
    started_at: datetime = dataclasses.field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    summary: GraphBuildSummaryOut | None = None
    error: str | None = None
    files_scanned: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    messages_parsed: int = 0
    messages_indexed: int = 0
    resumed: int = 0

    @property
    def bounded(self) -> bool:
        """Whether this run's render is deliberately partial.

        Returns:
            ``True`` when a message cap or date floor was requested — the
            runs that must never prune.
        """
        return self.message_limit is not None or self.since is not None

    def to_out(self) -> GraphBuildRunOut:
        """Render the wire form of this run.

        Returns:
            The :class:`~varagity.api.schemas.GraphBuildRunOut` snapshot.
        """
        return GraphBuildRunOut(
            run_id=self.run_id,
            state=self.state,
            reingest=self.reingest,
            bounded=self.bounded,
            message_limit=self.message_limit,
            since=self.since,
            started_at=self.started_at,
            finished_at=self.finished_at,
            summary=self.summary,
            error=self.error,
        )


class _RelayHandler(logging.Handler):
    """Forward ``varagity.graph`` log records into the event feed.

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
                EVENT_LOG, GraphBuildLogEvent(level=record.levelname, message=record.getMessage())
            )
        except Exception:  # logging must never raise into the build
            self.handleError(record)


def _clear_graph_stale() -> None:
    """Clear the persisted graph-stale flag (post-rebuild, best-effort).

    A completed ``reingest=true`` build re-indexed the corpus from scratch,
    so the graph no longer holds messages whose source file is gone and the
    "Re-build to apply" affordance can retire. Unreachability is logged,
    never raised — the build itself succeeded.
    """
    try:
        with AppSettingsStore() as store:
            store.set_graph_stale(False)
    except psycopg.OperationalError:
        logger.warning("could not clear the graph-stale flag — postgres unreachable")


class GraphBuildRunner:
    """Owner of the (single) background graph build and its event feed."""

    def __init__(
        self,
        flow: Callable[..., BuildReport] | None = None,
        *,
        service: GraphService | None = None,
        on_reingest_complete: Callable[[], None] | None = None,
    ) -> None:
        """Wire the runner's collaborators (all injectable for tests).

        Args:
            flow: The build callable, ``graph_build_flow``-compatible
                (injectable so tests run without Prefect or an engine).
            service: The graph service to build through; the process
                singleton, resolved on first use, when omitted.
            on_reingest_complete: Hook fired after a ``reingest=true`` build
                completes; defaults to clearing the graph-stale flag.
        """
        self._flow = flow if flow is not None else graph_build_flow
        self._service = service
        self._on_reingest_complete = (
            on_reingest_complete if on_reingest_complete is not None else _clear_graph_stale
        )
        self._lock = threading.Lock()
        self._run: _RunState | None = None
        self._events: list[Feed] = []
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Feed | None]]] = []
        self._scanned: dict[str, ScannedDocument] = {}

    @property
    def service(self) -> GraphService:
        """The graph service this runner builds through.

        Returns:
            The injected service, or the process singleton — resolved
            lazily, so constructing a runner never touches the registry.
        """
        if self._service is None:
            self._service = get_graph_service()
        return self._service

    # ── run control ────────────────────────────────────────────────────

    def start(
        self,
        *,
        reingest: bool = False,
        message_limit: int | None = None,
        since: date | None = None,
    ) -> GraphBuildRunOut:
        """Start a background graph build.

        Args:
            reingest: Wipe the engine's working directory first and index
                from scratch (the only thing that clears the graph-stale
                flag).
            message_limit: Index at most this many messages, newest first.
            since: Ignore messages older than this date.

        Returns:
            The new run's handle/state snapshot.

        Raises:
            GraphBuildAlreadyRunning: If a build is already in flight.
        """
        with self._lock:
            if self._run is not None and self._run.state == "running":
                raise GraphBuildAlreadyRunning(f"build {self._run.run_id} is still running")
            run = _RunState(
                run_id=uuid4().hex[:12],
                reingest=reingest,
                message_limit=message_limit,
                since=since,
            )
            self._run = run
            self._events = []
        # Snapshot before the thread starts: the caller's handle (and the
        # feed's first frame) deterministically says "running" even if the
        # build finishes faster than this function returns.
        started = run.to_out()
        self._emit(EVENT_STATUS, GraphBuildStatusEvent(run=started))
        thread = threading.Thread(
            target=self._execute, args=(run,), name=f"graph-build-{run.run_id}", daemon=True
        )
        thread.start()
        return started

    def snapshot(self) -> GraphBuildRunOut | None:
        """Report the current (or last) build without subscribing.

        Returns:
            The run snapshot, or ``None`` when nothing ever ran.
        """
        with self._lock:
            return self._run.to_out() if self._run is not None else None

    @property
    def building(self) -> bool:
        """Whether a build is in flight in this process.

        Returns:
            ``True`` while a run has not reached a terminal state.
        """
        with self._lock:
            return self._run is not None and self._run.state == "running"

    def scanned_documents(self) -> dict[str, ScannedDocument]:
        """Report what the last build's scan parsed, keyed by relative path.

        Returns:
            A copy of the per-file parse summaries; empty before any build
            has scanned the corpus (parsing a multi-gigabyte database is a
            build's job, not a list request's).
        """
        with self._lock:
            return dict(self._scanned)

    # ── the event feed ─────────────────────────────────────────────────

    async def subscribe(self) -> AsyncIterator[Feed]:
        """Stream this runner's events: full backlog first, then live.

        The stream ends after the terminal ``status`` frame (or immediately
        after an idle/terminal snapshot when nothing is running), so a
        client connecting at any point — including hours into a backfill —
        renders the same picture.

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
            yield (EVENT_STATUS, GraphBuildStatusEvent(run=self.snapshot()))
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
        """Scan, bound, optionally reset, build; publish the terminal state.

        Args:
            run: The run this thread owns (this thread is its only writer,
                so its counters are read here without the lock and written
                under it — the feed's readers see whole snapshots).
        """
        relay = _RelayHandler(self._emit)
        graph_logger = logging.getLogger(_RELAY_LOGGER)
        # The per-file parse summaries and the build diff log at INFO; the
        # feed must carry them regardless of the process LOG_LEVEL, so the
        # namespace is pinned to INFO for the run (records still pass root
        # handlers' own level filters, so the console stays quiet).
        previous_level = graph_logger.level
        graph_logger.setLevel(logging.INFO)
        graph_logger.addHandler(relay)
        try:
            report = self._run_build(run)
        except Exception as error:
            logger.exception("graph build %s failed", run.run_id)
            summary = self._summary(run, None)
            with self._lock:
                run.state = "failed"
                run.error = f"{type(error).__name__}: {error}"
                run.summary = summary
                run.finished_at = datetime.now(UTC)
        else:
            summary = self._summary(run, report)
            with self._lock:
                run.state = "completed"
                run.summary = summary
                run.finished_at = datetime.now(UTC)
        finally:
            graph_logger.removeHandler(relay)
            graph_logger.setLevel(previous_level)
            self._emit(EVENT_STATUS, GraphBuildStatusEvent(run=run.to_out()))
            self._close_feed()
            if run.state == "completed" and run.reingest:
                self._on_reingest_complete()

    def _run_build(self, run: _RunState) -> BuildReport:
        """Do the run's work on the worker thread.

        Args:
            run: The run being executed.

        Returns:
            The engine's build report.
        """
        batches = self._scan(run)
        bounded = apply_bounds(batches, message_limit=run.message_limit, since=run.since)
        indexed = sum(len(batch.messages) for batch in bounded)
        with self._lock:
            run.messages_indexed = indexed
        if run.bounded:
            self._emit(
                EVENT_PROGRESS,
                GraphBuildProgressEvent(stage="bound", current=indexed, total=run.messages_parsed),
            )
        if run.reingest:
            self._reset_workdir()
        resumed = self._inflight_documents()
        with self._lock:
            run.resumed = resumed
        if resumed:
            logger.info("resuming %d document(s) an earlier build left in flight", resumed)
        self._emit(EVENT_PROGRESS, GraphBuildProgressEvent(stage="index", total=indexed))
        stop = threading.Event()
        sampler = threading.Thread(
            target=self._sample_statuses,
            args=(stop,),
            name=f"graph-sample-{run.run_id}",
            daemon=True,
        )
        sampler.start()
        try:
            return self._flow(self.service, bounded, prune_removed=not run.bounded, verbose=0)
        finally:
            stop.set()
            sampler.join(timeout=_SAMPLE_INTERVAL_S * 2)

    def _scan(self, run: _RunState) -> list[MessageBatch]:
        """Parse every claimable file under ``GRAPH_DOCS_PATH``.

        A file that raises is counted and skipped rather than failing the
        build: one corrupt export must not cost the rest of an archive.

        Args:
            run: The run to update (counters plus the cached parse
                summaries the corpus listing reads).

        Returns:
            One batch per parsed file, in scan order.
        """
        root = Path(get_settings().GRAPH_DOCS_PATH)
        paths = scan_graph_corpus(root)
        with self._lock:
            run.files_scanned = len(paths)
            self._scanned = {}
        self._emit(EVENT_PROGRESS, GraphBuildProgressEvent(stage="scan", total=len(paths)))
        batches: list[MessageBatch] = []
        for index, path in enumerate(paths, start=1):
            if find_message_source(path) is None:
                continue  # sidecars, contacts files — listed, never parsed
            self._emit(
                EVENT_PROGRESS,
                GraphBuildProgressEvent(
                    stage="parse", file=path.name, current=index, total=len(paths)
                ),
            )
            try:
                batch = batch_for_path(path, root)
            except Exception as error:
                logger.warning("could not parse graph source %s: %s", path, error)
                with self._lock:
                    run.files_failed += 1
                continue
            batches.append(batch)
            scanned = ScannedDocument(
                relative_path=batch.relative_path,
                doc_id=batch.doc_id,
                parse=parse_summary(batch),
            )
            with self._lock:
                run.files_parsed += 1
                run.messages_parsed += len(batch.messages)
                self._scanned[batch.relative_path] = scanned
        return batches

    def _reset_workdir(self) -> None:
        """Drop the engine's working directory so the build starts clean.

        The session is closed first (the engine holds open handles on the
        files about to be removed) and reopened straight after, which both
        recreates the directory and keeps the window in which a concurrent
        *read* could see a half-removed workdir as short as an open takes.
        Concurrent writes cannot race this: the runner is single-flight and
        the service's write lock is the second gate.
        """
        workdir = self.service.workdir
        logger.info("reingest: wiping the graph working directory %s", workdir)
        self._emit(EVENT_PROGRESS, GraphBuildProgressEvent(stage="reset"))
        self.service.close()
        shutil.rmtree(workdir, ignore_errors=True)
        self.service.session()

    def _document_statuses(self) -> dict[str, int]:
        """Read the engine's document counts, tolerating an unopenable graph.

        Returns:
            Status name → document count; empty when the engine would not
            say (which a progress reader treats as "no news", never as
            "zero documents").
        """
        try:
            return self.service.document_statuses()
        except Exception:
            logger.debug("graph document statuses unavailable", exc_info=True)
            return {}

    def _inflight_documents(self) -> int:
        """Count documents a previous build left owing work.

        Returns:
            How many documents this run inherits — the ``resumed`` number
            that makes a kill/restart visible rather than mysterious.
        """
        statuses = self._document_statuses()
        return sum(count for name, count in statuses.items() if name.lower() in _INFLIGHT_STATUSES)

    def _sample_statuses(self, stop: threading.Event) -> None:
        """Emit engine document-status progress until the build finishes.

        Args:
            stop: Set by the build thread when the engine call returns.
        """
        while not stop.wait(_SAMPLE_INTERVAL_S):
            statuses = self._document_statuses()
            if not statuses:
                continue
            self._emit(
                EVENT_PROGRESS,
                GraphBuildProgressEvent(
                    stage="process",
                    docs_done=statuses.get("processed", 0),
                    docs_total=sum(statuses.values()),
                ),
            )

    def _summary(self, run: _RunState, report: BuildReport | None) -> GraphBuildSummaryOut:
        """Fold the run's counters and the engine's report into the wire summary.

        Args:
            run: The finished run.
            report: The engine's build report, or ``None`` when the run
                failed before the engine returned.

        Returns:
            The summary. A failed run still reports what it scanned, which
            is what makes "it never found the file" diagnosable.
        """
        return GraphBuildSummaryOut(
            files_scanned=run.files_scanned,
            files_parsed=run.files_parsed,
            files_failed=run.files_failed,
            messages_parsed=run.messages_parsed,
            messages_indexed=run.messages_indexed,
            messages_seen=report.messages_seen if report is not None else 0,
            wall_clock_s=report.wall_clock_s if report is not None else 0.0,
            resumed=run.resumed,
            documents=self._document_statuses(),
            failures=list(report.failures) if report is not None else [],
        )


_default_runner: GraphBuildRunner | None = None
_default_runner_lock = threading.Lock()


def get_graph_build_runner() -> GraphBuildRunner:
    """Provide the process-wide graph build runner (FastAPI dependency, override seam).

    Returns:
        The lazily created singleton. Construction resolves nothing — the
        service (and through it the engine registry) is looked up on the
        first build.
    """
    global _default_runner
    with _default_runner_lock:
        if _default_runner is None:
            _default_runner = GraphBuildRunner()
        return _default_runner
