"""The process-wide graph engine handle (spec_graphrag §5.2; ADR-017).

One process, one session, one writer. The engine's storages are
single-writer per working directory by explicit invariant, so the graph
corpus can only stay coherent if exactly one process ever opens it — which
is why builds are API-only and there is deliberately no CLI graph build
(stage-2 decision #4). This module is that single owner: a lazily-opened
session behind a process singleton, in the shape
:func:`varagity.api.ingest_runner.get_ingest_runner` established.

Two locks, doing different jobs:

* **A build lock**, non-reentrant and single-flight. A second build attempt
  raises :class:`GraphBuildInProgress` rather than queueing, because the
  caller (an HTTP route) wants a ``409``, not a request that blocks for a
  day. Deletes take it too — they mutate the same graph.
* **An open lock**, so two concurrent first-callers cannot open two sessions
  over the same directory.

**Reads deliberately take neither.** The session multiplexes its engine
calls onto a loop thread, so queries, exports, and status polls are answered
*while a multi-day backfill is still extracting* (stage-2 decision #10).
That is the whole reason the session's threading model changed at stage 2,
and the reason the graph corpus is usable during its own backfill.

Nothing here raises on a broken engine except :class:`GraphUnavailable`,
which callers turn into a structured degrade (ADR-017's degrade semantics) —
a graph the app cannot open must cost the graph turn, never the process.
"""

import logging
import threading
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from varagity.config import get_settings
from varagity.graph.base import GraphSession, get_graph_engine
from varagity.graph.records import BuildReport, GraphAnswer, GraphExport, GraphStats
from varagity.graph.sources.base import MessageBatch

logger = logging.getLogger(__name__)


class GraphBuildInProgress(RuntimeError):
    """Raised when a build is requested while one is already running."""


class GraphUnavailable(RuntimeError):
    """Raised when the graph engine's session cannot be opened."""


class GraphService:
    """Owner of the process's single graph session (see the module docstring)."""

    def __init__(self, engine_name: str | None = None, *, workdir: Path | None = None) -> None:
        """Resolve the engine and its working directory (both injectable).

        Nothing is opened here: a session that opened at import time would
        make an unbuilt graph an API startup failure, and would open the
        workdir in every process that merely imports this module.

        Args:
            engine_name: Registry name of the engine; ``None`` reads
                ``GRAPH_ENGINE``.
            workdir: The engine's working directory; ``None`` derives
                ``GRAPH_STORAGE_PATH/<engine>``, resolved absolute (relative
                workdirs have bitten this repo before — an engine consumes
                the path verbatim, from whatever cwd it happens to run in).

        Raises:
            KeyError: If ``engine_name`` names no registered engine.
        """
        settings = get_settings()
        self._engine_name = engine_name if engine_name is not None else settings.GRAPH_ENGINE
        self._engine = get_graph_engine(self._engine_name)
        self._workdir = (
            workdir
            if workdir is not None
            else (Path(settings.GRAPH_STORAGE_PATH) / self._engine_name).resolve()
        )
        self._stack: ExitStack | None = None
        self._session: GraphSession | None = None
        self._open_lock = threading.Lock()
        self._build_lock = threading.Lock()

    @property
    def engine_name(self) -> str:
        """Name of the engine this service resolved.

        Returns:
            The registry key (``GRAPH_ENGINE``).
        """
        return self._engine_name

    @property
    def workdir(self) -> Path:
        """Where the engine stores this process's graph.

        Returns:
            The absolute working directory.
        """
        return self._workdir

    @property
    def building(self) -> bool:
        """Whether a build or delete currently holds the write lock.

        Returns:
            ``True`` while one is in flight.
        """
        return self._build_lock.locked()

    def session(self) -> GraphSession:
        """Return the open session, opening it on first use.

        Returns:
            The process's one session.

        Raises:
            GraphUnavailable: If the engine's session cannot be opened. The
                failure is not cached — the next call tries again, so a
                transient one (a model service still warming up) does not
                need a restart to clear.
        """
        with self._open_lock:
            if self._session is not None:
                return self._session
            logger.info("opening the %s graph session in %s", self._engine_name, self._workdir)
            stack = ExitStack()
            try:
                session = stack.enter_context(self._engine.session(self._workdir))
            except Exception as exc:
                stack.close()
                raise GraphUnavailable(
                    f"could not open the {self._engine_name!r} graph session "
                    f"in {self._workdir}: {exc}"
                ) from exc
            self._stack, self._session = stack, session
            return session

    def build(
        self,
        batches: Sequence[MessageBatch],
        *,
        verbose: int = 0,
        prune_removed: bool = True,
    ) -> BuildReport:
        """Upsert a rendered corpus into the graph, one build at a time.

        Args:
            batches: Parsed source files.
            verbose: Validated console verbosity (0–2).
            prune_removed: Whether ``batches`` render the whole corpus (see
                :meth:`varagity.graph.base.GraphSession.build`).

        Returns:
            What the build did.

        Raises:
            GraphBuildInProgress: If another build or delete is running.
            GraphUnavailable: If the session cannot be opened.
        """
        with self._writing():
            return self.session().build(batches, verbose=verbose, prune_removed=prune_removed)

    def resume(self, *, verbose: int = 0) -> BuildReport:
        """Finish whatever a killed build left in flight.

        Args:
            verbose: Validated console verbosity (0–2).

        Returns:
            What the pass did.

        Raises:
            GraphBuildInProgress: If another build or delete is running.
            GraphUnavailable: If the session cannot be opened.
        """
        with self._writing():
            return self.session().resume(verbose=verbose)

    def delete_documents(self, doc_keys: Sequence[str]) -> int:
        """Retract documents from the graph.

        Args:
            doc_keys: Transcript document keys to remove.

        Returns:
            How many deletions the engine accepted.

        Raises:
            GraphBuildInProgress: If a build is running — deletion rebuilds
                the entities a document partly supported, which may not race
                extraction.
            GraphUnavailable: If the session cannot be opened.
        """
        with self._writing():
            return self.session().delete_documents(doc_keys)

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        """Answer one question from the graph — never blocked by a build.

        Args:
            question: The question, verbatim.
            mode: Engine query mode; ``None`` uses the adapter's primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The answer with its evidence.

        Raises:
            GraphUnavailable: If the session cannot be opened.
        """
        return self.session().query(question, mode=mode, verbose=verbose)

    def stats(self) -> GraphStats:
        """Report the graph's size.

        Returns:
            Entity/relation/community counts, each ``None`` where unknown.

        Raises:
            GraphUnavailable: If the session cannot be opened.
        """
        return self.session().stats()

    def document_statuses(self) -> dict[str, int]:
        """Count the engine's documents by processing status.

        Returns:
            Status name → document count, empty when the engine would not say.

        Raises:
            GraphUnavailable: If the session cannot be opened.
        """
        return self.session().document_statuses()

    def export(
        self,
        label: str = "*",
        *,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> GraphExport:
        """Read a renderable slice of the graph.

        Args:
            label: Entity name to centre on; ``"*"`` is the whole graph.
            max_depth: Hops to walk out from ``label``.
            max_nodes: Node cap.

        Returns:
            The slice.

        Raises:
            GraphUnavailable: If the session cannot be opened.
        """
        return self.session().export(label, max_depth=max_depth, max_nodes=max_nodes)

    def close(self) -> None:
        """Tear the session down (API lifespan shutdown); idempotent.

        A teardown that raised would take an orderly shutdown with it, so
        failures are logged and swallowed — the session is dropped either
        way, and the next call opens a fresh one.
        """
        with self._open_lock:
            stack, self._stack, self._session = self._stack, None, None
        if stack is None:
            return
        try:
            stack.close()
        except Exception:
            logger.warning("graph session teardown failed", exc_info=True)

    def _writing(self) -> "_WriteLock":
        """Take the single-flight write lock for the duration of a block.

        Returns:
            A context manager holding the lock.

        Raises:
            GraphBuildInProgress: If it is already held.
        """
        if not self._build_lock.acquire(blocking=False):
            raise GraphBuildInProgress("a graph build is already running")
        return _WriteLock(self._build_lock)


class _WriteLock:
    """Releases :class:`GraphService`'s write lock on block exit."""

    def __init__(self, lock: threading.Lock) -> None:
        """Adopt an already-acquired lock.

        Args:
            lock: The held lock to release on exit.
        """
        self._lock = lock

    def __enter__(self) -> None:
        """Enter the guarded block (the lock is already held)."""

    def __exit__(self, *exc_info: object) -> None:
        """Release the lock.

        Args:
            *exc_info: Ignored — the lock is released either way.
        """
        self._lock.release()


_default_service: GraphService | None = None
_default_service_lock = threading.Lock()


def get_graph_service() -> GraphService:
    """Provide the process-wide graph service (FastAPI dependency, override seam).

    Returns:
        The lazily created singleton. Construction only resolves settings and
        the registry — the engine session opens on first real use.
    """
    global _default_service
    with _default_service_lock:
        if _default_service is None:
            _default_service = GraphService()
        return _default_service
