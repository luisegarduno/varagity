"""The process-wide graph engine handle (spec_graphrag §5.2; ADR-017).

One process, one session, one writer. The engine's storages are
single-writer per working directory by explicit invariant, so the graph
corpus can only stay coherent if exactly one process ever opens it — which
is why builds are API-only and there is deliberately no CLI graph build
(stage-2 decision #4). This module is that single owner: a lazily-opened
session behind a process singleton, in the shape
:func:`varagity.api.ingest_runner.get_ingest_runner` established.

Two locks, doing different jobs:

* **A build lock**, non-reentrant and single-flight — the *write* lock. A
  second build attempt raises :class:`GraphBuildInProgress` rather than
  queueing, because the caller (an HTTP route) wants a ``409``, not a
  request that blocks for a day. Deletes take it too — they mutate the same
  graph — and so do a backup export and a backup restore (ADR-019), through
  :meth:`GraphService.writing`: holding it is what guarantees no file changes
  while the workdir is copied, and no build starts while it is swapped.
* **An open lock**, so two concurrent first-callers cannot open two sessions
  over the same directory.

**Reads deliberately take neither.** The session multiplexes its engine
calls onto a loop thread, so queries, exports, and status polls are answered
*while a multi-day backfill is still extracting* (stage-2 decision #10).
That is the whole reason the session's threading model changed at stage 2,
and the reason the graph corpus is usable during its own backfill.

**A backup restore swaps the whole workdir** (ADR-019):
:meth:`GraphService.replace_workdir` moves a restored copy into place under
*both* locks — the write lock, so no build can start, and the open lock, so
no lazy open (a status poll, a ``/graph`` read) can land on a half-swapped
folder — and puts the previous graph back if the new one won't open.
Constructing the service finishes a swap that a crash interrupted, before
anything in the process can read the workdir.

Nothing here raises on a broken engine except :class:`GraphUnavailable`,
which callers turn into a structured degrade (ADR-017's degrade semantics) —
a graph the app cannot open must cost the graph turn, never the process.
"""

import logging
import os
import re
import shutil
import threading
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path

from varagity.config import get_settings
from varagity.graph.base import GraphSession, get_graph_engine
from varagity.graph.records import (
    BuildReport,
    EngineFormat,
    GraphAnswer,
    GraphExport,
    GraphRetrieval,
    GraphStats,
)
from varagity.graph.sources.base import MessageBatch
from varagity.paths import fsync_directory

logger = logging.getLogger(__name__)

# The folders a backup restore parks beside the workdir while it swaps
# (ADR-019), each ``<workdir>.<kind>-<run id>`` — same parent, so every rename
# between them is atomic:
#
# * ``restore``  — the bundle, extracted (perhaps partly) for the swap;
# * ``previous`` — the live graph, moved aside until the restored one opens;
# * ``replaced`` — the previous graph once the restored one has opened — the
#   rename into this name is the swap's commit point, after which the
#   restored graph stands, crash or no crash.
_STAGED = "restore"
_PREVIOUS = "previous"
_REPLACED = "replaced"

# A run id may become part of a folder name, so it is held to a safe shape.
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def graph_workdir(engine_name: str | None = None) -> Path:
    """Locate the working directory an engine stores this graph in.

    The one derivation of "where the graph lives", shared by the service
    and by anything that must read the workdir *without* going through the
    engine — the Prometheus gauges, which resolve a path at scrape time and
    must not depend on the engine registry being warm
    (:mod:`varagity.observability.graph`).

    Args:
        engine_name: Registry name of the engine; ``None`` reads
            ``GRAPH_ENGINE``.

    Returns:
        The absolute directory (relative workdirs have bitten this repo
        before — an engine consumes the path verbatim, from whatever cwd it
        happens to run in). It may not exist yet: nothing is created here.
    """
    settings = get_settings()
    name = engine_name if engine_name is not None else settings.GRAPH_ENGINE
    return (Path(settings.GRAPH_STORAGE_PATH) / name).resolve()


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
        workdir in every process that merely imports this module. The one
        thing construction does on disk is finish a backup restore's swap
        that a crash interrupted (:meth:`_heal_interrupted_swap`) — first,
        before anything in this process can read, open, wipe or restore the
        workdir.

        Args:
            engine_name: Registry name of the engine; ``None`` reads
                ``GRAPH_ENGINE``.
            workdir: The engine's working directory; ``None`` derives
                :func:`graph_workdir` (``GRAPH_STORAGE_PATH/<engine>``,
                resolved absolute).

        Raises:
            KeyError: If ``engine_name`` names no registered engine.
        """
        self._engine_name = engine_name if engine_name is not None else get_settings().GRAPH_ENGINE
        self._engine = get_graph_engine(self._engine_name)
        self._workdir = workdir if workdir is not None else graph_workdir(self._engine_name)
        self._stack: ExitStack | None = None
        self._session: GraphSession | None = None
        self._open_lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._heal_interrupted_swap()

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
        """Whether a write — a build, resume, delete, or backup export or restore — holds it.

        Returns:
            ``True`` while one is in flight.
        """
        return self._build_lock.locked()

    def storage_format(self) -> EngineFormat:
        """Name the format the engine's workdir files are in (what a backup records).

        Returns:
            The engine's :meth:`~varagity.graph.base.GraphEngine.storage_format`
            — read from package metadata, so this opens no session.
        """
        return self._engine.storage_format()

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
        with self.writing():
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
        with self.writing():
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
        with self.writing():
            return self.session().delete_documents(doc_keys)

    def retrieve(
        self, question: str, *, mode: str | None = None, verbose: int = 0
    ) -> GraphRetrieval:
        """Find evidence for one question — never blocked by a build.

        The chat path's read: the API streams its own grounded answer over
        what this returns, so no engine answer call is spent here.

        Args:
            question: The search query, verbatim.
            mode: Engine query mode; ``None`` uses the adapter's primary.
            verbose: Validated console verbosity (0–2).

        Returns:
            The evidence and transcript excerpts.

        Raises:
            GraphUnavailable: If the session cannot be opened (the caller
                turns that into ADR-017's per-turn degrade).
        """
        return self.session().retrieve(question, mode=mode, verbose=verbose)

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
        max_nodes: int = 5000,
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
        way, and the next call opens a fresh one. The teardown runs under
        the open lock, so a concurrent first caller waits for it instead of
        opening a second session over files the first is still flushing.
        """
        with self._open_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        """Tear the session down; the caller holds the open lock.

        Closing finalizes the engine's storages (flushing what they buffer
        into the workdir) and resets the engine library's process-global
        state, so the next open reads the workdir from disk rather than
        re-attaching to the old session's memory.
        """
        stack, self._stack, self._session = self._stack, None, None
        if stack is None:
            return
        try:
            stack.close()
        except Exception:
            logger.warning("graph session teardown failed", exc_info=True)

    def staging_dir(self, run_id: str) -> Path:
        """Name the folder a restore extracts a bundle into before the swap.

        ``<workdir>.restore-<run>``: beside the workdir, so on the same
        filesystem — every rename of :meth:`replace_workdir` stays atomic.
        Nothing is created here.

        Args:
            run_id: The restore run's id.

        Returns:
            The staging folder's path.

        Raises:
            ValueError: If ``run_id`` isn't a safe folder-name suffix.
        """
        return self._swap_dir(_STAGED, run_id)

    def replace_workdir(
        self,
        staged: Path,
        run_id: str,
        *,
        on_open: Callable[[], None] | None = None,
    ) -> GraphStats:
        """Swap a restored working directory in for the live one (a graph restore, ADR-019).

        ``_reset_workdir``'s close → change → reopen move, made safe to lose.
        Under the **write lock** throughout (no build, resume, delete or
        export can start), and under the **open lock** for the renames (no
        lazy open — a status poll, a ``/graph`` read — can land on a
        half-swapped folder):

        1. close the session — the engine flushes what it buffers into the
           *old* files, and its process-global state is reset so the next
           open reads the new ones from disk;
        2. move the live workdir aside to ``<workdir>.previous-<run>`` (when
           there is one), then ``staged`` into its place;
        3. open the restored graph and read its size — opening is what
           checks the files (LightRAG asserts its vectors' ``embedding_dim``);
        4. rename the previous graph to ``<workdir>.replaced-<run>`` — the
           **commit point**: until it, any failure (or a crash, which the
           next process's construction heals) brings the previous graph
           back; after it, the restored graph stands and the replaced one is
           deleted.

        A failure before the commit undoes the swap: the restored folder
        goes back to ``staged`` (its owner removes it), the previous graph
        returns and its session is reopened, and the error is re-raised.

        Reads stay unlocked outside the renames, so a graph turn already
        streaming when the session closes can end in an error (chat refuses
        *new* graph turns for the whole restore, before they open one).

        Args:
            staged: The restored workdir, complete — at :meth:`staging_dir`,
                so the renames stay on one filesystem.
            run_id: The restore run's id (names the folders beside the
                workdir).
            on_open: Called once the restored folder is in place, just before
                it is opened — the restore's ``open`` stage.

        Returns:
            The restored graph's size, as its freshly opened session reports
            it.

        Raises:
            GraphBuildInProgress: If a write holds the write lock (nothing
                was moved).
            GraphUnavailable: If the restored graph would not open (the
                previous one is back in place).
            OSError: If a rename fails (whatever had moved is moved back
                first).
        """
        previous = self._swap_dir(_PREVIOUS, run_id)
        replaced = self._swap_dir(_REPLACED, run_id)
        with self.writing():
            had_previous = self._swap_in(staged, previous)
            try:
                if on_open is not None:
                    on_open()
                stats = self.session().stats()
                if had_previous:
                    os.replace(previous, replaced)  # the commit point
            except Exception:
                self._swap_back(staged, previous, had_previous=had_previous)
                raise
            fsync_directory(self._workdir.parent)
            shutil.rmtree(replaced, ignore_errors=True)
            if replaced.exists(follow_symlinks=False):
                logger.warning(
                    "could not delete the replaced graph (%s) — the next API start removes it",
                    replaced.name,
                )
        logger.info("swapped a restored graph into %s", self._workdir.name)
        return stats

    def _swap_dir(self, kind: str, run_id: str) -> Path:
        """Name one of a restore's folders beside the workdir.

        Args:
            kind: ``restore`` | ``previous`` | ``replaced``.
            run_id: The restore run's id.

        Returns:
            ``<workdir>.<kind>-<run_id>``.

        Raises:
            ValueError: If ``run_id`` isn't a safe folder-name suffix.
        """
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError(f"{run_id!r} can't name a folder beside the graph workdir")
        return self._workdir.with_name(f"{self._workdir.name}.{kind}-{run_id}")

    def _swap_in(self, staged: Path, previous: Path) -> bool:
        """Close the session and move the restored folder into the workdir's place.

        Both renames run under the open lock. A failed second rename puts
        the live workdir back before re-raising.

        Args:
            staged: The restored workdir.
            previous: Where the live workdir moves aside to.

        Returns:
            Whether there was a live workdir to move aside.

        Raises:
            OSError: If a rename fails.
        """
        fsync_directory(staged)  # the extracted names are durable before they go live
        with self._open_lock:
            self._close_locked()
            had_previous = self._workdir.exists(follow_symlinks=False)
            if had_previous:
                os.replace(self._workdir, previous)
            try:
                os.replace(staged, self._workdir)
            except OSError:
                if had_previous:
                    os.replace(previous, self._workdir)
                raise
        fsync_directory(self._workdir.parent)
        return had_previous

    def _swap_back(self, staged: Path, previous: Path, *, had_previous: bool) -> None:
        """Undo a swap whose restored graph would not open (or not commit).

        The restored folder returns to ``staged`` (its owner deletes it),
        the previous graph returns to the workdir, and — when there was one
        — its session is reopened. Never raises: the caller is already
        re-raising the failure that got here, and a rename that fails now
        leaves ``<workdir>.previous-<run>`` for the next process's
        construction to move back.

        Args:
            staged: Where the restored folder goes back to.
            previous: Where the previous graph waits.
            had_previous: Whether there was a previous graph at all (a
                restore onto a wiped stack has none to reopen).
        """
        with self._open_lock:
            self._close_locked()
            try:
                if self._workdir.exists(follow_symlinks=False):
                    os.replace(self._workdir, staged)
                if had_previous:
                    os.replace(previous, self._workdir)
            except OSError:
                logger.exception(
                    "could not put the previous graph back after a failed restore — the next "
                    "API start moves %s back into place",
                    previous.name,
                )
                return
        if not had_previous:
            logger.warning("the restored graph would not open — the graph is left unbuilt")
            return
        logger.warning("the restored graph would not open — the previous graph is back in place")
        try:
            self.session()
        except GraphUnavailable as error:
            logger.warning("the previous graph would not reopen either: %s", error)

    def _heal_interrupted_swap(self) -> None:
        """Finish a restore's swap that a crash interrupted (best-effort; at construction).

        A restore parks folders beside the workdir while it swaps (see
        :meth:`replace_workdir`), and a crash can leave any of them behind:

        * **a previous graph** (``.previous-<run>``) — the swap never reached
          its commit point, so the restore did not happen: whatever stands
          at the workdir (an unconfirmed restored graph, or nothing — the
          crash came between the renames) is removed and the previous graph
          moves back;
        * **a replaced graph** (``.replaced-<run>``) — the swap committed and
          the crash came mid-delete: it is removed, and the restored graph
          stands;
        * **a staging folder** (``.restore-<run>``) — a bundle extracted
          (perhaps partly) for a swap that never committed: it is removed.

        Run at construction — before anything in this process can read,
        open, wipe or restore the workdir — so a missing workdir can only
        mean a crash (never this process's own reingest wipe), and no
        restore of this process can have a folder in flight yet. Failures
        are logged, never raised: a graph that can't be healed must not cost
        the API its graph service; the next start tries again.
        """
        try:
            previous = self._swap_leftovers(_PREVIOUS)
            if previous:
                if self._workdir.exists(follow_symlinks=False):
                    shutil.rmtree(self._workdir)
                os.replace(previous[0], self._workdir)
                logger.warning(
                    "a graph restore was interrupted before it committed — %s is back in place",
                    self._workdir.name,
                )
            leftovers = [
                *previous[1:],
                *self._swap_leftovers(_REPLACED),
                *self._swap_leftovers(_STAGED),
            ]
            for leftover in leftovers:
                shutil.rmtree(leftover, ignore_errors=True)
                logger.info("removed %s, left by an interrupted graph restore", leftover.name)
        except OSError:
            logger.warning(
                "could not finish an interrupted graph restore beside %s",
                self._workdir.name,
                exc_info=True,
            )

    def _swap_leftovers(self, kind: str) -> list[Path]:
        """Find the folders of one kind a restore left beside the workdir.

        Args:
            kind: ``restore`` | ``previous`` | ``replaced``.

        Returns:
            The folders, most recently modified first (empty when the
            workdir's parent doesn't exist yet).

        Raises:
            OSError: If the parent can't be listed.
        """
        parent = self._workdir.parent
        if not parent.is_dir():
            return []
        prefix = f"{self._workdir.name}.{kind}-"
        found = [
            entry
            for entry in parent.iterdir()
            if entry.name.startswith(prefix)
            and _RUN_ID.fullmatch(entry.name[len(prefix) :]) is not None
            and entry.is_dir()
            and not entry.is_symlink()
        ]
        return sorted(found, key=lambda entry: entry.stat().st_mtime_ns, reverse=True)

    def writing(self) -> "_WriteLock":
        """Take the single-flight write lock for the duration of a block.

        The lock :meth:`build`, :meth:`resume` and :meth:`delete_documents`
        take, public so a caller that must see the workdir hold still can
        take it too — a backup export copies the files under it (ADR-019),
        so no build, resume or delete can start mid-copy, while reads stay
        unlocked and chat keeps answering (:meth:`replace_workdir` swaps a
        restored workdir in under it). Never blocks: a held lock is an
        answer (the caller's ``409``), not a wait.

        Returns:
            A context manager holding the lock (released on block exit,
            however the block ends).

        Raises:
            GraphBuildInProgress: If it is already held.
        """
        if not self._build_lock.acquire(blocking=False):
            raise GraphBuildInProgress(
                "the graph is being written — a build, a delete, or a backup export or restore "
                "holds its write lock"
            )
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
