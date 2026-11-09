"""Unit tests for the process-wide graph service.

Driven entirely by a fake engine: what is under test is not LightRAG but the
*ownership* rules the API depends on — one session per process, one build at
a time, reads that a build cannot block, and a broken engine that degrades a
turn instead of taking the process down.
"""

import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from varagity.config import get_settings
from varagity.graph import service as graph_service
from varagity.graph.base import GRAPH_ENGINE_REGISTRY
from varagity.graph.records import (
    BuildReport,
    EngineFormat,
    GraphAnswer,
    GraphEvidence,
    GraphExport,
    GraphExportNode,
    GraphRetrieval,
    GraphStats,
)
from varagity.graph.service import (
    GraphBuildInProgress,
    GraphService,
    GraphUnavailable,
    get_graph_service,
)
from varagity.graph.sources.base import MessageBatch


class FakeSession:
    """A graph session that records what it was asked to do."""

    def __init__(
        self,
        *,
        before_build: threading.Event | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.before_build = before_build
        self.on_close = on_close
        self.release = threading.Event()
        self.closed = False
        self.builds: list[bool] = []
        self.resumes = 0
        self.queries: list[tuple[str, str | None]] = []
        self.retrievals: list[tuple[str, str | None]] = []
        self.deleted: list[str] = []
        self.exports: list[tuple[str, int, int]] = []

    def build(
        self,
        batches: Sequence[MessageBatch],
        *,
        verbose: int = 0,
        prune_removed: bool = True,
    ) -> BuildReport:
        if self.before_build is not None:
            self.before_build.set()
            self.release.wait(timeout=5)
        self.builds.append(prune_removed)
        return BuildReport(messages_seen=len(batches), wall_clock_s=0.0)

    def resume(self, *, verbose: int = 0) -> BuildReport:
        self.resumes += 1
        return BuildReport(messages_seen=3, wall_clock_s=0.0)

    def retrieve(
        self, question: str, *, mode: str | None = None, verbose: int = 0
    ) -> GraphRetrieval:
        self.retrievals.append((question, mode))
        return GraphRetrieval(evidence=GraphEvidence(), mode=mode or "hybrid")

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        self.queries.append((question, mode))
        return GraphAnswer(
            answer="ok", evidence=GraphEvidence(), mode=mode or "hybrid", latency_s=0.0
        )

    def stats(self) -> GraphStats:
        return GraphStats(entities=4, relations=3, communities=None)

    def document_statuses(self) -> dict[str, int]:
        return {"processed": 9}

    def export(self, label: str = "*", *, max_depth: int = 3, max_nodes: int = 1000) -> GraphExport:
        self.exports.append((label, max_depth, max_nodes))
        return GraphExport(nodes=[GraphExportNode(id="Bob")])

    def delete_documents(self, doc_keys: Sequence[str]) -> int:
        self.deleted.extend(doc_keys)
        return len(doc_keys)

    def close(self) -> None:
        if self.on_close is not None:
            self.on_close()
        self.closed = True


FAKE_FORMAT = EngineFormat(name="lightrag", package="lightrag-hku", version="1.5.4")
"""What :class:`FakeEngine` says its files are (the shipped engine's shape)."""


class FakeEngine:
    """Hands out :class:`FakeSession`s, or refuses to."""

    def __init__(
        self,
        *,
        fail_open: bool = False,
        fail_close: bool = False,
        storage: EngineFormat = FAKE_FORMAT,
        on_open: Callable[[Path], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.storage = storage
        # A test's hook into every open: record what was opened, or raise to
        # refuse it (the way a real engine refuses files it can't load).
        self.on_open = on_open
        self.session_kwargs = kwargs
        self.opens = 0
        self.workdirs: list[Path] = []
        self.sessions: list[FakeSession] = []

    def storage_format(self) -> EngineFormat:
        return self.storage

    @contextmanager
    def session(self, workdir: Path) -> Iterator[FakeSession]:
        self.opens += 1
        self.workdirs.append(workdir)
        if self.fail_open:
            raise RuntimeError("engine storage is locked")
        if self.on_open is not None:
            self.on_open(workdir)
        session = FakeSession(**self.session_kwargs)
        self.sessions.append(session)
        try:
            yield session
        finally:
            session.close()
            if self.fail_close:
                raise RuntimeError("teardown exploded")


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Register a fake engine in the registry and hand it back.

    Settings-driven tests must register under a name ``GRAPH_ENGINE``
    actually accepts (the validator's tuple is hard-coded on purpose), so
    the default shadows the real adapter for the duration of the test.
    """

    def register(name: str = "lightrag", **kwargs: Any) -> FakeEngine:
        engine = FakeEngine(**kwargs)
        monkeypatch.setitem(GRAPH_ENGINE_REGISTRY, name, engine)
        return engine

    return register


def service(registered: Any, tmp_path: Path, **kwargs: Any) -> tuple[GraphService, FakeEngine]:
    engine = registered("_probe", **kwargs)
    return GraphService("_probe", workdir=tmp_path), engine


class TestResolution:
    def test_the_engine_and_workdir_come_from_settings(
        self, settings_env: Any, tmp_path: Path, registered: Any
    ) -> None:
        registered()
        settings_env(GRAPH_STORAGE_PATH=str(tmp_path / "root"))
        resolved = GraphService()
        assert resolved.engine_name == get_settings().GRAPH_ENGINE
        assert resolved.workdir == tmp_path / "root" / "lightrag"

    def test_the_workdir_is_absolute_however_it_was_configured(
        self, settings_env: Any, registered: Any
    ) -> None:
        """An engine consumes the path verbatim, from whatever cwd it runs in."""
        registered()
        settings_env(GRAPH_STORAGE_PATH="./graph-data")
        assert GraphService().workdir.is_absolute()

    def test_an_unregistered_engine_fails_loudly_at_construction(self) -> None:
        with pytest.raises(KeyError):
            GraphService("made_up")

    def test_construction_opens_nothing(self, registered: Any, tmp_path: Path) -> None:
        """An unbuilt graph must not be an API startup failure."""
        resolved, engine = service(registered, tmp_path)
        assert engine.opens == 0
        assert resolved.building is False


class TestSessionLifecycle:
    def test_the_session_opens_once_and_is_reused(self, registered: Any, tmp_path: Path) -> None:
        resolved, engine = service(registered, tmp_path)
        assert resolved.session() is resolved.session()
        assert engine.opens == 1
        assert engine.workdirs == [tmp_path]
        resolved.close()

    def test_concurrent_first_callers_still_open_one_session(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """★ Two sessions over one workdir would break the single-writer rule."""
        resolved, engine = service(registered, tmp_path)
        start = threading.Barrier(4)
        seen: list[Any] = []

        def open_it() -> None:
            start.wait(timeout=5)
            seen.append(resolved.session())

        threads = [threading.Thread(target=open_it) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        resolved.close()
        assert engine.opens == 1
        assert len({id(session) for session in seen}) == 1

    def test_a_refusing_engine_is_a_structured_error_not_a_crash(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, _ = service(registered, tmp_path, fail_open=True)
        with pytest.raises(GraphUnavailable, match="storage is locked"):
            resolved.session()

    def test_an_open_failure_is_not_cached(self, registered: Any, tmp_path: Path) -> None:
        """A model service still warming up must not need a restart to clear."""
        resolved, engine = service(registered, tmp_path, fail_open=True)
        with pytest.raises(GraphUnavailable):
            resolved.session()
        engine.fail_open = False
        assert resolved.session() is not None
        resolved.close()

    def test_close_tears_the_session_down_and_the_next_call_reopens(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, engine = service(registered, tmp_path)
        first = resolved.session()
        resolved.close()
        assert first.closed is True
        assert resolved.session() is not first
        assert engine.opens == 2
        resolved.close()

    def test_close_is_idempotent_and_safe_before_any_open(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, engine = service(registered, tmp_path)
        resolved.close()
        resolved.close()
        assert engine.opens == 0

    def test_a_failed_teardown_never_takes_the_shutdown_with_it(
        self, registered: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        resolved, _ = service(registered, tmp_path, fail_close=True)
        resolved.session()
        with caplog.at_level("WARNING", logger="varagity.graph.service"):
            resolved.close()
        assert "teardown failed" in caplog.text


class TestWriteLock:
    def test_writes_pass_through_to_the_session(self, registered: Any, tmp_path: Path) -> None:
        resolved, engine = service(registered, tmp_path)
        assert resolved.build([], prune_removed=False).messages_seen == 0
        assert resolved.resume().messages_seen == 3
        assert resolved.delete_documents(["a::1"]) == 1
        resolved.close()
        (session,) = engine.sessions
        assert session.builds == [False]
        assert session.resumes == 1
        assert session.deleted == ["a::1"]

    def test_a_second_build_is_refused_rather_than_queued(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """★ The caller wants a 409, not a request that blocks for a day."""
        entered = threading.Event()
        resolved, engine = service(registered, tmp_path, before_build=entered)
        builder = threading.Thread(target=lambda: resolved.build([]))
        builder.start()
        try:
            assert entered.wait(timeout=5)
            assert resolved.building is True
            with pytest.raises(GraphBuildInProgress):
                resolved.build([])
            with pytest.raises(GraphBuildInProgress):
                resolved.resume()
            with pytest.raises(GraphBuildInProgress):
                resolved.delete_documents(["a::1"])
        finally:
            engine.sessions[0].release.set()
            builder.join(timeout=5)
        assert resolved.building is False
        resolved.close()

    def test_reads_are_answered_while_a_build_holds_the_lock(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """★ Decision #10: the graph stays usable during its own backfill."""
        entered = threading.Event()
        resolved, engine = service(registered, tmp_path, before_build=entered)
        resolved.session()  # open before the build parks inside it
        builder = threading.Thread(target=lambda: resolved.build([]))
        builder.start()
        try:
            assert entered.wait(timeout=5)
            assert resolved.query("who is Bob?", mode="mix").answer == "ok"
            # The chat path's read — a graph question mid-backfill is the
            # whole reason reads take no lock (decision #10).
            assert resolved.retrieve("who is Bob?", mode="mix").mode == "mix"
            assert resolved.stats() == GraphStats(entities=4, relations=3, communities=None)
            assert resolved.document_statuses() == {"processed": 9}
            assert [node.id for node in resolved.export("Bob", max_nodes=5).nodes] == ["Bob"]
        finally:
            engine.sessions[0].release.set()
            builder.join(timeout=5)
        resolved.close()

    def test_the_lock_is_released_when_a_build_raises(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, engine = service(registered, tmp_path)

        def boom(*args: Any, **kwargs: Any) -> BuildReport:
            raise RuntimeError("extraction exploded")

        resolved.session().build = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="extraction exploded"):
            resolved.build([])
        assert resolved.building is False
        assert resolved.resume().messages_seen == 3  # the lock is free again
        resolved.close()

    def test_reads_forward_their_arguments(self, registered: Any, tmp_path: Path) -> None:
        resolved, engine = service(registered, tmp_path)
        resolved.query("q", mode="global")
        resolved.retrieve("q", mode="mix")
        resolved.export("Bob", max_depth=1, max_nodes=25)
        resolved.close()
        (session,) = engine.sessions
        assert session.queries == [("q", "global")]
        assert session.retrievals == [("q", "mix")]
        assert session.exports == [("Bob", 1, 25)]

    def test_an_unopenable_engine_surfaces_as_a_degradable_error(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """The chat route turns this into ADR-017's per-turn degrade."""
        resolved, _ = service(registered, tmp_path, fail_open=True)
        with pytest.raises(GraphUnavailable):
            resolved.retrieve("q")

    def test_writing_is_the_lock_every_write_takes(self, registered: Any, tmp_path: Path) -> None:
        """★ A backup export holds it to keep the workdir still (ADR-019)."""
        resolved, engine = service(registered, tmp_path)
        with resolved.writing():
            assert resolved.building is True
            with pytest.raises(GraphBuildInProgress, match="holds its write lock"):
                resolved.build([])
            with pytest.raises(GraphBuildInProgress):
                resolved.resume()
            with pytest.raises(GraphBuildInProgress):
                resolved.delete_documents(["a::1"])
            with pytest.raises(GraphBuildInProgress):
                resolved.writing()  # never re-entrant, never a wait
            # Reads take no lock: chat answers while the files are copied.
            assert resolved.stats() == GraphStats(entities=4, relations=3, communities=None)
        assert resolved.building is False
        assert resolved.resume().messages_seen == 3  # released on exit
        resolved.close()
        (session,) = engine.sessions
        assert session.builds == [] and session.deleted == []

    def test_a_build_in_flight_refuses_the_lock_to_anyone_else(
        self, registered: Any, tmp_path: Path
    ) -> None:
        entered = threading.Event()
        resolved, engine = service(registered, tmp_path, before_build=entered)
        builder = threading.Thread(target=lambda: resolved.build([]))
        builder.start()
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(GraphBuildInProgress):
                resolved.writing()
        finally:
            engine.sessions[0].release.set()
            builder.join(timeout=5)
        with resolved.writing():
            pass  # free again once the build returned
        resolved.close()

    def test_the_lock_is_released_when_the_block_raises(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, _ = service(registered, tmp_path)
        with pytest.raises(OSError, match="disk full"), resolved.writing():
            raise OSError("disk full")
        assert resolved.building is False


class TestStorageFormat:
    def test_the_service_passes_the_engines_format_through_without_opening(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """A backup records it; reading it must never open a session."""
        resolved, engine = service(registered, tmp_path)
        assert resolved.storage_format() == FAKE_FORMAT
        assert engine.opens == 0

    def test_each_engine_reports_its_own_format(self, registered: Any, tmp_path: Path) -> None:
        other = EngineFormat(name="_probe", package="probe-graph", version="2.0.1")
        resolved, _ = service(registered, tmp_path, storage=other)
        assert resolved.storage_format() == other


class TestSingleton:
    def test_the_service_is_created_once_per_process(
        self, settings_env: Any, registered: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """★ One owner of the workdir — a second would break the single-writer rule."""
        registered()
        settings_env(GRAPH_STORAGE_PATH=str(tmp_path))
        monkeypatch.setattr(graph_service, "_default_service", None)
        first = get_graph_service()
        assert get_graph_service() is first
        assert first.workdir == tmp_path / "lightrag"


class TestCloseUnderTheOpenLock:
    def test_the_teardown_holds_the_open_lock(self, registered: Any, tmp_path: Path) -> None:
        """A concurrent first caller waits for the teardown — never a second session meanwhile."""
        held: list[bool] = []
        resolved, _ = service(
            registered, tmp_path, on_close=lambda: held.append(resolved._open_lock.locked())
        )
        resolved.session()
        resolved.close()
        assert held == [True]


# ── the restore's swap (ADR-019) ───────────────────────────────────────────

LABEL = "graph.txt"


def make_graph(path: Path, label: str) -> Path:
    """Lay out a stand-in workdir whose one file says which graph it is."""
    path.mkdir(parents=True, exist_ok=True)
    (path / LABEL).write_text(label)
    return path


def label_of(path: Path) -> str | None:
    marker = path / LABEL
    return marker.read_text() if marker.is_file() else None


class Opened:
    """Records which graph each open saw, and refuses the ones it is told to."""

    def __init__(self, refuse: frozenset[str] = frozenset()) -> None:
        self.labels: list[str | None] = []
        self.refuse = refuse

    def __call__(self, workdir: Path) -> None:
        label = label_of(workdir)
        self.labels.append(label)
        if label in self.refuse:
            raise RuntimeError(f"vdb embedding_dim mismatch in the {label} graph")


def swap_service(
    registered: Any, tmp_path: Path, *, refuse: frozenset[str] = frozenset(), **kwargs: Any
) -> tuple[GraphService, FakeEngine, Opened]:
    """A service over ``<tmp>/graph-data/lightrag`` whose engine records (or refuses) opens."""
    opened = Opened(refuse)
    engine = registered("_probe", on_open=opened, **kwargs)
    return GraphService("_probe", workdir=tmp_path / "graph-data" / "lightrag"), engine, opened


def beside(service: GraphService) -> list[str]:
    """Every name in the workdir's parent — the workdir and whatever a restore left."""
    return sorted(path.name for path in service.workdir.parent.iterdir())


class TestReplaceWorkdir:
    def test_the_restored_folder_takes_the_workdirs_place(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """★ Closed, swapped, reopened on the restored files — and nothing left beside it."""
        resolved, _, opened = swap_service(registered, tmp_path)
        make_graph(resolved.workdir, "live")
        first = resolved.session()
        staged = make_graph(resolved.staging_dir("r1"), "restored")

        stats = resolved.replace_workdir(staged, "r1")

        assert stats == GraphStats(entities=4, relations=3, communities=None)
        assert label_of(resolved.workdir) == "restored"
        assert first.closed is True  # the old session went first…
        assert opened.labels == ["live", "restored"]  # …and the new one opened on the new files
        assert beside(resolved) == ["lightrag"]  # staging, previous, replaced: all gone
        assert resolved.building is False
        resolved.close()

    def test_a_restore_onto_a_wiped_stack(self, registered: Any, tmp_path: Path) -> None:
        resolved, _, opened = swap_service(registered, tmp_path)
        staged = make_graph(resolved.staging_dir("r1"), "restored")
        resolved.replace_workdir(staged, "r1")
        assert label_of(resolved.workdir) == "restored"
        assert opened.labels == ["restored"]
        assert beside(resolved) == ["lightrag"]
        resolved.close()

    def test_the_renames_run_under_both_locks(
        self, registered: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """★ No build can start, and no lazy open can land on a half-swapped folder."""
        resolved, _, _ = swap_service(registered, tmp_path)
        make_graph(resolved.workdir, "live")
        resolved.session()
        staged = make_graph(resolved.staging_dir("r1"), "restored")
        seen: list[tuple[str, str, bool, bool]] = []
        real_replace = os.replace

        def recording(source: Any, target: Any) -> None:
            locks = (resolved.building, resolved._open_lock.locked())
            seen.append((Path(source).name, Path(target).name, *locks))
            real_replace(source, target)

        monkeypatch.setattr(graph_service.os, "replace", recording)
        resolved.replace_workdir(staged, "r1")
        resolved.close()
        assert seen == [
            ("lightrag", "lightrag.previous-r1", True, True),
            ("lightrag.restore-r1", "lightrag", True, True),
            # The commit point: after the restored graph opened, still under
            # the write lock (the open lock is the open's own business).
            ("lightrag.previous-r1", "lightrag.replaced-r1", True, False),
        ]

    def test_a_lazy_open_waits_out_the_swap_and_gets_the_restored_graph(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """★ A status poll mid-swap is held off, then opens the restored graph."""
        closing = threading.Event()
        release = threading.Event()

        def slow_close() -> None:
            closing.set()
            release.wait(timeout=5)

        resolved, _, opened = swap_service(registered, tmp_path, on_close=slow_close)
        make_graph(resolved.workdir, "live")
        resolved.session()
        staged = make_graph(resolved.staging_dir("r1"), "restored")
        swapper = threading.Thread(target=resolved.replace_workdir, args=(staged, "r1"))
        swapper.start()
        try:
            assert closing.wait(timeout=5)  # the swap holds the open lock, closing the old one
            polled: list[Any] = []
            poller = threading.Thread(target=lambda: polled.append(resolved.session()))
            poller.start()
            poller.join(timeout=0.2)
            assert poller.is_alive()  # the lazy open waits…
        finally:
            release.set()
        swapper.join(timeout=5)
        poller.join(timeout=5)
        assert not swapper.is_alive() and not poller.is_alive()
        assert opened.labels == ["live", "restored"]  # …and lands on the restored graph
        assert polled == [resolved.session()]
        resolved.close()

    def test_a_restored_graph_that_will_not_open_puts_the_previous_one_back(
        self, registered: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """★ The previous graph returns and reopens; the restored folder goes back to staging."""
        resolved, engine, opened = swap_service(registered, tmp_path, refuse=frozenset({"broken"}))
        make_graph(resolved.workdir, "live")
        resolved.session()
        staged = make_graph(resolved.staging_dir("r1"), "broken")

        with (
            caplog.at_level(logging.WARNING, logger="varagity.graph.service"),
            pytest.raises(GraphUnavailable, match="embedding_dim mismatch"),
        ):
            resolved.replace_workdir(staged, "r1")

        assert label_of(resolved.workdir) == "live"
        assert label_of(staged) == "broken"  # back where its owner deletes it
        assert opened.labels == ["live", "broken", "live"]  # reopened after the rollback
        assert resolved.session() is engine.sessions[-1]
        assert beside(resolved) == ["lightrag", "lightrag.restore-r1"]
        assert resolved.building is False
        assert "the previous graph is back in place" in caplog.text
        resolved.close()

    def test_a_wiped_stack_whose_restore_will_not_open_stays_unbuilt(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, _, opened = swap_service(registered, tmp_path, refuse=frozenset({"broken"}))
        staged = make_graph(resolved.staging_dir("r1"), "broken")
        with pytest.raises(GraphUnavailable):
            resolved.replace_workdir(staged, "r1")
        assert not resolved.workdir.exists()
        assert opened.labels == ["broken"]  # nothing to reopen
        assert label_of(staged) == "broken"

    def test_a_restored_graph_that_cannot_report_its_size_is_rolled_back_too(
        self, registered: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolved, engine, opened = swap_service(registered, tmp_path)
        make_graph(resolved.workdir, "live")
        staged = make_graph(resolved.staging_dir("r1"), "restored")

        def unreadable(self: FakeSession) -> GraphStats:
            raise RuntimeError("graphml unreadable")

        monkeypatch.setattr(FakeSession, "stats", unreadable)
        with pytest.raises(RuntimeError, match="graphml unreadable"):
            resolved.replace_workdir(staged, "r1")
        assert label_of(resolved.workdir) == "live"
        assert engine.sessions[0].closed is True  # the restored graph's session, before the move
        assert opened.labels == ["restored", "live"]
        resolved.close()

    def test_a_commit_that_fails_is_rolled_back(
        self, registered: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolved, _, _ = swap_service(registered, tmp_path)
        make_graph(resolved.workdir, "live")
        staged = make_graph(resolved.staging_dir("r1"), "restored")
        real_replace = os.replace

        def refuse_commit(source: Any, target: Any) -> None:
            if Path(target).name == "lightrag.replaced-r1":
                raise OSError("read-only file system")
            real_replace(source, target)

        monkeypatch.setattr(graph_service.os, "replace", refuse_commit)
        with pytest.raises(OSError, match="read-only"):
            resolved.replace_workdir(staged, "r1")
        assert label_of(resolved.workdir) == "live"
        assert label_of(staged) == "restored"
        resolved.close()

    def test_a_failed_rename_puts_the_live_workdir_back(
        self, registered: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolved, _, opened = swap_service(registered, tmp_path)
        make_graph(resolved.workdir, "live")
        staged = make_graph(resolved.staging_dir("r1"), "restored")
        real_replace = os.replace

        def refuse_swap(source: Any, target: Any) -> None:
            if Path(source).name == "lightrag.restore-r1":
                raise OSError("no space left on device")
            real_replace(source, target)

        monkeypatch.setattr(graph_service.os, "replace", refuse_swap)
        with pytest.raises(OSError, match="no space"):
            resolved.replace_workdir(staged, "r1")
        assert label_of(resolved.workdir) == "live"
        assert label_of(staged) == "restored"
        assert beside(resolved) == ["lightrag", "lightrag.restore-r1"]
        assert opened.labels == []  # nothing was opened

    def test_a_held_write_lock_refuses_the_swap_and_moves_nothing(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, _, opened = swap_service(registered, tmp_path)
        make_graph(resolved.workdir, "live")
        staged = make_graph(resolved.staging_dir("r1"), "restored")
        with resolved.writing(), pytest.raises(GraphBuildInProgress):
            resolved.replace_workdir(staged, "r1")
        assert label_of(resolved.workdir) == "live"
        assert label_of(staged) == "restored"
        assert opened.labels == []

    def test_on_open_fires_once_the_folder_is_in_place_before_it_opens(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, _, opened = swap_service(registered, tmp_path)
        make_graph(resolved.workdir, "live")
        staged = make_graph(resolved.staging_dir("r1"), "restored")
        seen: list[tuple[str | None, list[str | None]]] = []
        resolved.replace_workdir(
            staged,
            "r1",
            on_open=lambda: seen.append((label_of(resolved.workdir), list(opened.labels))),
        )
        assert seen == [("restored", [])]
        resolved.close()

    def test_the_staging_folder_sits_beside_the_workdir(
        self, registered: Any, tmp_path: Path
    ) -> None:
        resolved, _, _ = swap_service(registered, tmp_path)
        staging = resolved.staging_dir("8c41d07e9a2b")
        assert staging == resolved.workdir.with_name("lightrag.restore-8c41d07e9a2b")
        assert not staging.exists()  # named, never created here

    @pytest.mark.parametrize("run_id", ["", "../x", "a/b", "a b", "x" * 65])
    def test_a_run_id_that_cannot_name_a_folder_is_refused(
        self, registered: Any, tmp_path: Path, run_id: str
    ) -> None:
        resolved, _, _ = swap_service(registered, tmp_path)
        with pytest.raises(ValueError, match="can't name a folder"):
            resolved.staging_dir(run_id)
        with pytest.raises(ValueError, match="can't name a folder"):
            resolved.replace_workdir(tmp_path, run_id)


def construct(registered: Any, root: Path) -> tuple[GraphService, FakeEngine]:
    """Construct a service over ``<root>/lightrag`` — which is when it heals."""
    engine = registered("_probe")
    return GraphService("_probe", workdir=root / "lightrag"), engine


class TestHealAtConstruction:
    """A crash mid-swap is finished by the next process's construction (ADR-019)."""

    def test_a_crash_between_the_renames_brings_the_previous_graph_back(
        self, registered: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        root = tmp_path / "graph-data"
        make_graph(root / "lightrag.previous-r1", "live")
        make_graph(root / "lightrag.restore-r1", "restored")
        with caplog.at_level(logging.WARNING, logger="varagity.graph.service"):
            healed, engine = construct(registered, root)
        assert label_of(healed.workdir) == "live"
        assert beside(healed) == ["lightrag"]
        assert engine.opens == 0  # healing opens nothing
        assert "interrupted before it committed" in caplog.text

    def test_a_swap_that_never_committed_is_rolled_back(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """Swapped in but never confirmed open: the restore did not happen."""
        root = tmp_path / "graph-data"
        make_graph(root / "lightrag", "restored")
        make_graph(root / "lightrag.previous-r1", "live")
        healed, _ = construct(registered, root)
        assert label_of(healed.workdir) == "live"
        assert beside(healed) == ["lightrag"]

    def test_a_committed_swap_keeps_the_restored_graph(
        self, registered: Any, tmp_path: Path
    ) -> None:
        """Past the commit point, a half-deleted replaced graph is only debris."""
        root = tmp_path / "graph-data"
        make_graph(root / "lightrag", "restored")
        make_graph(root / "lightrag.replaced-r1", "live")
        healed, _ = construct(registered, root)
        assert label_of(healed.workdir) == "restored"
        assert beside(healed) == ["lightrag"]

    def test_a_leftover_staging_folder_is_removed(self, registered: Any, tmp_path: Path) -> None:
        root = tmp_path / "graph-data"
        make_graph(root / "lightrag", "live")
        make_graph(root / "lightrag.restore-r1", "half extracted")
        healed, _ = construct(registered, root)
        assert label_of(healed.workdir) == "live"
        assert beside(healed) == ["lightrag"]

    def test_the_newest_previous_graph_comes_back_and_the_rest_go(
        self, registered: Any, tmp_path: Path
    ) -> None:
        root = tmp_path / "graph-data"
        older = make_graph(root / "lightrag.previous-r1", "older")
        newer = make_graph(root / "lightrag.previous-r2", "newer")
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        healed, _ = construct(registered, root)
        assert label_of(healed.workdir) == "newer"
        assert beside(healed) == ["lightrag"]

    def test_nothing_else_beside_the_workdir_is_touched(
        self, registered: Any, tmp_path: Path
    ) -> None:
        root = tmp_path / "graph-data"
        make_graph(root / "lightrag", "live")
        others = [
            "lightrag-notes",  # another name altogether
            "lightrag.previous-",  # no run id
            "lightrag.previous-a b",  # not a run id
            "other.restore-r1",  # another workdir's
        ]
        for name in others:
            make_graph(root / name, name)
        (root / "lightrag.restore-r9").write_text("a file, not a restore's folder")
        healed, _ = construct(registered, root)
        assert beside(healed) == sorted(["lightrag", "lightrag.restore-r9", *others])
        assert label_of(healed.workdir) == "live"

    def test_with_nothing_to_heal_construction_changes_nothing(
        self, registered: Any, tmp_path: Path
    ) -> None:
        root = tmp_path / "graph-data"
        make_graph(root / "lightrag", "live")
        healed, _ = construct(registered, root)
        assert beside(healed) == ["lightrag"]
        construct(registered, tmp_path / "nowhere")  # a missing root is no error either
        assert not (tmp_path / "nowhere").exists()

    def test_a_heal_that_fails_is_logged_and_construction_still_succeeds(
        self,
        registered: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A graph that can't be healed must not cost the API its graph service."""
        root = tmp_path / "graph-data"
        make_graph(root / "lightrag.previous-r1", "live")

        def refuse(source: Any, target: Any) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(graph_service.os, "replace", refuse)
        with caplog.at_level(logging.WARNING, logger="varagity.graph.service"):
            healed, _ = construct(registered, root)
        assert healed.workdir == root / "lightrag"
        assert "could not finish an interrupted graph restore" in caplog.text
        assert (root / "lightrag.previous-r1").exists()  # left for the next start
