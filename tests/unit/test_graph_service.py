"""Unit tests for the process-wide graph service.

Driven entirely by a fake engine: what is under test is not LightRAG but the
*ownership* rules the API depends on — one session per process, one build at
a time, reads that a build cannot block, and a broken engine that degrades a
turn instead of taking the process down.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from varagity.config import get_settings
from varagity.graph import service as graph_service
from varagity.graph.base import GRAPH_ENGINE_REGISTRY
from varagity.graph.records import (
    BuildReport,
    GraphAnswer,
    GraphEvidence,
    GraphExport,
    GraphExportNode,
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

    def __init__(self, *, before_build: threading.Event | None = None) -> None:
        self.before_build = before_build
        self.release = threading.Event()
        self.closed = False
        self.builds: list[bool] = []
        self.resumes = 0
        self.queries: list[tuple[str, str | None]] = []
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
        self.closed = True


class FakeEngine:
    """Hands out :class:`FakeSession`s, or refuses to."""

    def __init__(self, *, fail_open: bool = False, fail_close: bool = False, **kwargs: Any) -> None:
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.session_kwargs = kwargs
        self.opens = 0
        self.workdirs: list[Path] = []
        self.sessions: list[FakeSession] = []

    @contextmanager
    def session(self, workdir: Path) -> Iterator[FakeSession]:
        self.opens += 1
        self.workdirs.append(workdir)
        if self.fail_open:
            raise RuntimeError("engine storage is locked")
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
        resolved.export("Bob", max_depth=1, max_nodes=25)
        resolved.close()
        (session,) = engine.sessions
        assert session.queries == [("q", "global")]
        assert session.exports == [("Bob", 1, 25)]


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
