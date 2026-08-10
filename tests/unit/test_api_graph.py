"""Unit tests for the graph corpus routes, the build runner, and its flow.

Everything runs against doubles: a fake graph service (no engine, no
workdir), an injected build callable (no Prefect), and a scripted settings
store (no postgres). What is under test is the machinery this phase added —
upload sniffing, the directory listing, the stale flag on delete, the
kill-switch posture, one-build-at-a-time, event replay, the bounded render's
no-prune rule, and the resume accounting.
"""

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.sse import parse_sse
from varagity.api.deps import get_app_settings_store, get_graph_build_preflight
from varagity.api.graph_runner import (
    EVENT_LOG,
    EVENT_PROGRESS,
    EVENT_STATUS,
    Feed,
    GraphBuildAlreadyRunning,
    GraphBuildRunner,
    apply_bounds,
    get_graph_build_runner,
    scan_graph_corpus,
)
from varagity.api.main import create_app
from varagity.graph.records import (
    BuildReport,
    GraphExport,
    GraphExportEdge,
    GraphExportNode,
)
from varagity.graph.service import GraphUnavailable, get_graph_service
from varagity.graph.sources.base import MessageBatch, SourceMessage
from varagity.pipeline.graph_flow import GraphBatches

TICK = 0.02  # worker-thread scheduling slack for wait loops


# ── doubles ────────────────────────────────────────────────────────────


class FakeService:
    """A graph service that records what the runner asked it to do."""

    def __init__(
        self,
        workdir: Path,
        *,
        statuses: dict[str, int] | None = None,
        gate: threading.Event | None = None,
        fail_with: Exception | None = None,
        statuses_raise: Exception | None = None,
        graph: GraphExport | None = None,
        export_raise: Exception | None = None,
    ) -> None:
        self._workdir = workdir
        self.statuses = statuses if statuses is not None else {}
        self.gate = gate
        self.fail_with = fail_with
        self.statuses_raise = statuses_raise
        self.graph = graph if graph is not None else GraphExport()
        self.export_raise = export_raise
        self.builds: list[tuple[int, bool]] = []
        self.exports: list[tuple[str, int, int]] = []
        self.closes = 0
        self.opens = 0

    @property
    def workdir(self) -> Path:
        return self._workdir

    def export(self, label: str = "*", *, max_depth: int = 3, max_nodes: int = 1000) -> GraphExport:
        self.exports.append((label, max_depth, max_nodes))
        if self.export_raise is not None:
            raise self.export_raise
        if label == "*":
            return self.graph
        # Label-faithful, like the real engine: the lookup is *exact*, and a
        # miss is an empty slice — the behavior the entity route's canonical-
        # spelling fallback exists for (a case-blind fake hid exactly that).
        if any(node.id == label for node in self.graph.nodes):
            return self.graph
        return GraphExport()

    def build(
        self,
        batches: Sequence[MessageBatch],
        *,
        verbose: int = 0,
        prune_removed: bool = True,
    ) -> BuildReport:
        if self.gate is not None:
            assert self.gate.wait(timeout=5)
        if self.fail_with is not None:
            raise self.fail_with
        self.builds.append((sum(len(batch.messages) for batch in batches), prune_removed))
        return BuildReport(
            messages_seen=sum(len(batch.messages) for batch in batches), wall_clock_s=1.5
        )

    def document_statuses(self) -> dict[str, int]:
        if self.statuses_raise is not None:
            raise self.statuses_raise
        return dict(self.statuses)

    def close(self) -> None:
        self.closes += 1

    def session(self) -> object:
        self.opens += 1
        return self


class FakeSettingsStore:
    """Records the stale flag without touching postgres."""

    def __init__(self, *, corpus_stale: bool = False, graph_stale: bool = False) -> None:
        self.corpus_stale = corpus_stale
        self.graph_stale = graph_stale

    def is_corpus_stale(self) -> bool:
        return self.corpus_stale

    def set_corpus_stale(self, stale: bool) -> None:
        self.corpus_stale = stale

    def is_graph_stale(self) -> bool:
        return self.graph_stale

    def set_graph_stale(self, stale: bool) -> None:
        self.graph_stale = stale


def message(guid: str, *, when: datetime, thread: str = "t1") -> SourceMessage:
    return SourceMessage(
        guid=guid,
        thread_id=thread,
        thread_name="Bob",
        sender_handle="+15551234567",
        sender_name="Bob",
        is_from_me=False,
        timestamp=when,
        text=f"message {guid}",
    )


def batch(name: str, messages: Sequence[SourceMessage]) -> MessageBatch:
    return MessageBatch(doc_id=f"doc-{name}", relative_path=name, messages=list(messages))


def scripted_flow(
    *,
    gate: threading.Event | None = None,
    fail_with: Exception | None = None,
    log_lines: bool = False,
) -> Callable[..., BuildReport]:
    """Build a ``graph_build_flow``-compatible double."""

    def flow(
        service: Any,
        batches: GraphBatches,
        *,
        prune_removed: bool = True,
        verbose: int = 0,
    ) -> BuildReport:
        if log_lines:
            import logging

            logging.getLogger("varagity.graph.engines.lightrag").info(
                "graph build: 2 new, 0 changed, 1 unchanged, 0 removed"
            )
        if gate is not None:
            assert gate.wait(timeout=5)
        if fail_with is not None:
            raise fail_with
        return service.build(batches.batches, prune_removed=prune_removed, verbose=verbose)

    return flow


def make_runner(tmp_path: Path, **kwargs: Any) -> tuple[GraphBuildRunner, FakeService]:
    service = kwargs.pop("service", None) or FakeService(tmp_path / "lightrag")
    kwargs.setdefault("on_reingest_complete", lambda: None)
    runner = GraphBuildRunner(kwargs.pop("flow", scripted_flow()), service=service, **kwargs)
    return runner, service


async def collect(runner: GraphBuildRunner, timeout: float = 5.0) -> list[Feed]:
    async def drain() -> list[Feed]:
        return [item async for item in runner.subscribe()]

    return await asyncio.wait_for(drain(), timeout=timeout)


async def wait_terminal(runner: GraphBuildRunner, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while (snapshot := runner.snapshot()) is None or snapshot.state == "running":
        assert asyncio.get_running_loop().time() < deadline, "build never finished"
        await asyncio.sleep(TICK)


# ── fixtures ───────────────────────────────────────────────────────────


_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)

_CHAT_DB_SCHEMA = """
CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, display_name TEXT);
CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
CREATE TABLE message (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT,
    text TEXT,
    attributedBody BLOB,
    date INTEGER,
    is_from_me INTEGER,
    handle_id INTEGER,
    associated_message_type INTEGER,
    associated_message_guid TEXT
);
CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
"""

# The two message timestamps every fixture database carries (the parse
# summary's first/last).
CHAT_DB_FIRST = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
CHAT_DB_LAST = datetime(2026, 1, 2, 9, 0, tzinfo=UTC)


def write_chat_db(path: Path) -> Path:
    """Write a throwaway ``chat.db`` the iMessage source parses to two messages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_CHAT_DB_SCHEMA)
        conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+15551234567')")
        conn.execute("INSERT INTO chat (ROWID, guid, display_name) VALUES (1, 'chat-1', 'Bob')")
        conn.execute("INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (1, 1)")
        for rowid, (guid, when, text) in enumerate(
            (("m1", CHAT_DB_FIRST, "did you fix the keyboard?"), ("m2", CHAT_DB_LAST, "yes")),
            start=1,
        ):
            nanos = int((when - _APPLE_EPOCH).total_seconds()) * 1_000_000_000
            conn.execute(
                "INSERT INTO message (ROWID, guid, text, attributedBody, date, is_from_me,"
                " handle_id, associated_message_type, associated_message_guid)"
                " VALUES (?, ?, ?, NULL, ?, 0, 1, 0, NULL)",
                (rowid, guid, text, nanos),
            )
            conn.execute(
                "INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, ?)", (rowid,)
            )
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def graph_root(tmp_path: Path, settings_env: Callable[..., None]) -> Path:
    root = tmp_path / "graph-docs"
    root.mkdir()
    settings_env(
        GRAPH_DOCS_PATH=str(root),
        GRAPH_STORAGE_PATH=str(tmp_path / "graph-data"),
        GRAPH_ENABLED="true",
        GRAPH_UPLOAD_MAX_MB=1,
        # The repo .env points this at the container's contacts file; a host
        # run that tried to read it would fail every parse (the eval harness
        # pins it empty for the same reason).
        GRAPH_HANDLE_NAMES_FILE="",
    )
    return root


@pytest.fixture
def store() -> FakeSettingsStore:
    return FakeSettingsStore()


def make_app(
    *,
    runner: GraphBuildRunner | None = None,
    service: object | None = None,
    store: FakeSettingsStore | None = None,
) -> FastAPI:
    application = create_app()
    if runner is not None:
        application.dependency_overrides[get_graph_build_runner] = lambda: runner
    if service is not None:
        application.dependency_overrides[get_graph_service] = lambda: service
    if store is not None:
        application.dependency_overrides[get_app_settings_store] = lambda: store

    async def _noop_preflight() -> None:
        return None

    application.dependency_overrides[get_graph_build_preflight] = lambda: _noop_preflight
    return application


async def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.request(method, path, **kwargs)


def part(name: str, content: bytes) -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (name, content, "application/octet-stream"))


# ── the pure helpers ───────────────────────────────────────────────────


class TestScan:
    def test_lists_every_file_including_the_ones_no_source_claims(self, tmp_path: Path) -> None:
        """Sidecars and contacts files are part of what the user dropped in."""
        (tmp_path / "nested").mkdir()
        for name in ("chat.db", "chat.db-wal", "contacts.txt", "nested/other.db"):
            (tmp_path / name).write_bytes(b"x")
        assert [path.name for path in scan_graph_corpus(tmp_path)] == [
            "chat.db",
            "chat.db-wal",
            "contacts.txt",
            "other.db",
        ]

    def test_a_missing_root_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert scan_graph_corpus(tmp_path / "nope") == []


class TestBounds:
    def batches(self) -> list[MessageBatch]:
        return [
            batch(
                "a.db",
                [
                    message("m1", when=datetime(2026, 1, 1, tzinfo=UTC)),
                    message("m2", when=datetime(2026, 3, 1, tzinfo=UTC)),
                ],
            ),
            batch(
                "b.db",
                [
                    message("m3", when=datetime(2026, 5, 1, tzinfo=UTC)),
                    message("m4", when=datetime(2026, 7, 1, tzinfo=UTC)),
                ],
            ),
        ]

    def test_no_bounds_passes_the_batches_through(self) -> None:
        batches = self.batches()
        assert apply_bounds(batches, message_limit=None, since=None) == batches

    def test_the_cap_keeps_the_newest_messages_across_files(self) -> None:
        """★ The cap is global, not per file — "N messages", not "N each"."""
        kept = apply_bounds(self.batches(), message_limit=3, since=None)
        assert [m.guid for b in kept for m in b.messages] == ["m2", "m3", "m4"]

    def test_since_is_a_date_floor(self) -> None:
        kept = apply_bounds(self.batches(), message_limit=None, since=date(2026, 4, 1))
        assert [m.guid for b in kept for m in b.messages] == ["m3", "m4"]

    def test_a_batch_left_with_nothing_is_dropped(self) -> None:
        """An empty document would claim the conversation is empty."""
        kept = apply_bounds(self.batches(), message_limit=1, since=None)
        assert [b.relative_path for b in kept] == ["b.db"]

    def test_the_cap_applies_after_the_floor(self) -> None:
        kept = apply_bounds(self.batches(), message_limit=2, since=date(2026, 2, 1))
        assert [m.guid for b in kept for m in b.messages] == ["m3", "m4"]

    def test_batch_identity_survives_narrowing(self) -> None:
        """doc_id and relative_path are the manifest diff's join keys."""
        kept = apply_bounds(self.batches(), message_limit=1, since=None)
        assert (kept[0].doc_id, kept[0].relative_path) == ("doc-b.db", "b.db")


# ── the runner ─────────────────────────────────────────────────────────


class TestRunnerLifecycle:
    async def test_idle_runner_reports_a_single_idle_status(self, tmp_path: Path) -> None:
        runner, _ = make_runner(tmp_path)
        events = await collect(runner)
        assert [name for name, _ in events] == [EVENT_STATUS]
        assert events[0][1].run is None

    async def test_a_completed_build_replays_scan_parse_index_and_the_summary(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        write_chat_db(graph_root / "chat.db")
        (graph_root / "chat.db-wal").write_bytes(b"not a database")
        runner, service = make_runner(tmp_path, flow=scripted_flow(log_lines=True))
        assert runner.start().state == "running"
        await wait_terminal(runner)

        events = await collect(runner)
        names = [name for name, _ in events]
        assert names[0] == EVENT_STATUS and events[0][1].run.state == "running"
        assert names[-1] == EVENT_STATUS and events[-1][1].run.state == "completed"

        stages = [p.stage for name, p in events if name == EVENT_PROGRESS]
        assert stages[0] == "scan"
        assert "parse" in stages and "index" in stages
        assert "bound" not in stages  # unbounded run

        logs = [p for name, p in events if name == EVENT_LOG]
        assert any("0 changed" in log.message for log in logs)

        summary = events[-1][1].run.summary
        assert summary is not None
        # Two files on disk, one of them a sidecar no source claims.
        assert (summary.files_scanned, summary.files_parsed, summary.files_failed) == (2, 1, 0)
        assert (summary.messages_parsed, summary.messages_indexed) == (2, 2)
        assert summary.wall_clock_s == 1.5
        assert service.builds == [(2, True)]  # full-corpus render prunes

    async def test_an_unparseable_source_is_counted_not_fatal(
        self, graph_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """★ One corrupt export must not cost the rest of an archive."""
        write_chat_db(graph_root / "chat.db")
        import varagity.api.graph_runner as runner_module

        def boom(*args: Any, **kwargs: Any) -> MessageBatch:
            raise sqlite3.DatabaseError("file is not a database")

        monkeypatch.setattr(runner_module, "batch_for_path", boom)
        runner, _ = make_runner(tmp_path)
        runner.start()
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.state == "completed"
        assert snapshot.summary is not None
        assert (snapshot.summary.files_parsed, snapshot.summary.files_failed) == (0, 1)

    async def test_a_bounded_run_never_prunes(self, graph_root: Path, tmp_path: Path) -> None:
        """★ Decision #9: a partial render must not delete the rest of the archive."""
        write_chat_db(graph_root / "chat.db")
        runner, service = make_runner(tmp_path)
        started = runner.start(message_limit=1)
        assert started.bounded is True
        await wait_terminal(runner)
        assert service.builds == [(1, False)]  # capped to one message, no prune
        stages = [p.stage for name, p in await collect(runner) if name == EVENT_PROGRESS]
        assert "bound" in stages

    async def test_a_date_floor_also_makes_the_run_bounded(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        runner, service = make_runner(tmp_path)
        assert runner.start(since=date(2026, 1, 1)).bounded is True
        await wait_terminal(runner)
        assert service.builds == [(0, False)]

    async def test_second_start_while_running_raises(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        gate = threading.Event()
        runner, _ = make_runner(tmp_path, flow=scripted_flow(gate=gate))
        runner.start()
        try:
            with pytest.raises(GraphBuildAlreadyRunning):
                runner.start()
            assert runner.building is True
        finally:
            gate.set()
        await wait_terminal(runner)
        assert runner.building is False
        runner.start()  # a terminal build frees the slot
        await wait_terminal(runner)

    async def test_live_subscription_streams_until_terminal(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        gate = threading.Event()
        runner, _ = make_runner(tmp_path, flow=scripted_flow(gate=gate))
        runner.start()

        async def consume() -> list[Feed]:
            items: list[Feed] = []
            async for item in runner.subscribe():
                items.append(item)
                if item[0] == EVENT_STATUS and item[1].run is not None:
                    gate.set()  # release the build once we're demonstrably live
            return items

        events = await asyncio.wait_for(consume(), timeout=5)
        assert events[-1][0] == EVENT_STATUS
        assert events[-1][1].run.state == "completed"

    async def test_a_failed_build_reports_the_error_and_still_summarizes_the_scan(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """A failed run must still say what it scanned — else "it never found the file"."""
        write_chat_db(graph_root / "chat.db")
        runner, _ = make_runner(
            tmp_path, flow=scripted_flow(fail_with=RuntimeError("gpu fell over"))
        )
        runner.start()
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None
        assert snapshot.state == "failed"
        assert "gpu fell over" in (snapshot.error or "")
        assert snapshot.summary is not None and snapshot.summary.files_scanned == 1
        assert snapshot.summary.messages_seen == 0

    async def test_reingest_wipes_the_workdir_and_reopens(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        workdir = tmp_path / "lightrag"
        workdir.mkdir()
        (workdir / "graph_chunk_entity_relation.graphml").write_text("stale")
        service = FakeService(workdir)
        runner, _ = make_runner(tmp_path, service=service)
        runner.start(reingest=True)
        await wait_terminal(runner)
        assert not workdir.exists()  # the fake reopen creates nothing
        assert (service.closes, service.opens) == (1, 1)

    async def test_a_plain_build_never_touches_the_workdir(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ Resume is the normal case: the discipline is not to wipe."""
        workdir = tmp_path / "lightrag"
        workdir.mkdir()
        (workdir / "kv_store_doc_status.json").write_text("{}")
        service = FakeService(workdir)
        runner, _ = make_runner(tmp_path, service=service)
        runner.start()
        await wait_terminal(runner)
        assert (workdir / "kv_store_doc_status.json").exists()
        assert service.closes == 0

    async def test_inflight_documents_are_reported_as_resumed(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ A kill/restart must be visible, not mysterious."""
        service = FakeService(
            tmp_path / "lightrag", statuses={"processed": 4, "failed": 1, "pending": 2}
        )
        runner, _ = make_runner(tmp_path, service=service)
        runner.start()
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.summary is not None
        assert snapshot.summary.resumed == 3
        assert snapshot.summary.documents == {"processed": 4, "failed": 1, "pending": 2}

    async def test_an_engine_that_will_not_report_statuses_is_no_news(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        service = FakeService(tmp_path / "lightrag", statuses_raise=GraphUnavailable("locked"))
        runner, _ = make_runner(tmp_path, service=service)
        runner.start()
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.state == "completed"
        assert snapshot.summary is not None and snapshot.summary.documents == {}

    async def test_reingest_completion_fires_the_stale_clear_hook(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        cleared: list[bool] = []
        runner, _ = make_runner(tmp_path, on_reingest_complete=lambda: cleared.append(True))
        runner.start(reingest=True)
        await wait_terminal(runner)
        assert cleared == [True]

    async def test_a_plain_build_does_not_clear_stale(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ Only a completed rebuild clears it — the corpus-stale rule's twin."""
        cleared: list[bool] = []
        runner, _ = make_runner(tmp_path, on_reingest_complete=lambda: cleared.append(True))
        runner.start(reingest=False)
        await wait_terminal(runner)
        assert cleared == []

    async def test_a_failed_reingest_does_not_clear_stale(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        cleared: list[bool] = []
        runner, _ = make_runner(
            tmp_path,
            flow=scripted_flow(fail_with=RuntimeError("boom")),
            on_reingest_complete=lambda: cleared.append(True),
        )
        runner.start(reingest=True)
        await wait_terminal(runner)
        assert cleared == []

    def test_the_service_is_resolved_lazily(self, tmp_path: Path) -> None:
        """Constructing a runner must not touch the engine registry."""
        runner = GraphBuildRunner()
        assert runner._service is None


# ── the routes ─────────────────────────────────────────────────────────


class TestUpload:
    async def test_a_real_database_lands_and_is_kept(self, graph_root: Path) -> None:
        source = write_chat_db(graph_root / "probe.db")
        payload = source.read_bytes()
        source.unlink()
        response = await request(
            make_app(), "POST", "/api/graph/documents", files=[part("chat.db", payload)]
        )
        assert response.status_code == 201
        (entry,) = response.json()["files"]
        assert entry["stored"] is True
        assert (graph_root / "chat.db").exists()

    async def test_a_file_no_source_claims_is_removed_again(self, graph_root: Path) -> None:
        """★ A corpus directory must not accumulate files nothing can read."""
        response = await request(
            make_app(), "POST", "/api/graph/documents", files=[part("notes.db", b"plain text")]
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "no_file_stored"
        assert "unsupported_graph_source" in body["error"]["message"]
        assert not (graph_root / "notes.db").exists()

    async def test_wal_sidecars_ride_along_unsniffed(self, graph_root: Path) -> None:
        """★ A live chat.db keeps its newest messages in the -wal file."""
        source = write_chat_db(graph_root / "probe.db")
        payload = source.read_bytes()
        source.unlink()
        response = await request(
            make_app(),
            "POST",
            "/api/graph/documents",
            files=[
                part("chat.db", payload),
                part("chat.db-wal", b"\x00wal frames"),
                part("chat.db-shm", b"\x00shm"),
            ],
        )
        assert response.status_code == 201
        assert [entry["stored"] for entry in response.json()["files"]] == [True, True, True]
        assert (graph_root / "chat.db-wal").exists()

    async def test_the_graph_cap_is_its_own_setting(
        self, graph_root: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_UPLOAD_MAX_MB=0)
        response = await request(
            make_app(), "POST", "/api/graph/documents", files=[part("chat.db", b"x" * 4096)]
        )
        assert response.status_code == 422
        assert "file_too_large" in response.json()["error"]["message"]

    async def test_the_corpus_extension_allow_list_does_not_apply(self, graph_root: Path) -> None:
        """★ ``.db`` is not an ALLOWED_EXTENSIONS entry — the sniff is the gate."""
        from varagity.config import get_settings

        assert ".db" not in get_settings().allowed_extension_set
        source = write_chat_db(graph_root / "probe.db")
        payload = source.read_bytes()
        source.unlink()
        response = await request(
            make_app(), "POST", "/api/graph/documents", files=[part("archive.db", payload)]
        )
        assert response.status_code == 201
        assert response.json()["files"][0]["stored"] is True

    async def test_a_rejected_file_says_the_sniff_refused_it_not_its_extension(
        self, graph_root: Path
    ) -> None:
        response = await request(
            make_app(), "POST", "/api/graph/documents", files=[part("notes.md", b"# hello")]
        )
        assert response.status_code == 422
        message = response.json()["error"]["message"]
        assert "unsupported_graph_source" in message
        assert "extension_not_allowed" not in message

    async def test_upload_while_disabled_is_a_structured_403(
        self, graph_root: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_ENABLED="false")
        response = await request(
            make_app(), "POST", "/api/graph/documents", files=[part("chat.db", b"x")]
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "graph_disabled"

    async def test_too_many_files_is_rejected_before_anything_is_written(
        self, graph_root: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(UPLOAD_MAX_FILES=1)
        response = await request(
            make_app(),
            "POST",
            "/api/graph/documents",
            files=[part("a.db", b"x"), part("b.db", b"x")],
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "too_many_files"
        assert list(graph_root.iterdir()) == []


class TestList:
    async def test_lists_everything_on_disk_with_no_parse_summary_before_a_build(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        write_chat_db(graph_root / "chat.db")
        (graph_root / "contacts.txt").write_text("+15551234567=Bob\n")
        runner, _ = make_runner(tmp_path)
        response = await request(make_app(runner=runner), "GET", "/api/graph/documents")
        assert response.status_code == 200
        entries = response.json()
        assert [entry["name"] for entry in entries] == ["chat.db", "contacts.txt"]
        assert all(entry["parse"] is None and entry["doc_id"] is None for entry in entries)
        assert entries[0]["size_bytes"] > 0

    async def test_a_scanned_file_carries_its_doc_id_and_parse_summary(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        write_chat_db(graph_root / "chat.db")
        runner, _ = make_runner(tmp_path)
        runner.start()
        await wait_terminal(runner)
        response = await request(make_app(runner=runner), "GET", "/api/graph/documents")
        (entry,) = response.json()
        assert entry["doc_id"]
        assert entry["parse"]["messages"] == 2
        assert entry["parse"]["threads"] == 1
        assert datetime.fromisoformat(entry["parse"]["first"]) == CHAT_DB_FIRST
        assert datetime.fromisoformat(entry["parse"]["last"]) == CHAT_DB_LAST

    async def test_a_missing_corpus_directory_lists_empty(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_DOCS_PATH=str(tmp_path / "absent"))
        runner, _ = make_runner(tmp_path)
        response = await request(make_app(runner=runner), "GET", "/api/graph/documents")
        assert response.json() == []


class TestDelete:
    async def test_deleting_a_source_flags_the_graph_stale(
        self, graph_root: Path, store: FakeSettingsStore
    ) -> None:
        """★ The graph keeps those messages until a rebuild — say so."""
        write_chat_db(graph_root / "chat.db")
        response = await request(make_app(store=store), "DELETE", "/api/graph/documents/chat.db")
        assert response.status_code == 200
        assert response.json() == {"relative_path": "chat.db", "graph_stale": True}
        assert not (graph_root / "chat.db").exists()
        assert store.graph_stale is True

    async def test_emptied_folders_are_pruned(
        self, graph_root: Path, store: FakeSettingsStore
    ) -> None:
        write_chat_db(graph_root / "old" / "chat.db")
        response = await request(
            make_app(store=store), "DELETE", "/api/graph/documents/old/chat.db"
        )
        assert response.status_code == 200
        assert not (graph_root / "old").exists()

    async def test_an_unknown_file_is_a_structured_404(
        self, graph_root: Path, store: FakeSettingsStore
    ) -> None:
        response = await request(make_app(store=store), "DELETE", "/api/graph/documents/nope.db")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "graph_document_not_found"

    async def test_a_path_escaping_the_corpus_is_a_404_not_a_delete(
        self, graph_root: Path, tmp_path: Path, store: FakeSettingsStore
    ) -> None:
        """★ Containment: the path is data, never authority."""
        outside = tmp_path / "secrets.db"
        outside.write_bytes(b"keep me")
        response = await request(
            make_app(store=store), "DELETE", "/api/graph/documents/../secrets.db"
        )
        assert response.status_code == 404
        assert outside.exists()


class TestBuildRoute:
    async def test_post_returns_a_202_run_handle(self, graph_root: Path, tmp_path: Path) -> None:
        runner, _ = make_runner(tmp_path)
        response = await request(
            make_app(runner=runner), "POST", "/api/graph/build", json={"message_limit": 60}
        )
        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "running"
        assert (body["bounded"], body["message_limit"]) == (True, 60)
        await wait_terminal(runner)

    async def test_a_second_build_is_a_structured_409(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        gate = threading.Event()
        runner, _ = make_runner(tmp_path, flow=scripted_flow(gate=gate))
        app = make_app(runner=runner)
        try:
            first = await request(app, "POST", "/api/graph/build", json={})
            second = await request(app, "POST", "/api/graph/build", json={})
            assert first.status_code == 202
            assert second.status_code == 409
            assert second.json()["error"]["code"] == "graph_build_running"
        finally:
            gate.set()
        await wait_terminal(runner)

    async def test_build_while_disabled_is_a_structured_403(
        self, graph_root: Path, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_ENABLED="false")
        runner, _ = make_runner(tmp_path)
        response = await request(make_app(runner=runner), "POST", "/api/graph/build", json={})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "graph_disabled"
        assert runner.snapshot() is None  # nothing started

    async def test_preflight_failure_is_a_structured_503(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        from fastapi import HTTPException

        async def down() -> None:
            raise HTTPException(
                status_code=503,
                detail={"code": "llamacpp_unreachable", "message": "llamacpp unreachable"},
            )

        runner, _ = make_runner(tmp_path)
        app = make_app(runner=runner)
        app.dependency_overrides[get_graph_build_preflight] = lambda: down
        response = await request(app, "POST", "/api/graph/build", json={})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "llamacpp_unreachable"

    async def test_an_unknown_body_field_is_rejected(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        runner, _ = make_runner(tmp_path)
        response = await request(
            make_app(runner=runner), "POST", "/api/graph/build", json={"reingset": True}
        )
        assert response.status_code == 422


class TestStatusStream:
    async def test_replays_a_terminal_build_and_closes(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        write_chat_db(graph_root / "chat.db")
        runner, _ = make_runner(tmp_path)
        runner.start()
        await wait_terminal(runner)

        app = make_app(runner=runner)
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://api") as client,
            client.stream("GET", "/api/graph/build/status") as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join([chunk async for chunk in response.aiter_text()])

        events = parse_sse(body)
        names = [name for name, _ in events]
        assert names[0] == "status" and events[0][1]["run"]["state"] == "running"
        assert names[-1] == "status" and events[-1][1]["run"]["state"] == "completed"
        stages = [data["stage"] for name, data in events if name == "progress"]
        assert stages[:2] == ["scan", "parse"]

    async def test_is_a_single_idle_frame_with_no_build(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        runner, _ = make_runner(tmp_path)
        app = make_app(runner=runner)
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://api") as client,
            client.stream("GET", "/api/graph/build/status") as response,
        ):
            body = "".join([chunk async for chunk in response.aiter_text()])
        events = parse_sse(body)
        assert len(events) == 1
        name, data = events[0]
        assert name == "status" and data["run"] is None


class TestStatusRoute:
    async def test_an_unbuilt_graph_answers_without_opening_the_engine(
        self, graph_root: Path, tmp_path: Path, store: FakeSettingsStore
    ) -> None:
        """★ A status poll must not pay an engine's startup on a fresh stack."""
        service = FakeService(tmp_path / "absent", statuses={"processed": 3})
        runner, _ = make_runner(tmp_path)
        response = await request(
            make_app(runner=runner, service=service, store=store), "GET", "/api/graph/status"
        )
        assert response.status_code == 200
        body = response.json()
        assert body == {
            "enabled": True,
            "stale": False,
            "building": False,
            "documents": {},
            "entities": None,
            "relations": None,
            "message_guids": None,
            "last_build": None,
        }

    async def test_a_built_graph_reports_the_summary_sidecar_and_statuses(
        self, graph_root: Path, tmp_path: Path, store: FakeSettingsStore
    ) -> None:
        from varagity.graph.manifest import ManifestDoc, WorkdirManifest, save_summary

        workdir = tmp_path / "lightrag"
        manifest = WorkdirManifest(
            docs={
                "t1::2026-01-01": ManifestDoc(content_sha256="a" * 64, message_guids=["m1", "m2"])
            }
        )
        save_summary(workdir, manifest, entities=347, relations=812)
        service = FakeService(workdir, statuses={"processed": 9, "failed": 1})
        runner, _ = make_runner(tmp_path)
        response = await request(
            make_app(runner=runner, service=service, store=store), "GET", "/api/graph/status"
        )
        body = response.json()
        assert (body["entities"], body["relations"], body["message_guids"]) == (347, 812, 2)
        assert body["documents"] == {"processed": 9, "failed": 1}

    async def test_an_unopenable_engine_degrades_instead_of_500ing(
        self, graph_root: Path, tmp_path: Path, store: FakeSettingsStore
    ) -> None:
        workdir = tmp_path / "lightrag"
        workdir.mkdir()
        service = FakeService(workdir, statuses_raise=GraphUnavailable("storage is locked"))
        runner, _ = make_runner(tmp_path)
        response = await request(
            make_app(runner=runner, service=service, store=store), "GET", "/api/graph/status"
        )
        assert response.status_code == 200
        assert response.json()["documents"] == {}

    async def test_the_kill_switch_and_stale_flag_are_reported_honestly(
        self, graph_root: Path, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_ENABLED="false")
        service = FakeService(tmp_path / "absent")
        runner, _ = make_runner(tmp_path)
        response = await request(
            make_app(runner=runner, service=service, store=FakeSettingsStore(graph_stale=True)),
            "GET",
            "/api/graph/status",
        )
        body = response.json()
        assert (body["enabled"], body["stale"]) == (False, True)

    async def test_a_running_build_is_reported_with_its_run(
        self, graph_root: Path, tmp_path: Path, store: FakeSettingsStore
    ) -> None:
        gate = threading.Event()
        runner, _ = make_runner(tmp_path, flow=scripted_flow(gate=gate))
        runner.start()
        try:
            response = await request(
                make_app(runner=runner, service=FakeService(tmp_path / "absent"), store=store),
                "GET",
                "/api/graph/status",
            )
            body = response.json()
            assert body["building"] is True
            assert body["last_build"]["state"] == "running"
        finally:
            gate.set()
        await wait_terminal(runner)


# ── the graph view's read surface ──────────────────────────────────────


def sample_graph(*, truncated: bool = False) -> GraphExport:
    """A two-entity slice with the provenance the drill-down joins on."""
    return GraphExport(
        nodes=[
            GraphExportNode(
                id="Bob Nakamura",
                entity_type="person",
                description="Bob talks about keyboards.",
                degree=2,
                doc_keys=["fx-thread-hw::2024-08-09", "fx-thread-hw::2015-04-17"],
            ),
            GraphExportNode(id="Keyboard", entity_type="technology", degree=1),
        ],
        edges=[
            GraphExportEdge(
                id="Bob Nakamura-Keyboard",
                source="Bob Nakamura",
                target="Keyboard",
                label="prefers",
                description="Bob prefers mechanical keyboards.",
            ),
            GraphExportEdge(id="Jane-Keyboard", source="Jane", target="Keyboard"),
        ],
        truncated=truncated,
    )


def built_workdir(tmp_path: Path) -> Path:
    """A workdir carrying a manifest, i.e. one a build has written."""
    from varagity.graph.manifest import ManifestDoc, WorkdirManifest, save_manifest

    workdir = tmp_path / "lightrag"
    save_manifest(
        workdir,
        WorkdirManifest(
            docs={
                "fx-thread-hw::2024-08-09": ManifestDoc(
                    content_sha256="a" * 64,
                    message_guids=["m1", "m2", "m3"],
                    thread_name="Bob Nakamura",
                    span="2024-08-09",
                )
            }
        ),
    )
    return workdir


class TestExportRoute:
    async def test_the_whole_graph_is_the_default_slice(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(make_app(service=service), "GET", "/api/graph/export")
        assert response.status_code == 200
        body = response.json()
        assert [node["id"] for node in body["nodes"]] == ["Bob Nakamura", "Keyboard"]
        assert body["nodes"][0]["entity_type"] == "person"
        assert body["nodes"][0]["degree"] == 2
        assert len(body["edges"]) == 2
        assert body["truncated"] is False
        assert service.exports == [("*", 3, 1000)]

    async def test_provenance_keys_stay_off_the_export_wire(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ Every node would carry dozens of keys for a picture drawing none."""
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(make_app(service=service), "GET", "/api/graph/export")
        assert "doc_keys" not in response.json()["nodes"][0]

    async def test_truncation_is_passed_through_not_hidden(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ The view must say it drew a slice, never imply it drew everything."""
        service = FakeService(built_workdir(tmp_path), graph=sample_graph(truncated=True))
        response = await request(make_app(service=service), "GET", "/api/graph/export")
        assert response.json()["truncated"] is True

    async def test_the_caps_are_forwarded_to_the_engine(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(
            make_app(service=service),
            "GET",
            "/api/graph/export?label=Bob%20Nakamura&max_depth=2&max_nodes=50",
        )
        assert response.status_code == 200
        assert service.exports == [("Bob Nakamura", 2, 50)]

    async def test_asking_past_the_ceiling_is_a_422_not_a_silent_clamp(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ A caller asking for more than the view renders should hear so."""
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(
            make_app(service=service), "GET", "/api/graph/export?max_nodes=2001"
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
        assert service.exports == []

    async def test_an_unbuilt_graph_answers_empty_without_opening_the_engine(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ A page load must not initialize a workdir to find it empty."""
        service = FakeService(tmp_path / "absent", graph=sample_graph())
        response = await request(make_app(service=service), "GET", "/api/graph/export")
        assert response.status_code == 200
        assert response.json() == {"nodes": [], "edges": [], "truncated": False}
        assert service.exports == []

    async def test_an_unopenable_engine_is_a_structured_503(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ "Nothing built" and "engine broken" must not draw the same picture."""
        service = FakeService(
            built_workdir(tmp_path), export_raise=GraphUnavailable("storage is locked")
        )
        response = await request(make_app(service=service), "GET", "/api/graph/export")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "graph_unavailable"

    async def test_export_while_disabled_is_a_structured_403(
        self, graph_root: Path, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_ENABLED="false")
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(make_app(service=service), "GET", "/api/graph/export")
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "graph_disabled"
        assert service.exports == []


class TestEntityDetailRoute:
    async def test_an_entity_carries_its_relations_and_source_days(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(
            make_app(service=service), "GET", "/api/graph/entities/Bob%20Nakamura"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["entity"]["id"] == "Bob Nakamura"
        assert body["entity"]["description"] == "Bob talks about keyboards."
        # Only the edges that touch it — "Jane-Keyboard" is a neighbour's edge.
        assert [edge["id"] for edge in body["relations"]] == ["Bob Nakamura-Keyboard"]
        assert service.exports == [("Bob Nakamura", 1, 2000)]

    async def test_source_days_resolve_through_the_manifest(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ The manifest is what turns a doc_key into a readable day card."""
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(
            make_app(service=service), "GET", "/api/graph/entities/Bob%20Nakamura"
        )
        known, unknown = response.json()["sources"]
        assert known == {
            "doc_key": "fx-thread-hw::2024-08-09",
            "thread_name": "Bob Nakamura",
            "span": "2024-08-09",
            "message_count": 3,
        }
        # A key the manifest does not know still renders — the graph holds it.
        assert unknown == {
            "doc_key": "fx-thread-hw::2015-04-17",
            "thread_name": "fx-thread-hw",
            "span": "2015-04-17",
            "message_count": 0,
        }

    async def test_an_entity_the_engine_normalized_still_resolves(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        """★ Names arrive from citations in their own spelling.

        The engine's label lookup is exact, so the wrong casing must resolve
        through the whole-graph fallback, not luck.
        """
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(
            make_app(service=service), "GET", "/api/graph/entities/bob%20nakamura"
        )
        assert response.status_code == 200
        assert response.json()["entity"]["id"] == "Bob Nakamura"
        # Exact slice (empty) → whole-graph lookup → canonical re-slice.
        assert service.exports == [
            ("bob nakamura", 1, 2000),
            ("*", 1, 2000),
            ("Bob Nakamura", 1, 2000),
        ]

    async def test_an_unknown_entity_is_a_structured_404(
        self, graph_root: Path, tmp_path: Path
    ) -> None:
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(make_app(service=service), "GET", "/api/graph/entities/Nobody")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "entity_not_found"

    async def test_an_unbuilt_graph_has_no_entities(self, graph_root: Path, tmp_path: Path) -> None:
        service = FakeService(tmp_path / "absent", graph=sample_graph())
        response = await request(
            make_app(service=service), "GET", "/api/graph/entities/Bob%20Nakamura"
        )
        assert response.status_code == 404
        assert service.exports == []

    async def test_detail_while_disabled_is_a_structured_403(
        self, graph_root: Path, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        settings_env(GRAPH_ENABLED="false")
        service = FakeService(built_workdir(tmp_path), graph=sample_graph())
        response = await request(
            make_app(service=service), "GET", "/api/graph/entities/Bob%20Nakamura"
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "graph_disabled"


# ── the settings surface ───────────────────────────────────────────────


class TestSettingsSurface:
    def test_the_kill_switch_is_overridable_in_its_own_group(self) -> None:
        from varagity.api.runtime_settings import OVERRIDABLE

        entry = OVERRIDABLE["GRAPH_ENABLED"]
        assert (entry.group, entry.reingest_affecting) == ("graph", False)

    def test_the_engine_is_not_overridable(self) -> None:
        """★ Switching engines re-indexes from scratch — a redeploy, not a toggle."""
        from varagity.api.runtime_settings import OVERRIDABLE

        assert "GRAPH_ENGINE" not in OVERRIDABLE


@pytest.fixture(autouse=True)
def _isolate_runtime_overrides() -> Iterator[None]:
    """Keep GRAPH_ENABLED patches from leaking into other test modules."""
    from varagity.api import runtime_settings

    yield
    runtime_settings.reset_for_tests()
