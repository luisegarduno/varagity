"""Unit tests for the backup runner and its API routes (ADR-019).

The runner executes an injected export flow (the ``test_api_ingest.py``
pattern), so no Prefect or postgres is touched — but the injected flow runs
the *real* :func:`~varagity.backup.rag.export_rag` over a fake snapshot, so
the bundle the routes list and serve is a real zip in a tmp ``DOCS_PATH``.
What's under test: the gate as the one-run-at-a-time rule, replay-then-live
subscription, the terminal frame's saved path, and the structured route
errors.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from tests.sse import parse_sse
from tests.unit.test_backup_rag import FakeStore, chunk, document
from varagity.api.backup_runner import (
    EVENT_LOG,
    EVENT_PROGRESS,
    EVENT_STATUS,
    BackupRunner,
    Feed,
    current_ingest_settings,
    get_backup_runners,
)
from varagity.api.corpus_gate import CorpusBusy, CorpusGate
from varagity.api.deps import get_backup_preflight, get_vector_store
from varagity.api.main import create_app
from varagity.backup.rag import RagExport, export_rag

TICK = 0.02  # worker-thread scheduling slack for wait loops
DIM = 4
INGEST_SETTINGS = {"CHUNK_SIZE": "400", "CONTEXTUALIZE": "true"}


@pytest.fixture
def docs_root(tmp_path: Path, settings_env: Callable[..., None]) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    settings_env(DOCS_PATH=str(root), EMBEDDING_MODEL="test-model", EMBEDDING_DIM=DIM)
    return root


def corpus_store(docs_root: Path) -> FakeStore:
    return FakeStore(
        [document("doc-a", str(docs_root / "a.md"), 2)],
        [chunk("doc-a", 0, 0, [0.5] * DIM), chunk("doc-a", 1, 1, [0.25] * DIM)],
    )


def scripted_flow(
    docs_root: Path,
    *,
    hold: threading.Event | None = None,
    fail_with: Exception | None = None,
) -> Callable[..., RagExport]:
    """An export-flow double running the real export over a fake snapshot.

    Args:
        docs_root: The tmp corpus root (its fake rows live inside it).
        hold: When given, the flow blocks on it first (the ``running`` state
            becomes observable deterministically).
        fail_with: Raise this instead of exporting.
    """

    def flow(dest: Path, *, ingest_settings: dict[str, str], on_progress: Any = None) -> RagExport:
        if hold is not None:
            assert hold.wait(timeout=5)
        if fail_with is not None:
            raise fail_with
        return export_rag(
            dest, corpus_store(docs_root), ingest_settings, stale=False, on_progress=on_progress
        )

    return flow


def make_runner(docs_root: Path, **kwargs: Any) -> BackupRunner:
    flow = kwargs.pop("flow", None) or scripted_flow(docs_root)
    kwargs.setdefault("gate", CorpusGate("rag"))
    kwargs.setdefault("ingest_settings", lambda: dict(INGEST_SETTINGS))
    return BackupRunner("rag", export_flow=flow, **kwargs)


async def collect(runner: BackupRunner, timeout: float = 5.0) -> list[Feed]:
    async def drain() -> list[Feed]:
        return [item async for item in runner.subscribe()]

    return await asyncio.wait_for(drain(), timeout=timeout)


async def wait_terminal(runner: BackupRunner, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while (snapshot := runner.snapshot()) is None or snapshot.state == "running":
        assert asyncio.get_running_loop().time() < deadline, "run never finished"
        await asyncio.sleep(TICK)


class TestRunnerLifecycle:
    async def test_idle_runner_reports_a_single_idle_status(self, docs_root: Path) -> None:
        runner = make_runner(docs_root)
        events = await collect(runner)
        assert [name for name, _ in events] == [EVENT_STATUS]
        assert events[0][1].run is None
        assert runner.snapshot() is None
        assert runner.running is False

    async def test_completed_export_replays_its_whole_run(self, docs_root: Path) -> None:
        runner = make_runner(docs_root)
        started = runner.start("export")
        assert started.state == "running"
        assert started.corpus == "rag" and started.operation == "export"
        assert started.bundle is None and started.saved_to is None  # nothing whole yet
        await wait_terminal(runner)

        events = await collect(runner)
        names = [name for name, _ in events]
        first, last = events[0][1], events[-1][1]
        assert names[0] == EVENT_STATUS and first.run.state == "running"
        assert names[-1] == EVENT_STATUS and last.run.state == "completed"

        stages = [payload.stage for name, payload in events if name == EVENT_PROGRESS]
        assert stages[0] == "snapshot" and stages[-1] == "finalize"
        assert "documents" in stages and "chunks" in stages
        logs = [payload.message for name, payload in events if name == EVENT_LOG]
        assert any("exported 1 document(s) and 2 chunk(s)" in line for line in logs)

        run = last.run
        assert run.bundle is not None and run.bundle.startswith("varagity-rag-")
        assert run.saved_to == f"docs/varagity-exports/{run.bundle}"
        assert run.summary.counts == {"documents": 1, "chunks": 2}
        bundle_path = docs_root / "varagity-exports" / run.bundle
        assert run.summary.size_bytes == bundle_path.stat().st_size
        assert run.finished_at is not None and run.error is None

    async def test_the_bundle_records_the_settings_captured_at_start(self, docs_root: Path) -> None:
        from varagity.backup.bundle import read_manifest

        runner = make_runner(docs_root)
        runner.start("export")
        await wait_terminal(runner)
        run = runner.snapshot()
        assert run is not None and run.bundle is not None
        manifest = read_manifest(docs_root / "varagity-exports" / run.bundle)
        assert manifest.ingest_settings == INGEST_SETTINGS

    async def test_a_second_export_while_one_runs_is_corpus_busy(self, docs_root: Path) -> None:
        hold = threading.Event()
        gate = CorpusGate("rag")
        runner = make_runner(docs_root, flow=scripted_flow(docs_root, hold=hold), gate=gate)
        runner.start("export")
        try:
            assert runner.running is True
            assert gate.holder == "export"
            with pytest.raises(CorpusBusy, match="an export is running"):
                runner.start("export")
        finally:
            hold.set()
        await wait_terminal(runner)
        assert gate.holder is None  # the terminal frame gave the corpus back
        runner.start("export")  # …so the next export starts at once
        await wait_terminal(runner)

    async def test_another_operation_holding_the_corpus_refuses_the_export(
        self, docs_root: Path
    ) -> None:
        gate = CorpusGate("rag")
        gate.claim("ingest")
        runner = make_runner(docs_root, gate=gate)
        with pytest.raises(CorpusBusy) as caught:
            runner.start("export")
        assert caught.value.holder == "ingest"
        assert gate.holder == "ingest"  # untouched by the refusal
        assert runner.snapshot() is None  # no run was created

    async def test_live_subscription_streams_until_terminal(self, docs_root: Path) -> None:
        hold = threading.Event()
        runner = make_runner(docs_root, flow=scripted_flow(docs_root, hold=hold))
        runner.start("export")

        async def consume() -> list[Feed]:
            items: list[Feed] = []
            async for item in runner.subscribe():
                items.append(item)
                hold.set()  # release the flow once we're demonstrably live
            return items

        events = await asyncio.wait_for(consume(), timeout=5)
        assert events[0][0] == EVENT_STATUS
        assert events[-1][1].run.state == "completed"
        assert any(name == EVENT_PROGRESS for name, _ in events)

    async def test_a_failed_export_reports_and_frees_the_corpus(self, docs_root: Path) -> None:
        gate = CorpusGate("rag")
        runner = make_runner(
            docs_root, flow=scripted_flow(docs_root, fail_with=OSError("disk full")), gate=gate
        )
        runner.start("export")
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None
        assert snapshot.state == "failed"
        assert snapshot.error == "OSError: disk full"
        assert snapshot.bundle is None and snapshot.summary is None
        assert gate.holder is None
        events = await collect(runner)
        assert events[-1][1].run.state == "failed"

    async def test_a_flow_returning_nothing_is_a_failed_run(self, docs_root: Path) -> None:
        runner = make_runner(docs_root, flow=lambda dest, **kwargs: None)
        runner.start("export")
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.state == "failed"
        assert snapshot.error == "the export returned no bundle"

    async def test_a_thread_that_will_not_start_gives_the_corpus_back(
        self, docs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = CorpusGate("rag")
        runner = make_runner(docs_root, gate=gate)

        def refuse(self: threading.Thread) -> None:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", refuse)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            runner.start("export")
        assert gate.holder is None
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.state == "failed"

    async def test_backup_log_lines_are_relayed_into_the_feed(self, docs_root: Path) -> None:
        def chatty(dest: Path, *, ingest_settings: dict[str, str], on_progress: Any) -> Any:
            logging.getLogger("varagity.backup.rag").warning("drift spotted")
            logging.getLogger("varagity.backup.rag").debug("too quiet to relay")
            return scripted_flow(docs_root)(
                dest, ingest_settings=ingest_settings, on_progress=on_progress
            )

        runner = make_runner(docs_root, flow=chatty)
        runner.start("export")
        await wait_terminal(runner)
        logs = [p for name, p in await collect(runner) if name == EVENT_LOG]
        assert [(log.level, log.message) for log in logs][0] == ("WARNING", "drift spotted")
        assert all("too quiet" not in log.message for log in logs)


class TestRunnerWiring:
    def test_ingest_settings_are_the_reingest_affecting_ones_in_env_form(
        self, settings_env: Callable[..., None]
    ) -> None:
        settings_env(
            CHUNKING_STRATEGY="markdown_aware",
            CHUNK_SIZE=800,
            CHUNK_OVERLAP=100,
            CONTEXTUALIZE="false",
            OCR_ENGINE="tesseract",
        )
        assert current_ingest_settings() == {
            "CHUNKING_STRATEGY": "markdown_aware",
            "CHUNK_OVERLAP": "100",
            "CHUNK_SIZE": "800",
            "CONTEXTUALIZE": "false",
            "OCR_ENGINE": "tesseract",
        }

    def test_the_process_has_one_rag_runner_on_the_shared_gate(self) -> None:
        from varagity.api.corpus_gate import get_corpus_gates

        runners = get_backup_runners()
        assert set(runners) == {"rag"}
        assert get_backup_runners()["rag"] is runners["rag"]
        assert runners["rag"]._gate is get_corpus_gates()["rag"]

    def test_a_corpus_without_an_export_flow_is_refused_at_construction(self) -> None:
        with pytest.raises(KeyError):
            BackupRunner("graph", gate=CorpusGate("graph"))


class FakeVectorStore:
    def __init__(self, documents: int) -> None:
        self.documents = documents

    def document_count(self) -> int:
        return self.documents


def make_app(
    runner: BackupRunner,
    *,
    documents: int = 1,
    preflight: Callable[[str, str], Any] | None = None,
) -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_backup_runners] = lambda: {"rag": runner}

    async def _noop(corpus: str, operation: str) -> None:
        return None

    application.dependency_overrides[get_backup_preflight] = lambda: preflight or _noop
    application.dependency_overrides[get_vector_store] = lambda: FakeVectorStore(documents)
    return application


async def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.request(method, path, **kwargs)


async def stream_body(app: FastAPI, path: str) -> str:
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://api") as client,
        client.stream("GET", path) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return "".join([chunk async for chunk in response.aiter_text()])


class TestRoutes:
    async def test_export_round_trip(self, docs_root: Path) -> None:
        """★ 202 → replayed SSE → listed → served as an attachment."""
        runner = make_runner(docs_root)
        app = make_app(runner)

        started = await request(app, "POST", "/api/backups/rag/export")
        assert started.status_code == 202
        body = started.json()
        assert body["state"] == "running"
        assert body["corpus"] == "rag" and body["operation"] == "export"
        await wait_terminal(runner)

        events = parse_sse(await stream_body(app, "/api/backups/rag/status"))
        names = [name for name, _ in events]
        assert names[0] == "status" and events[0][1]["run"]["state"] == "running"
        assert names[-1] == "status"
        final = events[-1][1]["run"]
        assert final["state"] == "completed"
        assert "progress" in names
        name = final["bundle"]
        assert final["saved_to"] == f"docs/varagity-exports/{name}"

        listed = await request(app, "GET", "/api/backups/rag")
        assert listed.status_code == 200
        (entry,) = listed.json()
        assert entry["name"] == name
        assert entry["manifest"]["counts"] == {"documents": 1, "chunks": 2}
        assert entry["manifest"]["embedding_model"] == "test-model"
        assert entry["manifest"]["corpus"] == "rag"
        bundle_path = docs_root / "varagity-exports" / name
        assert entry["size_bytes"] == bundle_path.stat().st_size

        download = await request(app, "GET", f"/api/backups/rag/files/{name}")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        assert download.headers["content-disposition"] == f'attachment; filename="{name}"'
        assert download.content == bundle_path.read_bytes()

    async def test_a_second_export_is_a_structured_409(self, docs_root: Path) -> None:
        hold = threading.Event()
        runner = make_runner(docs_root, flow=scripted_flow(docs_root, hold=hold))
        app = make_app(runner)
        try:
            first = await request(app, "POST", "/api/backups/rag/export")
            second = await request(app, "POST", "/api/backups/rag/export")
            assert first.status_code == 202
            assert second.status_code == 409
            error = second.json()["error"]
            assert error["code"] == "corpus_busy"
            assert "an export is running" in error["message"]
        finally:
            hold.set()
        await wait_terminal(runner)

    async def test_an_empty_corpus_is_nothing_to_export(self, docs_root: Path) -> None:
        runner = make_runner(docs_root)
        response = await request(make_app(runner, documents=0), "POST", "/api/backups/rag/export")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "nothing_to_export"
        assert runner.snapshot() is None  # refused before any run existed

    async def test_an_unreachable_postgres_is_a_structured_503(self, docs_root: Path) -> None:
        calls: list[tuple[str, str]] = []

        async def down(corpus: str, operation: str) -> None:
            calls.append((corpus, operation))
            raise HTTPException(
                status_code=503,
                detail={"code": "postgres_unreachable", "message": "postgres unreachable"},
            )

        runner = make_runner(docs_root)
        response = await request(
            make_app(runner, preflight=down), "POST", "/api/backups/rag/export"
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "postgres_unreachable"
        assert calls == [("rag", "export")]

    async def test_a_squatted_exports_folder_is_a_structured_500(self, docs_root: Path) -> None:
        (docs_root / "varagity-exports").write_text("a file where the folder should be")
        runner = make_runner(docs_root)
        response = await request(make_app(runner), "POST", "/api/backups/rag/export")
        assert response.status_code == 500
        error = response.json()["error"]
        assert error["code"] == "docs_path_not_writable"
        assert "docs/varagity-exports/" in error["message"]
        assert runner.snapshot() is None

    async def test_an_unwritable_corpus_root_is_a_structured_500(self, docs_root: Path) -> None:
        docs_root.chmod(0o555)
        try:
            response = await request(
                make_app(make_runner(docs_root)), "POST", "/api/backups/rag/export"
            )
        finally:
            docs_root.chmod(0o755)  # pytest's tmp cleanup needs the write bit back
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "docs_path_not_writable"

    async def test_a_read_only_exports_folder_is_a_structured_500(self, docs_root: Path) -> None:
        exports = docs_root / "varagity-exports"
        exports.mkdir()
        exports.chmod(0o555)
        try:
            response = await request(
                make_app(make_runner(docs_root)), "POST", "/api/backups/rag/export"
            )
        finally:
            exports.chmod(0o755)
        assert response.status_code == 500
        assert "permission denied" in response.json()["error"]["message"]

    async def test_the_list_is_empty_before_any_export(self, docs_root: Path) -> None:
        response = await request(make_app(make_runner(docs_root)), "GET", "/api/backups/rag")
        assert response.status_code == 200
        assert response.json() == []

    async def test_a_hand_copied_zip_is_listed_without_a_manifest(self, docs_root: Path) -> None:
        import zipfile

        exports = docs_root / "varagity-exports"
        exports.mkdir()
        with zipfile.ZipFile(exports / "holiday-photos.zip", "w") as archive:
            archive.writestr("a.jpg", b"...")
        response = await request(make_app(make_runner(docs_root)), "GET", "/api/backups/rag")
        assert [(entry["name"], entry["manifest"]) for entry in response.json()] == [
            ("holiday-photos.zip", None)
        ]

    @pytest.mark.parametrize(
        "name",
        [
            "missing.zip",  # not there
            ".varagity-rag-20261108-1312.zip.partial",  # in flight: never served
            ".hidden.zip",
            "notes.txt",  # not a bundle name
            "%2e%2e",  # the folder's parent, encoded (the router decodes it to "..")
            "link.zip",  # a symlink leading out of the folder
        ],
    )
    async def test_names_that_are_not_bundles_in_the_folder_are_404(
        self, docs_root: Path, tmp_path: Path, name: str
    ) -> None:
        exports = docs_root / "varagity-exports"
        exports.mkdir()
        (exports / ".varagity-rag-20261108-1312.zip.partial").write_bytes(b"partial")
        (exports / ".hidden.zip").write_bytes(b"hidden")
        (exports / "notes.txt").write_text("not a bundle")
        outside = tmp_path / "secret.zip"
        outside.write_bytes(b"secret")
        (exports / "link.zip").symlink_to(outside)

        response = await request(
            make_app(make_runner(docs_root)), "GET", f"/api/backups/rag/files/{name}"
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "bundle_not_found"

    @pytest.mark.parametrize("name", ["..", ".", "../docs.zip", "sub/x.zip", ""])
    def test_the_resolver_refuses_anything_but_a_bare_bundle_name(
        self, docs_root: Path, name: str
    ) -> None:
        from varagity.api.routes.backups import _bundle_path

        (docs_root / "varagity-exports").mkdir()
        (docs_root / "docs.zip").write_bytes(b"one level up")
        with pytest.raises(HTTPException) as caught:
            _bundle_path("rag", name)
        assert caught.value.status_code == 404

    async def test_a_traversing_path_never_reaches_the_route(self, docs_root: Path) -> None:
        response = await request(
            make_app(make_runner(docs_root)), "GET", "/api/backups/rag/files/..%2F..%2Fsecret.zip"
        )
        assert response.status_code == 404  # no route matches a multi-segment name

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/api/backups/graph"),
            ("POST", "/api/backups/graph/export"),
            ("GET", "/api/backups/graph/status"),
            ("GET", "/api/backups/graph/files/x.zip"),
            ("GET", "/api/backups/everything"),
        ],
    )
    async def test_only_the_rag_corpus_has_backups_yet(
        self, docs_root: Path, method: str, path: str
    ) -> None:
        response = await request(make_app(make_runner(docs_root)), method, path)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_the_status_stream_is_one_idle_frame_with_no_run(self, docs_root: Path) -> None:
        events = parse_sse(
            await stream_body(make_app(make_runner(docs_root)), "/api/backups/rag/status")
        )
        assert events == [("status", {"run": None})]
