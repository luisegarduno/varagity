"""Unit tests for the backup runner and its API routes (ADR-019).

The runner executes injected export and restore flows (the
``test_api_ingest.py`` pattern), so no Prefect or postgres is touched — but
the injected flows run the *real* :func:`~varagity.backup.rag.export_rag` and
:func:`~varagity.backup.rag.restore_rag` over fake stores, so the bundles the
routes list, verify, serve and restore are real zips in a tmp
``DOCS_PATH``. What's under test: the gate as the one-run-at-a-time rule,
replay-then-live subscription, the terminal frame's saved path and restore
summary, the restore route's up-front refusals, the list's cheap verdict,
and the remembered damage. (That the real ``rag-restore`` flow writes the
stale flag only on completion is ``test_pipeline_flows.py``'s, under the
Prefect harness.)
"""

import asyncio
import logging
import os
import threading
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from tests.sse import parse_sse
from tests.unit.test_backup_rag import (
    FakeRebuildBM25,
    FakeRestoreStore,
    FakeStore,
    chunk,
    document,
    rezip,
)
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
from varagity.api.routes import backups as backups_routes
from varagity.api.routes.backups import DamagedBundles, get_damaged_bundles
from varagity.backup.rag import (
    CHUNKS_MEMBER,
    RagExport,
    RagRestore,
    derive_corpus_stale,
    export_rag,
    restore_rag,
)

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


def scripted_restore_flow(
    docs_root: Path,
    *,
    store: FakeRestoreStore | None = None,
    bm25: FakeRebuildBM25 | None = None,
    hold: threading.Event | None = None,
) -> Callable[..., RagRestore]:
    """A restore-flow double: the real restore over fake stores, then the stale derivation.

    Args:
        docs_root: The tmp corpus root (the re-rooting target).
        store: The fake vector store to replace (a fresh one by default).
        bm25: The fake BM25 store to rebuild (a fresh one by default).
        hold: When given, the flow blocks on it first.
    """

    def flow(
        bundle: Path,
        *,
        ingest_settings: Callable[[], Mapping[str, str]],
        on_progress: Any = None,
    ) -> RagRestore:
        if hold is not None:
            assert hold.wait(timeout=5)
        result = restore_rag(
            bundle,
            store or FakeRestoreStore(),  # type: ignore[arg-type]
            bm25 or FakeRebuildBM25(),  # type: ignore[arg-type]
            docs_root,
            on_progress=on_progress,
        )
        return replace(result, stale=derive_corpus_stale(result.manifest, ingest_settings()))

    return flow


def make_runner(docs_root: Path, **kwargs: Any) -> BackupRunner:
    flow = kwargs.pop("flow", None) or scripted_flow(docs_root)
    kwargs.setdefault("restore_flow", scripted_restore_flow(docs_root))
    kwargs.setdefault("gate", CorpusGate("rag"))
    kwargs.setdefault("ingest_settings", lambda: dict(INGEST_SETTINGS))
    return BackupRunner("rag", export_flow=flow, **kwargs)


def exported_bundle(docs_root: Path) -> Path:
    """Write one real bundle into the exports folder (no run involved)."""
    return export_rag(
        docs_root / "varagity-exports", corpus_store(docs_root), INGEST_SETTINGS, stale=False
    ).bundle.path


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
    def __init__(self, documents: int, *, lock_free: bool = True) -> None:
        self.documents = documents
        self.lock_free = lock_free

    def document_count(self) -> int:
        return self.documents

    def corpus_lock_free(self) -> bool:
        return self.lock_free


def make_app(
    runner: BackupRunner,
    *,
    documents: int = 1,
    preflight: Callable[[str, str], Any] | None = None,
    lock_free: bool = True,
    damaged: DamagedBundles | None = None,
) -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_backup_runners] = lambda: {"rag": runner}

    async def _noop(corpus: str, operation: str) -> None:
        return None

    application.dependency_overrides[get_backup_preflight] = lambda: preflight or _noop
    application.dependency_overrides[get_vector_store] = lambda: FakeVectorStore(
        documents, lock_free=lock_free
    )
    memory = damaged if damaged is not None else DamagedBundles()
    application.dependency_overrides[get_damaged_bundles] = lambda: memory
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
            ("POST", "/api/backups/graph/restore"),
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


# ── restore (phase 2) ─────────────────────────────────────────────────────


class TestRestoreRunner:
    async def test_a_completed_restore_replays_its_four_stages(self, docs_root: Path) -> None:
        bundle = exported_bundle(docs_root)
        store = FakeRestoreStore()
        runner = make_runner(docs_root, restore_flow=scripted_restore_flow(docs_root, store=store))
        started = runner.start("restore", bundle.name)
        assert started.operation == "restore"
        assert started.bundle == bundle.name  # named from the start, unlike an export
        assert started.saved_to is None
        await wait_terminal(runner)

        events = await collect(runner)
        stages = [payload.stage for name, payload in events if name == EVENT_PROGRESS]
        assert stages[0] == "validate"
        assert stages.index("load") < stages.index("index") < stages.index("bm25")
        logs = [payload.message for name, payload in events if name == EVENT_LOG]
        assert any(f"from {bundle.name}" in line for line in logs)

        run = events[-1][1].run
        assert run.state == "completed"
        assert run.summary.counts == {"documents": 1, "chunks": 2, "bm25": 2}
        assert run.summary.size_bytes == bundle.stat().st_size
        assert run.summary.stale is False  # the bundle's settings are the stack's
        assert run.saved_to is None
        assert [row.doc_id for row in store.documents] == ["doc-a"]

    async def test_drifted_settings_come_back_stale(self, docs_root: Path) -> None:
        bundle = exported_bundle(docs_root)
        runner = make_runner(
            docs_root, ingest_settings=lambda: {**INGEST_SETTINGS, "CHUNK_SIZE": "800"}
        )
        runner.start("restore", bundle.name)
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.summary is not None
        assert snapshot.summary.stale is True

    async def test_one_backup_run_at_a_time_whatever_the_operation(self, docs_root: Path) -> None:
        bundle = exported_bundle(docs_root)
        hold = threading.Event()
        gate = CorpusGate("rag")
        runner = make_runner(
            docs_root, gate=gate, restore_flow=scripted_restore_flow(docs_root, hold=hold)
        )
        runner.start("restore", bundle.name)
        try:
            assert gate.holder == "restore"
            with pytest.raises(CorpusBusy, match="a restore is running"):
                runner.start("restore", bundle.name)
            with pytest.raises(CorpusBusy, match="a restore is running"):
                runner.start("export")
            with pytest.raises(CorpusBusy):
                runner.ensure_free()
        finally:
            hold.set()
        await wait_terminal(runner)
        assert gate.holder is None
        runner.ensure_free()

    async def test_a_failed_restore_reports_and_frees_the_corpus(self, docs_root: Path) -> None:
        bundle = exported_bundle(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            chunks = archive.read(CHUNKS_MEMBER)
        damaged = rezip(
            bundle, {CHUNKS_MEMBER: chunks.replace(b"chunk 1", b"chunk 7")}, reseal=False
        )
        gate = CorpusGate("rag")
        runner = make_runner(docs_root, gate=gate)
        runner.start("restore", damaged.name)
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.state == "failed"
        assert snapshot.error is not None
        assert snapshot.error.startswith("BundleDamaged: chunks.jsonl does not match")
        assert snapshot.summary is None
        assert gate.holder is None

    async def test_a_restore_flow_returning_nothing_is_a_failed_run(self, docs_root: Path) -> None:
        runner = make_runner(docs_root, restore_flow=lambda bundle, **kwargs: None)
        runner.start("restore", "x.zip")
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None and snapshot.error == "the restore returned no result"

    async def test_start_checks_its_arguments(self, docs_root: Path) -> None:
        runner = make_runner(docs_root)
        with pytest.raises(ValueError, match="a restore takes the bundle"):
            runner.start("restore")
        with pytest.raises(ValueError, match="an export takes none"):
            runner.start("export", "x.zip")
        assert runner.snapshot() is None

    def test_a_corpus_without_a_restore_refuses_one(self, docs_root: Path) -> None:
        runner = BackupRunner(
            "graph", gate=CorpusGate("graph"), export_flow=scripted_flow(docs_root)
        )
        with pytest.raises(ValueError, match="the graph corpus has no restore"):
            runner.start("restore", "g.zip")

    async def test_the_flow_gets_the_settings_provider_not_a_snapshot(
        self, docs_root: Path
    ) -> None:
        """The stale flag is judged against the settings current when it is written."""
        bundle = exported_bundle(docs_root)
        seen: list[Any] = []

        def flow(bundle: Path, *, ingest_settings: Any, on_progress: Any = None) -> Any:
            seen.append(ingest_settings)
            return None

        settings = {"current": dict(INGEST_SETTINGS)}
        runner = make_runner(
            docs_root, restore_flow=flow, ingest_settings=lambda: settings["current"]
        )
        runner.start("restore", bundle.name)
        await wait_terminal(runner)
        (provider,) = seen
        settings["current"] = {"CHUNK_SIZE": "800"}  # a PATCH lands mid-restore…
        assert provider() == {"CHUNK_SIZE": "800"}  # …and the provider sees it


async def restore(app: FastAPI, name: str) -> httpx.Response:
    return await request(app, "POST", "/api/backups/rag/restore", json={"name": name})


class TestRestoreRoute:
    async def test_restore_round_trip(self, docs_root: Path) -> None:
        """★ 202 → the replayed stream walks the four stages → completed."""
        bundle = exported_bundle(docs_root)
        runner = make_runner(docs_root)
        app = make_app(runner)

        started = await restore(app, bundle.name)
        assert started.status_code == 202
        body = started.json()
        assert body["operation"] == "restore" and body["bundle"] == bundle.name
        await wait_terminal(runner)

        events = parse_sse(await stream_body(app, "/api/backups/rag/status"))
        stages = [data["stage"] for name, data in events if name == "progress"]
        assert {"validate", "load", "index", "bm25"} <= set(stages)
        final = events[-1][1]["run"]
        assert final["state"] == "completed"
        assert final["summary"]["counts"] == {"documents": 1, "chunks": 2, "bm25": 2}
        assert final["summary"]["stale"] is False

    async def test_an_unknown_bundle_is_404(self, docs_root: Path) -> None:
        exported_bundle(docs_root)
        runner = make_runner(docs_root)
        response = await restore(make_app(runner), "missing.zip")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "bundle_not_found"
        assert runner.snapshot() is None

    async def test_a_bundle_from_another_embedding_model_is_422_incompatible(
        self, docs_root: Path, settings_env: Callable[..., None]
    ) -> None:
        bundle = exported_bundle(docs_root)
        settings_env(EMBEDDING_MODEL="BAAI/bge-m3")
        runner = make_runner(docs_root)
        response = await restore(make_app(runner), bundle.name)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "bundle_incompatible"
        assert error["message"] == (
            f"cannot restore {bundle.name}: Embedded with test-model — this stack embeds with "
            "BAAI/bge-m3."
        )
        assert runner.snapshot() is None  # refused before any run existed

    async def test_a_damaged_bundle_is_422_invalid_and_then_lists_as_damaged(
        self, docs_root: Path
    ) -> None:
        """★ The list never hashes; the restore does — and remembers what it found."""
        bundle = exported_bundle(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            chunks = archive.read(CHUNKS_MEMBER)
        damaged = rezip(
            bundle, {CHUNKS_MEMBER: chunks.replace(b"chunk 1", b"chunk 7")}, reseal=False
        )
        bundle.unlink()
        runner = make_runner(docs_root)
        app = make_app(runner)

        (entry,) = (await request(app, "GET", "/api/backups/rag")).json()
        assert entry["name"] == damaged.name
        assert entry["restorable"] is True  # the cheap verdict can't see it

        response = await restore(app, damaged.name)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "bundle_invalid"
        assert "chunks.jsonl does not match its recorded sha256" in error["message"]
        assert runner.snapshot() is None

        (entry,) = (await request(app, "GET", "/api/backups/rag")).json()
        assert entry["restorable"] is False
        assert entry["reason"] == (
            "chunks.jsonl does not match its recorded sha256 — the bundle is damaged."
        )

        # A changed file is a new file: judged afresh by the cheap verdict.
        os.utime(damaged, (3_000_000, 3_000_000))
        (entry,) = (await request(app, "GET", "/api/backups/rag")).json()
        assert entry["restorable"] is True

    async def test_an_unsound_bundle_is_422_invalid(self, docs_root: Path) -> None:
        exports = docs_root / "varagity-exports"
        exports.mkdir()
        with zipfile.ZipFile(exports / "photos.zip", "w") as archive:
            archive.writestr("a.jpg", b"...")
        response = await restore(make_app(make_runner(docs_root)), "photos.zip")
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "bundle_invalid"
        assert (
            error["message"] == "cannot restore photos.zip: No bundle.json — not a Varagity backup."
        )

    async def test_a_busy_corpus_is_409_before_the_bundle_is_hashed(
        self, docs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = exported_bundle(docs_root)
        hashed: list[str] = []
        monkeypatch.setattr(
            backups_routes, "verify_bundle", lambda path, corpus: hashed.append(path.name)
        )
        gate = CorpusGate("rag")
        gate.claim("ingest")
        response = await restore(make_app(make_runner(docs_root, gate=gate)), bundle.name)
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "corpus_busy"
        assert "an ingest is running" in error["message"]
        assert hashed == []

    async def test_an_ingest_in_another_process_is_409(self, docs_root: Path) -> None:
        bundle = exported_bundle(docs_root)
        runner = make_runner(docs_root)
        response = await restore(make_app(runner, lock_free=False), bundle.name)
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "corpus_busy"
        assert "an ingest in another process holds the corpus lock" in error["message"]
        assert runner.snapshot() is None

    async def test_losing_the_race_for_the_gate_is_409(self, docs_root: Path) -> None:
        """Free at the pre-check, claimed by the time the run starts."""
        bundle = exported_bundle(docs_root)
        gate = CorpusGate("rag")
        runner = make_runner(docs_root, gate=gate)
        real_check = runner.ensure_free

        def check_then_lose() -> None:
            real_check()
            gate.claim("export")  # another operation slips in after the check

        runner.ensure_free = check_then_lose  # type: ignore[method-assign]
        response = await restore(make_app(runner), bundle.name)
        assert response.status_code == 409
        assert "an export is running" in response.json()["error"]["message"]

    async def test_unreachable_stores_are_a_structured_503(self, docs_root: Path) -> None:
        calls: list[tuple[str, str]] = []

        async def down(corpus: str, operation: str) -> None:
            calls.append((corpus, operation))
            raise HTTPException(
                status_code=503,
                detail={"code": "es_unreachable", "message": "elasticsearch unreachable"},
            )

        response = await restore(make_app(make_runner(docs_root), preflight=down), "x.zip")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "es_unreachable"
        assert calls == [("rag", "restore")]

    @pytest.mark.parametrize("body", [{}, {"name": ""}, {"name": "a.zip", "force": True}])
    async def test_a_malformed_body_is_a_validation_error(
        self, docs_root: Path, body: dict[str, Any]
    ) -> None:
        response = await request(
            make_app(make_runner(docs_root)), "POST", "/api/backups/rag/restore", json=body
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


class TestListedVerdicts:
    async def test_a_bundle_made_under_this_stacks_settings_has_no_drift(
        self, docs_root: Path
    ) -> None:
        export_rag(
            docs_root / "varagity-exports",
            corpus_store(docs_root),
            backups_routes.current_ingest_settings(),
            stale=False,
        )
        (entry,) = (
            await request(make_app(make_runner(docs_root)), "GET", "/api/backups/rag")
        ).json()
        assert entry["restorable"] is True
        assert entry["reason"] is None
        assert entry["drift"] == []

    async def test_settings_the_bundle_never_recorded_count_as_drift(self, docs_root: Path) -> None:
        exported_bundle(docs_root)  # records only CHUNK_SIZE and CONTEXTUALIZE
        (entry,) = (
            await request(make_app(make_runner(docs_root)), "GET", "/api/backups/rag")
        ).json()
        assert "CHUNKING_STRATEGY" in entry["drift"] and "OCR_ENGINE" in entry["drift"]

    async def test_drift_names_the_settings_that_differ(
        self, docs_root: Path, settings_env: Callable[..., None]
    ) -> None:
        exported = export_rag(
            docs_root / "varagity-exports",
            corpus_store(docs_root),
            backups_routes.current_ingest_settings(),
            stale=False,
        )
        settings_env(CHUNK_SIZE=800)
        (entry,) = (
            await request(make_app(make_runner(docs_root)), "GET", "/api/backups/rag")
        ).json()
        assert entry["name"] == exported.bundle.name
        assert entry["restorable"] is True  # drift flags stale; it never refuses
        assert entry["drift"] == ["CHUNK_SIZE"]

    async def test_an_incompatible_bundle_lists_its_reason(
        self, docs_root: Path, settings_env: Callable[..., None]
    ) -> None:
        exported_bundle(docs_root)
        settings_env(EMBEDDING_DIM=8)
        (entry,) = (
            await request(make_app(make_runner(docs_root)), "GET", "/api/backups/rag")
        ).json()
        assert entry["restorable"] is False
        assert entry["reason"] == "4-dimensional embeddings — this stack embeds into 8 dimensions."
        assert entry["manifest"]["counts"] == {"documents": 1, "chunks": 2}

    async def test_a_hand_copied_zip_lists_why_it_cannot_restore(self, docs_root: Path) -> None:
        exports = docs_root / "varagity-exports"
        exports.mkdir()
        with zipfile.ZipFile(exports / "holiday-photos.zip", "w") as archive:
            archive.writestr("a.jpg", b"...")
        (entry,) = (
            await request(make_app(make_runner(docs_root)), "GET", "/api/backups/rag")
        ).json()
        assert entry["manifest"] is None
        assert entry["restorable"] is False
        assert entry["reason"] == "No bundle.json — not a Varagity backup."
        assert entry["drift"] == []

    def test_the_damage_memory_keys_by_name_size_and_mtime(self, docs_root: Path) -> None:
        from varagity.backup.bundle import list_bundles

        bundle = exported_bundle(docs_root)
        memory = DamagedBundles()
        (listing,) = list_bundles(bundle.parent, "rag")
        assert memory.reason("rag", listing) is None
        memory.remember("rag", bundle.name, bundle, "broken")
        assert memory.reason("rag", listing) == "broken"
        assert memory.reason("graph", listing) is None  # per corpus
        memory.remember("rag", "gone.zip", docs_root / "gone.zip", "never stat-able")  # no-op
        assert get_damaged_bundles() is get_damaged_bundles()  # one per process
