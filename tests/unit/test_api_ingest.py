"""Unit tests for the ingest runner and its API routes (spec_v2 §4.2).

The runner executes an injected flow against injected base stages, so no
Prefect/stores/models are touched; what's under test is the machinery the
plan added — one-run-at-a-time, the emitting stage wrappers (including the
contextualize per-chunk ticks), the ``on_file`` outcome frames, log
relaying, replay-then-live subscription, and the structured route errors.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.sse import parse_sse
from varagity.api.deps import get_ingest_preflight
from varagity.api.ingest_runner import (
    EVENT_LOG,
    EVENT_PROGRESS,
    EVENT_STATUS,
    Feed,
    IngestAlreadyRunning,
    IngestRunner,
    get_ingest_runner,
)
from varagity.api.main import create_app
from varagity.ingest.discovery import Buckets
from varagity.ingest.loader import IngestStages, IngestSummary
from varagity.ingest.parsers import RawDocument

TICK = 0.02  # worker-thread scheduling slack for wait loops


class ScriptedProgress:
    """Minimal rich-Progress double implementing the sub-bar protocol."""

    def __init__(self) -> None:
        self.advances = 0

    def add_task(self, description: str, total: float | None = None, **kwargs: Any) -> int:
        return 1

    def advance(self, task_id: int, advance: float = 1) -> None:
        self.advances += int(advance)

    def remove_task(self, task_id: int) -> None:
        return None


def scripted_base_stages() -> IngestStages:
    """Fake base stages honoring the loader's per-stage call signatures."""

    def discover(docs_path: str, verbose: int | None = None) -> Buckets:
        return Buckets(text_like=[Path("a.txt"), Path("b.txt")])

    def parse(parser: Any, path: Path, *, verbose: int) -> RawDocument:
        return RawDocument(text="text " * 50, source_meta={"file_name": path.name})

    def chunk(raw: RawDocument, *, verbose: int) -> list[Any]:
        return ["chunk-one", "chunk-two"]

    def contextualize(
        *,
        document_text: str,
        chunk_texts: list[str],
        llm: Any,
        file_name: str,
        progress: Any,
        verbose: int,
    ) -> list[str | None]:
        # Mirrors loader.contextualize_chunks' sub-bar protocol: the proxy
        # the runner injects must see one advance per chunk.
        task_id = progress.add_task(f"  ↳ contextualizing {file_name}", total=len(chunk_texts))
        for _ in chunk_texts:
            progress.advance(task_id)
        progress.remove_task(task_id)
        return ["blurb"] * len(chunk_texts)

    def embed(texts: list[str], *, embeddings: Any, verbose: int) -> list[list[float]]:
        return [[0.1] for _ in texts]

    def store(records: list[Any], vectors: list[Any], *, store: Any, bm25: Any) -> None:
        return None

    return IngestStages(
        discover=discover,
        parse=parse,
        chunk=chunk,
        contextualize=contextualize,
        embed=embed,
        store=store,
    )


def scripted_flow(
    *,
    summary: IngestSummary | None = None,
    gate: threading.Event | None = None,
    fail_with: Exception | None = None,
    log_lines: bool = False,
) -> Callable[..., IngestSummary]:
    """Build a flow double that drives the stage seam like the loader would.

    Args:
        summary: The counters to return (a 2-file success by default).
        gate: When given, the flow blocks on it before finishing (lets a
            test observe the ``running`` state deterministically).
        fail_with: Raise this after the first file instead of finishing.
        log_lines: Emit a loader-style log line (relay coverage).

    Returns:
        The ``ingest_flow``-compatible callable.
    """

    def flow(
        *,
        reingest: bool,
        verbose: int,
        stages: IngestStages,
        on_file: Callable[[Path, str, int], None],
    ) -> IngestSummary:
        progress = ScriptedProgress()
        buckets = stages.discover("/docs", verbose=verbose)
        files = list(buckets.text_like)
        if log_lines:
            logging.getLogger("varagity.ingest.loader").info(
                "%s: unchanged — skipping (already ingested)", files[0].name
            )
        for index, path in enumerate(files):
            raw = stages.parse(None, path, verbose=verbose)
            chunks = stages.chunk(raw, verbose=verbose)
            stages.contextualize(
                document_text=raw.text,
                chunk_texts=[str(c) for c in chunks],
                llm=None,
                file_name=path.name,
                progress=progress,
                verbose=verbose,
            )
            stages.embed([str(c) for c in chunks], embeddings=None, verbose=verbose)
            stages.store([SimpleNamespace(file_name=path.name)], [[0.1]], store=None, bm25=None)
            if fail_with is not None and index == 0:
                raise fail_with
            on_file(path, "ingested", len(chunks))
        if gate is not None:
            assert gate.wait(timeout=5)
        return summary if summary is not None else IngestSummary(discovered=2, ingested=2, chunks=4)

    return flow


def make_runner(**kwargs: Any) -> IngestRunner:
    kwargs.setdefault("base_stages", scripted_base_stages())
    kwargs.setdefault("on_reingest_complete", lambda: None)
    return IngestRunner(kwargs.pop("flow", scripted_flow()), **kwargs)


async def collect(runner: IngestRunner, timeout: float = 5.0) -> list[Feed]:
    async def drain() -> list[Feed]:
        return [item async for item in runner.subscribe()]

    return await asyncio.wait_for(drain(), timeout=timeout)


async def wait_terminal(runner: IngestRunner, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while (snapshot := runner.snapshot()) is None or snapshot.state == "running":
        assert asyncio.get_running_loop().time() < deadline, "run never finished"
        await asyncio.sleep(TICK)


class TestRunnerLifecycle:
    async def test_idle_runner_reports_a_single_idle_status(self) -> None:
        events = await collect(make_runner())
        assert [name for name, _ in events] == [EVENT_STATUS]
        assert events[0][1].run is None

    async def test_completed_run_replays_the_full_event_sequence(self) -> None:
        runner = make_runner(flow=scripted_flow(log_lines=True))
        started = runner.start(reingest=False)
        assert started.state == "running"
        await wait_terminal(runner)

        events = await collect(runner)
        names = [name for name, _ in events]
        assert names[0] == EVENT_STATUS and events[0][1].run.state == "running"
        assert names[-1] == EVENT_STATUS and events[-1][1].run.state == "completed"

        progress = [payload for name, payload in events if name == EVENT_PROGRESS]
        stages = [p.stage for p in progress]
        assert stages[0] == "discover" and progress[0].total == 2
        # Per file: parse → chunk → contextualize (start + 2 ticks) → embed
        # → store → file_done, twice.
        assert stages.count("parse") == 2
        assert stages.count("file_done") == 2
        ticks = [p for p in progress if p.stage == "contextualize" and p.file == "a.txt"]
        assert [(t.current, t.total) for t in ticks] == [(0, 2), (1, 2), (2, 2)]
        done_frames = [p for p in progress if p.stage == "file_done"]
        assert [(p.files_done, p.files_total) for p in done_frames] == [(1, 2), (2, 2)]
        assert done_frames[0].outcome == "ingested"

        logs = [payload for name, payload in events if name == EVENT_LOG]
        assert any("skipping" in log.message for log in logs)

        summary = events[-1][1].run.summary
        assert summary is not None and summary.ingested == 2 and summary.chunks == 4

    async def test_second_start_while_running_raises(self) -> None:
        gate = threading.Event()
        runner = make_runner(flow=scripted_flow(gate=gate))
        runner.start(reingest=False)
        try:
            with pytest.raises(IngestAlreadyRunning):
                runner.start(reingest=False)
        finally:
            gate.set()
        await wait_terminal(runner)
        runner.start(reingest=False)  # a terminal run frees the slot
        await wait_terminal(runner)

    async def test_live_subscription_streams_until_terminal(self) -> None:
        gate = threading.Event()
        runner = make_runner(flow=scripted_flow(gate=gate))
        runner.start(reingest=False)

        async def consume() -> list[Feed]:
            items: list[Feed] = []
            async for item in runner.subscribe():
                items.append(item)
                if item[0] == EVENT_STATUS and items[-1][1].run is not None:
                    gate.set()  # release the flow once we're demonstrably live
            return items

        events = await asyncio.wait_for(consume(), timeout=5)
        assert events[-1][0] == EVENT_STATUS
        assert events[-1][1].run.state == "completed"

    async def test_failed_flow_reports_error_and_frees_the_slot(self) -> None:
        runner = make_runner(flow=scripted_flow(fail_with=RuntimeError("es fell over")))
        runner.start(reingest=False)
        await wait_terminal(runner)
        snapshot = runner.snapshot()
        assert snapshot is not None
        assert snapshot.state == "failed"
        assert "es fell over" in (snapshot.error or "")
        assert snapshot.summary is None
        events = await collect(runner)
        assert events[-1][1].run.state == "failed"

    async def test_reingest_completion_fires_the_stale_clear_hook(self) -> None:
        cleared: list[bool] = []
        runner = make_runner(on_reingest_complete=lambda: cleared.append(True))
        runner.start(reingest=True)
        await wait_terminal(runner)
        assert cleared == [True]

    async def test_plain_ingest_does_not_fire_the_stale_clear_hook(self) -> None:
        cleared: list[bool] = []
        runner = make_runner(on_reingest_complete=lambda: cleared.append(True))
        runner.start(reingest=False)
        await wait_terminal(runner)
        assert cleared == []

    async def test_failed_reingest_does_not_clear_stale(self) -> None:
        cleared: list[bool] = []
        runner = make_runner(
            flow=scripted_flow(fail_with=RuntimeError("boom")),
            on_reingest_complete=lambda: cleared.append(True),
        )
        runner.start(reingest=True)
        await wait_terminal(runner)
        assert cleared == []


@pytest.fixture
def app() -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_ingest_runner] = lambda: make_runner()

    async def _noop_preflight() -> None:
        return None

    application.dependency_overrides[get_ingest_preflight] = lambda: _noop_preflight
    return application


class TestRoutes:
    async def test_post_returns_a_202_run_handle(self, app: FastAPI) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            response = await client.post("/api/ingest", json={"reingest": True})
        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "running"
        assert body["reingest"] is True
        assert body["run_id"]

    async def test_second_post_while_running_is_a_structured_409(self) -> None:
        gate = threading.Event()
        runner = make_runner(flow=scripted_flow(gate=gate))
        application = create_app()
        application.dependency_overrides[get_ingest_runner] = lambda: runner

        async def _noop() -> None:
            return None

        application.dependency_overrides[get_ingest_preflight] = lambda: _noop
        try:
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
                first = await client.post("/api/ingest", json={})
                second = await client.post("/api/ingest", json={})
            assert first.status_code == 202
            assert second.status_code == 409
            assert second.json()["error"]["code"] == "ingest_already_running"
        finally:
            gate.set()
        await wait_terminal(runner)

    async def test_preflight_failure_is_a_structured_503(self, app: FastAPI) -> None:
        from fastapi import HTTPException

        async def down() -> None:
            raise HTTPException(
                status_code=503,
                detail={"code": "es_unreachable", "message": "elasticsearch unreachable"},
            )

        app.dependency_overrides[get_ingest_preflight] = lambda: down
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            response = await client.post("/api/ingest", json={})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "es_unreachable"

    async def test_status_stream_replays_a_terminal_run_and_closes(self) -> None:
        runner = make_runner()
        runner.start(reingest=False)
        await wait_terminal(runner)

        application = create_app()
        application.dependency_overrides[get_ingest_runner] = lambda: runner
        transport = httpx.ASGITransport(app=application)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://api") as client,
            client.stream("GET", "/api/ingest/status") as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join([chunk async for chunk in response.aiter_text()])

        events = parse_sse(body)
        names = [name for name, _ in events]
        assert names[0] == "status" and events[0][1]["run"]["state"] == "running"
        assert names[-1] == "status" and events[-1][1]["run"]["state"] == "completed"
        assert "progress" in names
        file_done = [
            data for name, data in events if name == "progress" and data["stage"] == "file_done"
        ]
        assert file_done and file_done[-1]["files_done"] == 2

    async def test_status_stream_is_a_single_idle_frame_with_no_run(self, app: FastAPI) -> None:
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://api") as client,
            client.stream("GET", "/api/ingest/status") as response,
        ):
            body = "".join([chunk async for chunk in response.aiter_text()])
        events = parse_sse(body)
        assert len(events) == 1
        name, data = events[0]
        assert name == "status" and data["run"] is None
