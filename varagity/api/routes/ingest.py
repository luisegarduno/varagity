"""``POST /api/ingest`` + ``GET /api/ingest/status`` (spec_v2 §4.2).

Triggering returns a run handle immediately (the flow runs on a background
thread — see :mod:`varagity.api.ingest_runner`); the status endpoint is an
SSE stream of the run's events — a ``status`` snapshot, per-stage/per-file
``progress`` frames (with per-chunk contextualize ticks), relayed ``log``
lines, and the terminal ``status`` with the summary counters. The stream
replays from the start, so connecting mid-run or after completion renders
the same picture the CLI's ``rich`` display showed live.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from varagity.api.deps import get_ingest_preflight
from varagity.api.ingest_runner import IngestAlreadyRunning, IngestRunner, get_ingest_runner
from varagity.api.schemas import ErrorResponse, IngestRunOut, IngestStartRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

RunnerDep = Annotated[IngestRunner, Depends(get_ingest_runner)]


@router.post(
    "/api/ingest",
    status_code=202,
    responses={409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def start_ingest(
    payload: IngestStartRequest,
    runner: RunnerDep,
    preflight: Annotated[Callable[[], Awaitable[None]], Depends(get_ingest_preflight)],
) -> IngestRunOut:
    """Trigger a background ingest run and return its handle.

    Args:
        payload: ``reingest=true`` re-processes unchanged files (the
            stale-corpus action; setting changes don't change content
            hashes).
        runner: The process-wide ingest runner.
        preflight: The awaitable reachability check (structured 503).

    Returns:
        The new run's snapshot (state ``running``).

    Raises:
        HTTPException: ``409 ingest_already_running`` while a run is in
            flight; ``503 <service>_unreachable`` when a required backing
            service is down.
    """
    await preflight()
    try:
        return runner.start(reingest=payload.reingest)
    except IngestAlreadyRunning as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "ingest_already_running", "message": str(error)},
        ) from error


@router.get("/api/ingest/status", response_class=EventSourceResponse)
async def ingest_status(runner: RunnerDep) -> AsyncIterator[ServerSentEvent]:
    """Stream the current (or last) ingest run's progress as SSE.

    Frames: ``status`` (snapshot; also terminal, with the summary),
    ``progress`` (per stage/file/chunk), ``log`` (relayed pipeline lines).
    With no run in this API process the stream is a single idle ``status``
    frame; after a terminal run it replays that run and closes.

    Args:
        runner: The process-wide ingest runner.

    Yields:
        The framed events, oldest first, then live until terminal.
    """
    async for event, payload in runner.subscribe():
        yield ServerSentEvent(event=event, data=payload)
