"""``POST /api/chat`` — the streaming question endpoint (spec_v2 §4.2, §4.3).

Reuses the pipeline, never reimplements it: the request runs
:func:`~varagity.pipeline.query_flow.query_stream_flow` (every stage a
tracked Prefect task, exactly like the CLI) in a worker thread, while this
module's async generator relays the flow's callback events to the client as
typed SSE frames — ``retrieval`` first (the evidence before the prose),
then ``reasoning``/``token`` deltas, then ``done`` after the turn persists.

Failure surfaces split by stream state: dependency outages and bad
references are caught by dependencies *before* the stream opens (clean
structured ``503``/``404``/``422``); anything after the 200 flushed is an
in-band ``error`` event. A client disconnect cancels this generator, whose
cleanup flips the bridge's abort flag; the flow notices between tokens and
closes the LLM stream, freeing the GPU (spec_v2 §4.3 cancellation).
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.sse import EventSourceResponse, ServerSentEvent

from varagity.api.deps import (
    get_conversation_store_factory,
    get_llm,
    get_retriever_resolver,
    get_services_preflight,
)
from varagity.api.schemas import (
    ChatOverrides,
    ChatRequest,
    DoneEvent,
    ErrorResponse,
    RetrievalEvent,
    UsageInfo,
)
from varagity.api.streaming import (
    EventBridge,
    delta_event,
    done_event,
    error_event,
    retrieval_event,
)
from varagity.config import get_settings
from varagity.models.llm import LLMClient
from varagity.models.stream import Kind
from varagity.pipeline import query_stream_flow
from varagity.pipeline.query_flow import StreamedQueryState
from varagity.retrieval.base import Retriever
from varagity.stores.conversation_store import ConversationStore
from varagity.stores.records import RetrievedChunk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@dataclass
class ChatPlan:
    """A validated, ready-to-run chat request.

    Assembled by :func:`prepare_chat` so every failure that *can* be
    detected before streaming is raised while the response status can
    still express it.

    Attributes:
        payload: The validated request body.
        retriever: The resolved retrieval method.
        method: Its registry name (recorded on the turn).
        top_k: Chunks to retrieve (override or settings default).
        reranked_to: ``RERANK_TOP_N`` when the ``reranked`` method narrows
            the list; ``None`` otherwise.
        llm: The chat client.
        store_factory: Conversation-store constructor (persistence runs in
            a worker thread with its own short-lived connection).
    """

    payload: ChatRequest
    retriever: Retriever
    method: str
    top_k: int
    reranked_to: int | None
    llm: LLMClient
    store_factory: Callable[[], ConversationStore]


async def prepare_chat(
    payload: ChatRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    resolve_retriever: Annotated[Callable[[str], Retriever], Depends(get_retriever_resolver)],
    store_factory: Annotated[
        Callable[[], ConversationStore], Depends(get_conversation_store_factory)
    ],
    services_preflight: Annotated[Callable[[], Awaitable[None]], Depends(get_services_preflight)],
) -> ChatPlan:
    """Validate the request and resolve everything the stream will need.

    Checks run cheapest-first: body shape (FastAPI, 422) → retrieval-method
    resolution (422) → dependency reachability (503, *before* the stream
    opens) → conversation existence (404).

    Args:
        payload: The request body (FastAPI shares the parse with the route).
        llm: The chat client provider.
        resolve_retriever: The retrieval-method resolver.
        store_factory: The conversation-store factory.
        services_preflight: The awaitable reachability check (raises the
            structured 503).

    Returns:
        The assembled plan.

    Raises:
        HTTPException: ``422 unknown_retrieval_method`` for an override
            naming no registered method; ``503 <service>_unreachable`` for a
            down dependency; ``404 conversation_not_found`` for an unknown
            ``conversation_id``.
    """
    settings = get_settings()
    overrides = payload.overrides or ChatOverrides()
    method = overrides.retrieval_method or settings.RETRIEVAL_METHOD
    try:
        retriever = resolve_retriever(method)
    except KeyError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "unknown_retrieval_method", "message": str(error)},
        ) from error

    await services_preflight()

    if payload.conversation_id is not None:

        def _exists() -> bool:
            with store_factory() as store:
                return store.conversation_exists(payload.conversation_id or "")

        if not await run_in_threadpool(_exists):
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "conversation_not_found",
                    "message": f"No conversation with id {payload.conversation_id!r}",
                },
            )

    return ChatPlan(
        payload=payload,
        retriever=retriever,
        method=method,
        top_k=overrides.top_k or settings.TOP_K,
        reranked_to=settings.RERANK_TOP_N if method == "reranked" else None,
        llm=llm,
        store_factory=store_factory,
    )


def _swallow_outcome(future: "asyncio.Future[StreamedQueryState]") -> None:
    """Consume an abandoned flow future's outcome so asyncio doesn't warn.

    The stream side stops awaiting the flow when the client disconnects;
    whatever the flow then returns (or raises while winding down) has no
    consumer.

    Args:
        future: The completed flow future.
    """
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        logger.debug("abandoned chat flow ended with %s: %s", type(error).__name__, error)


@router.post(
    "/api/chat",
    response_class=EventSourceResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def chat(plan: Annotated[ChatPlan, Depends(prepare_chat)]) -> AsyncIterator[ServerSentEvent]:
    """Answer one question as a typed SSE stream (spec_v2 §4.3).

    Event order: ``retrieval`` (the provenance payload) → ``reasoning``/
    ``token`` deltas → ``done`` (ids, full answer, usage + per-stage
    latency). A failure after the stream opened emits an in-band ``error``
    event instead. Aborted turns (client disconnect) persist nothing — the
    turn is persisted *at* ``done``.

    Args:
        plan: The validated request plan.

    Yields:
        The framed SSE events, in protocol order.
    """
    bridge = EventBridge()
    started = time.monotonic()
    timings: dict[str, int] = {}

    def _elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    def _on_retrieved(chunks: list[RetrievedChunk]) -> None:
        timings["retrieval"] = _elapsed_ms()
        bridge.emit_frame(
            retrieval_event(
                RetrievalEvent(
                    chunks=chunks,
                    method=plan.method,
                    top_k=plan.top_k,
                    reranked_to=plan.reranked_to,
                )
            )
        )

    def _on_delta(kind: Kind, text: str) -> None:
        bridge.emit_frame(delta_event(kind, text))

    def _run_flow() -> StreamedQueryState:
        try:
            return query_stream_flow(
                plan.payload.query,
                retriever=plan.retriever,
                llm=plan.llm,
                k=plan.top_k,
                verbose=0,
                on_retrieved=_on_retrieved,
                on_delta=_on_delta,
                should_abort=bridge.should_abort,
            )
        finally:
            bridge.close()

    flow_future = asyncio.ensure_future(run_in_threadpool(_run_flow))
    flow_future.add_done_callback(_swallow_outcome)
    try:
        async for frame in bridge.events():
            yield frame
        state = await flow_future
        if state["aborted"]:
            logger.info("chat turn aborted by the client; nothing persisted")
            return

        timings["generation"] = _elapsed_ms() - timings.get("retrieval", 0)

        def _persist() -> tuple[str, str]:
            with plan.store_factory() as store:
                conversation_id = plan.payload.conversation_id
                if conversation_id is None:
                    conversation_id = store.create_conversation().conversation_id
                store.append_message(conversation_id, "user", plan.payload.query)
                message_id = store.append_message(
                    conversation_id,
                    "assistant",
                    state["answer"],
                    retrieval_method=plan.method,
                    latency_ms=dict(timings),
                    reasoning=state["reasoning"] or None,
                    sources=state["retrieved"],
                )
                return conversation_id, message_id

        conversation_id, message_id = await run_in_threadpool(_persist)
        _auto_title_in_background(plan, conversation_id)
        timings["total"] = _elapsed_ms()
        usage = state["usage"] or {}
        yield done_event(
            DoneEvent(
                message_id=message_id,
                conversation_id=conversation_id,
                answer=state["answer"],
                usage=UsageInfo(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    latency_ms=dict(timings),
                ),
            )
        )
    except Exception as error:
        # The 200 already flushed — failures from here are in-band frames.
        logger.exception("chat stream failed mid-flight")
        yield error_event("pipeline_error", f"{type(error).__name__}: {error}")
    finally:
        bridge.abort()


def _auto_title_in_background(plan: ChatPlan, conversation_id: str) -> None:
    """Fire-and-forget auto-titling so ``done`` never waits on a second LLM call.

    A reasoning model can take seconds over a one-line title; the client
    already has its answer. The worker owns a short-lived store connection
    and swallows every failure — titling must never break a turn.

    Args:
        plan: The request plan (store factory + LLM).
        conversation_id: The conversation to (maybe) title.
    """

    def _title() -> None:
        try:
            with plan.store_factory() as store:
                store.auto_title(conversation_id, plan.payload.query, llm=plan.llm)
        except Exception:  # pragma: no cover — best-effort by design
            logger.warning("background auto-title failed", exc_info=True)

    asyncio.get_running_loop().run_in_executor(None, _title)
