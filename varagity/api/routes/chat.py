"""``POST /api/chat`` — the streaming question endpoint (spec_v2 §4.2, §4.3).

Reuses the pipeline, never reimplements it: the request runs
:func:`~varagity.pipeline.query_flow.query_stream_flow` (every stage a
tracked Prefect task, exactly like the CLI) in a worker thread, while this
module's async generator relays the flow's callback events to the client as
typed SSE frames — ``retrieval`` first (the evidence before the prose),
then ``reasoning``/``token`` deltas, then ``done`` after the turn persists.

``corpus="graph"`` (spec_graphrag §4.2 — the field a router will later fill)
swaps the flow for :func:`~varagity.pipeline.graph_flow.graph_query_stream_flow`
and nothing else: same protocol, same event order, same persistence, with
the ``retrieval`` event carrying graph evidence instead of chunks. Whether
the graph *can* answer is settled in :func:`prepare_chat`, because ADR-017's
degrade (kill switch off, engine unavailable → answer from the chunk corpus,
requested corpus still recorded) has to be decided before any evidence is on
the wire.

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
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.sse import EventSourceResponse, ServerSentEvent

from varagity.api.deps import (
    get_chat_engine_resolver,
    get_conversation_store_factory,
    get_graph_chat_preflight,
    get_llm,
    get_retriever_resolver,
    get_services_preflight,
)
from varagity.api.schemas import (
    ChatOverrides,
    ChatRequest,
    DoneEvent,
    ErrorResponse,
    GraphEntityOut,
    GraphRelationOut,
    GraphRetrievalPayload,
    GraphTranscriptOut,
    RetrievalEvent,
    StatsEvent,
    UsageInfo,
)
from varagity.api.streaming import (
    EventBridge,
    delta_event,
    done_event,
    error_event,
    retrieval_event,
    stats_event,
)
from varagity.chat.base import ChatEngine, PreparedQuery, Turn
from varagity.config import get_settings
from varagity.graph.records import GraphRetrieval
from varagity.graph.service import GraphService, get_graph_service
from varagity.models.llm import GenerationTimings, LLMClient
from varagity.models.stream import Kind
from varagity.pipeline import graph_query_stream_flow, query_stream_flow
from varagity.pipeline.graph_flow import StreamedGraphState
from varagity.pipeline.query_flow import StreamedQueryState
from varagity.retrieval.base import Retriever
from varagity.stores.conversation_store import ConversationStore
from varagity.stores.records import GraphSourceSnapshot, RetrievedChunk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Throughput readings below this many decoded tokens are discarded. The
# model server's counters start at predicted_n=1 / predicted_ms=0.001,
# which computes to a literal 1,000,000 tok/s — real, and meaningless.
# The average needs a few tokens before it means anything.
_STATS_WARMUP_TOKENS = 8

# Floor on the gap between `stats` frames. The server reports timings on
# every chunk; at ~56 tok/s that is a frame per ~18 ms, which would swamp
# the deltas the user actually came for. 250 ms reads as live.
_STATS_MIN_INTERVAL_S = 0.25

# Per-transcript cap on the evidence excerpt, both on the wire and in the
# persisted snapshot. A retrieved graph passage is a slice of a whole
# thread-day (rendered up to
# `varagity.graph.render.DEFAULT_TRANSCRIPT_MAX_CHARS`), where a chunk-RAG
# chunk is a few hundred characters — uncapped, a ten-transcript turn would
# put ~80 kB into one SSE frame and the same again into every transcript
# read. Generous enough that a day's conversation still reads as one.
_EXCERPT_MAX_CHARS = 4000


@dataclass
class ChatPlan:
    """A validated, ready-to-run chat request.

    Assembled by :func:`prepare_chat` so every failure that *can* be
    detected before streaming is raised while the response status can
    still express it — including the graph degrade decision, which is
    settled here rather than mid-stream (ADR-017's semantics say a degraded
    turn answers from the chunk corpus, and by the time evidence has been
    emitted it is too late to change corpus).

    Attributes:
        payload: The validated request body.
        retriever: The resolved retrieval method.
        method: Its registry name (recorded on the turn).
        top_k: Chunks to retrieve (override or settings default).
        reranked_to: ``RERANK_TOP_N`` when the ``reranked`` method narrows
            the list; ``None`` otherwise.
        engine: The resolved chat engine (override or
            ``settings.CHAT_ENGINE``).
        engine_name: Its registry name (recorded on the turn, like
            ``method``).
        history: The conversation's recent turns, oldest first (empty for
            a fresh conversation) — the engine's condense input.
        llm: The chat client.
        store_factory: Conversation-store constructor (persistence runs in
            a worker thread with its own short-lived connection).
        corpus: The corpus the request asked for (``"rag"`` | ``"graph"``),
            persisted on the turn whatever actually answered it.
        graph_service: The opened graph service when a graph turn will
            really run on the graph; ``None`` for a chunk turn **and** for a
            degraded graph turn (kill switch off, engine unavailable) —
            which is what makes ``graph_service is None`` the single "answer
            from chunks" test in the streaming body.
        graph_mode: The engine query mode a graph turn retrieves with,
            resolved once here so a mid-turn settings PATCH cannot make the
            event and the persisted record disagree.
    """

    payload: ChatRequest
    retriever: Retriever
    method: str
    top_k: int
    reranked_to: int | None
    engine: ChatEngine
    engine_name: str
    history: list[Turn]
    llm: LLMClient
    store_factory: Callable[[], ConversationStore]
    corpus: str = "rag"
    graph_service: GraphService | None = None
    graph_mode: str = ""

    @property
    def retrieval_label(self) -> str:
        """What actually retrieved, in the corpus's own vocabulary.

        Returns:
            The graph query mode for a live graph turn, else the retrieval
            method's registry name — including for a *degraded* graph turn,
            which really did retrieve chunks (:class:`RetrievalEvent`'s
            ``method`` and the persisted ``retrieval_method`` both use this).
        """
        return self.graph_mode if self.graph_service is not None else self.method


@dataclass(frozen=True)
class _TurnOutcome:
    """One finished turn, whichever corpus produced it.

    The two flows return different state dicts (chunks vs. graph evidence);
    everything downstream of them — abort handling, ``done``, persistence —
    is identical. Normalizing once here keeps that shared tail written once.

    Attributes:
        prepared: The chat engine's search/answer query split.
        answer: The generated, ``<think>``-stripped answer.
        reasoning: The captured ``<think>`` stream (``""`` when none).
        aborted: ``True`` when the client disconnected mid-generation.
        usage: Server-reported token counts, or ``None``.
        tokens_per_second: Final decode throughput, or ``None``.
        chunks: The retrieved chunks (empty for a live graph turn).
        retrieval: The graph retrieval (``None`` for a chunk turn and for a
            degraded graph turn — which is exactly what makes the persisted
            ``graph_evidence`` NULL there).
    """

    prepared: PreparedQuery
    answer: str
    reasoning: str
    aborted: bool
    usage: dict[str, int] | None
    tokens_per_second: float | None
    chunks: list[RetrievedChunk]
    retrieval: GraphRetrieval | None

    @classmethod
    def from_chunks(cls, state: StreamedQueryState) -> "_TurnOutcome":
        """Normalize a chunk-RAG turn.

        Args:
            state: The streaming query flow's state.

        Returns:
            The outcome.
        """
        return cls(
            prepared=state["prepared"],
            answer=state["answer"],
            reasoning=state["reasoning"],
            aborted=state["aborted"],
            usage=state["usage"],
            tokens_per_second=state["tokens_per_second"],
            chunks=state["retrieved"],
            retrieval=None,
        )

    @classmethod
    def from_graph(cls, state: StreamedGraphState) -> "_TurnOutcome":
        """Normalize a graph turn.

        Args:
            state: The streaming graph flow's state.

        Returns:
            The outcome.
        """
        return cls(
            prepared=state["prepared"],
            answer=state["answer"],
            reasoning=state["reasoning"],
            aborted=state["aborted"],
            usage=state["usage"],
            tokens_per_second=state["tokens_per_second"],
            chunks=[],
            retrieval=state["retrieval"],
        )


def graph_retrieval_payload(retrieval: GraphRetrieval) -> GraphRetrievalPayload:
    """Map a graph retrieval onto the SSE ``retrieval`` event's graph half.

    Two things are dropped on the way out, both deliberately:
    :attr:`~varagity.graph.records.GraphEvidence.raw` (the engine-native
    payload, kept for autopsies, not for browsers) and everything past
    :data:`_EXCERPT_MAX_CHARS` of each passage — a transcript day can run to
    thousands of characters and the panel shows a card, not a file.

    Args:
        retrieval: What the graph returned.

    Returns:
        The wire payload.
    """
    return GraphRetrievalPayload(
        mode=retrieval.mode,
        entities=[
            GraphEntityOut(name=entity.name, type=entity.type, summary=entity.summary)
            for entity in retrieval.evidence.entities
        ],
        relations=[
            GraphRelationOut(
                source=relation.source,
                target=relation.target,
                label=relation.label,
                description=relation.description,
            )
            for relation in retrieval.evidence.relations
        ],
        transcripts=[
            GraphTranscriptOut(
                doc_key=excerpt.doc_key,
                thread_name=excerpt.thread_name,
                span=excerpt.span,
                excerpt=_clip(excerpt.text),
                message_count=len(excerpt.message_guids),
            )
            for excerpt in retrieval.excerpts
        ],
    )


def graph_evidence(retrieval: GraphRetrieval) -> dict[str, Any]:
    """Build the persisted ``messages.graph_evidence`` snapshot.

    The entities and relations, plus the mode that found them — the same
    shape the live event carries, minus the transcripts, which persist as
    ``message_sources`` rows instead (stage-2 decision #12).

    Args:
        retrieval: What the graph returned.

    Returns:
        The JSONB-ready snapshot.
    """
    payload = graph_retrieval_payload(retrieval)
    return payload.model_dump(mode="json", exclude={"transcripts"})


def graph_snapshots(retrieval: GraphRetrieval) -> list[GraphSourceSnapshot]:
    """Build the persisted evidence rows for the cited transcript days.

    Args:
        retrieval: What the graph returned.

    Returns:
        One snapshot per cited transcript day, best first.
    """
    return [
        GraphSourceSnapshot(
            doc_key=excerpt.doc_key,
            thread_name=excerpt.thread_name,
            span=excerpt.span,
            excerpt=_clip(excerpt.text),
            message_guids=list(excerpt.message_guids),
        )
        for excerpt in retrieval.excerpts
    ]


def _clip(text: str) -> str:
    """Cut a transcript passage to its wire/storage budget.

    Args:
        text: The retrieved passage.

    Returns:
        The passage, elided at :data:`_EXCERPT_MAX_CHARS` when longer.
    """
    if len(text) <= _EXCERPT_MAX_CHARS:
        return text
    return text[:_EXCERPT_MAX_CHARS].rstrip() + " […]"


class RecordingEngine:
    """Per-request engine wrapper that records the prepare outcome.

    The SSE ``retrieval`` event carries ``condensed_query`` (spec_v3
    §4.7), but the flow's ``on_retrieved`` callback only delivers chunks
    and the flow state only returns after the stream ends — too late. The
    condense stage runs strictly before retrieval in the same worker
    thread, so a delegating wrapper observes the
    :class:`~varagity.chat.base.PreparedQuery` exactly when the retrieval
    callback needs it, with no flow-signature change. Per-request by
    construction: registry engines are shared singletons and must not
    carry request state.
    """

    def __init__(self, inner: ChatEngine) -> None:
        """Wrap one resolved engine for one request.

        Args:
            inner: The registry engine that does the actual preparing.
        """
        self._inner = inner
        self.prepared: PreparedQuery | None = None

    def prepare(
        self,
        query: str,
        *,
        history: Sequence[Turn],
        llm: LLMClient | None,
        verbose: int,
    ) -> PreparedQuery:
        """Delegate to the wrapped engine, keeping its outcome readable.

        Args:
            query: The user's question, verbatim.
            history: Prior turns, oldest first.
            llm: Chat client for engines that condense.
            verbose: Validated console verbosity.

        Returns:
            The wrapped engine's split, unchanged.
        """
        self.prepared = self._inner.prepare(query, history=history, llm=llm, verbose=verbose)
        return self.prepared

    @property
    def condensed_query(self) -> str | None:
        """The rewritten search query, or ``None`` when none was used.

        Returns:
            ``search_query`` when the engine condensed this turn; ``None``
            before the condense stage ran and whenever the search used the
            user's words verbatim.
        """
        if self.prepared is None or not self.prepared.condensed:
            return None
        return self.prepared.search_query


async def _open_graph(service: GraphService) -> GraphService | None:
    """Open the graph session for a graph turn, or decide to degrade.

    ADR-017's degrade semantics, both halves: the kill switch is an INFO
    (the operator turned it off on purpose), an engine that will not open is
    a WARNING (something is wrong, but a graph question must still be
    answered from what the app *does* have). Neither ever fails the request.

    Opening here rather than mid-stream is the point: once the ``retrieval``
    event has been emitted the corpus is committed, so the last moment to
    fall back to chunk RAG is before the stream starts.

    Args:
        service: The process's graph service.

    Returns:
        The service when the graph will really answer, else ``None``.
    """
    if not get_settings().GRAPH_ENABLED:
        logger.info(
            "graph turn requested while GRAPH_ENABLED=false — answering from the chunk corpus"
        )
        return None
    try:
        await run_in_threadpool(service.session)
    except Exception as error:
        logger.warning(
            "graph turn degraded to the chunk corpus — the graph session would not open: %s",
            error,
        )
        return None
    return service


async def prepare_chat(
    payload: ChatRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    resolve_retriever: Annotated[Callable[[str], Retriever], Depends(get_retriever_resolver)],
    resolve_engine: Annotated[Callable[[str], ChatEngine], Depends(get_chat_engine_resolver)],
    store_factory: Annotated[
        Callable[[], ConversationStore], Depends(get_conversation_store_factory)
    ],
    services_preflight: Annotated[Callable[[], Awaitable[None]], Depends(get_services_preflight)],
    graph_service: Annotated[GraphService, Depends(get_graph_service)],
    graph_preflight: Annotated[Callable[[], Awaitable[None]], Depends(get_graph_chat_preflight)],
) -> ChatPlan:
    """Validate the request and resolve everything the stream will need.

    Checks run cheapest-first: body shape (FastAPI, 422 — an unknown
    ``corpus`` never gets past it) → retrieval-method resolution (422) →
    chat-engine resolution (override or settings, 422) → the graph degrade
    decision, which picks *which* dependencies the turn even needs →
    dependency reachability (503, *before* the stream opens) → conversation
    existence (404), whose store round-trip doubles as the history load for
    the chat engine (bounded by ``CONDENSE_HISTORY_TURNS``).

    A graph turn resolves the chunk retriever too: it is cheap, and it is
    what answers the turn if the graph degrades.

    Args:
        payload: The request body (FastAPI shares the parse with the route).
        llm: The chat client provider.
        resolve_retriever: The retrieval-method resolver.
        resolve_engine: The chat-engine resolver.
        store_factory: The conversation-store factory.
        services_preflight: The awaitable reachability check (raises the
            structured 503).
        graph_service: The process-wide graph service.
        graph_preflight: The graph turn's reachability check (Elasticsearch
            is not on that path).

    Returns:
        The assembled plan.

    Raises:
        HTTPException: ``422 unknown_retrieval_method`` /
            ``422 unknown_chat_engine`` for an override naming no
            registered implementation; ``503 <service>_unreachable`` for a
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
    engine_name = overrides.chat_engine or settings.CHAT_ENGINE
    try:
        engine = resolve_engine(engine_name)
    except KeyError as error:
        # Reachable only through the override: the settings value is
        # validator-gated to registered names at config load.
        raise HTTPException(
            status_code=422,
            detail={"code": "unknown_chat_engine", "message": str(error)},
        ) from error

    corpus = payload.corpus or "rag"
    # The degrade decision comes first because it decides *what* to preflight:
    # a graph turn never touches Elasticsearch, a degraded one always does.
    # Opening the session dials no service — it initializes local storages —
    # so nothing here depends on the check that follows it.
    graph = await _open_graph(graph_service) if corpus == "graph" else None
    await (graph_preflight() if graph is not None else services_preflight())

    history: list[Turn] = []
    if payload.conversation_id is not None:
        conversation_id = payload.conversation_id
        history_turns = settings.CONDENSE_HISTORY_TURNS

        def _load_history() -> list[tuple[str, str]] | None:
            with store_factory() as store:
                if not store.conversation_exists(conversation_id):
                    return None
                return store.recent_turns(conversation_id, limit=history_turns)

        turns = await run_in_threadpool(_load_history)
        if turns is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "conversation_not_found",
                    "message": f"No conversation with id {conversation_id!r}",
                },
            )
        history = [Turn(role=role, content=content) for role, content in turns]

    return ChatPlan(
        payload=payload,
        retriever=retriever,
        method=method,
        top_k=overrides.top_k or settings.TOP_K,
        reranked_to=settings.RERANK_TOP_N if method == "reranked" else None,
        engine=engine,
        engine_name=engine_name,
        history=history,
        llm=llm,
        store_factory=store_factory,
        corpus=corpus,
        graph_service=graph,
        graph_mode=settings.GRAPH_QUERY_MODE,
    )


def _swallow_outcome(future: "asyncio.Future[_TurnOutcome]") -> None:
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
    ``token`` deltas, interleaved with throttled ``stats`` frames while the
    model server reports throughput → ``done`` (ids, full answer, usage +
    per-stage latency). A failure after the stream opened emits an in-band
    ``error`` event instead. Aborted turns (client disconnect) persist
    nothing — the turn is persisted *at* ``done``.

    ``stats`` is optional in the protocol: it appears only when the model
    server reports its own decode counters (llama.cpp does; see
    :class:`~varagity.models.llm.GenerationTimings`), so clients must treat
    zero ``stats`` frames as normal rather than as a stalled stream.

    Args:
        plan: The validated request plan.

    Yields:
        The framed SSE events, in protocol order.
    """
    bridge = EventBridge()
    started = time.monotonic()
    timings: dict[str, int] = {}
    # Observes the condense outcome for the retrieval event: the condense
    # stage runs before retrieval in the flow's worker thread, so the
    # recorder is filled by the time _on_retrieved fires.
    engine = RecordingEngine(plan.engine)

    def _elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    def _emit_retrieval(
        chunks: list[RetrievedChunk], graph: GraphRetrievalPayload | None = None
    ) -> None:
        timings["retrieval"] = _elapsed_ms()
        bridge.emit_frame(
            retrieval_event(
                RetrievalEvent(
                    chunks=chunks,
                    method=plan.retrieval_label,
                    top_k=plan.top_k,
                    reranked_to=plan.reranked_to,
                    condensed_query=engine.condensed_query,
                    corpus=plan.corpus,
                    graph=graph,
                )
            )
        )

    def _on_retrieved(chunks: list[RetrievedChunk]) -> None:
        _emit_retrieval(chunks)

    def _on_graph_retrieved(retrieval: GraphRetrieval) -> None:
        _emit_retrieval([], graph_retrieval_payload(retrieval))

    def _on_delta(kind: Kind, text: str) -> None:
        bridge.emit_frame(delta_event(kind, text))

    last_stats_at = 0.0

    def _on_stats(timings: GenerationTimings) -> None:
        # Called from the flow's worker thread, once per model-server
        # chunk. Both gates live here rather than in the client: the
        # transport reports what the server said, the edge decides what is
        # worth a frame.
        nonlocal last_stats_at
        rate = timings.tokens_per_second
        if rate is None or timings.predicted_n < _STATS_WARMUP_TOKENS:
            return
        now = time.monotonic()
        if now - last_stats_at < _STATS_MIN_INTERVAL_S:
            return
        last_stats_at = now
        bridge.emit_frame(
            stats_event(StatsEvent(tokens_per_second=rate, completion_tokens=timings.predicted_n))
        )

    def _run_flow() -> _TurnOutcome:
        try:
            if plan.graph_service is not None:
                return _TurnOutcome.from_graph(
                    graph_query_stream_flow(
                        plan.payload.query,
                        plan.graph_service,
                        history=plan.history,
                        engine=engine,
                        llm=plan.llm,
                        mode=plan.graph_mode,
                        verbose=0,
                        on_retrieved=_on_graph_retrieved,
                        on_delta=_on_delta,
                        should_abort=bridge.should_abort,
                        on_stats=_on_stats,
                    )
                )
            return _TurnOutcome.from_chunks(
                query_stream_flow(
                    plan.payload.query,
                    history=plan.history,
                    engine=engine,
                    retriever=plan.retriever,
                    llm=plan.llm,
                    k=plan.top_k,
                    verbose=0,
                    on_retrieved=_on_retrieved,
                    on_delta=_on_delta,
                    should_abort=bridge.should_abort,
                    on_stats=_on_stats,
                )
            )
        finally:
            bridge.close()

    flow_future = asyncio.ensure_future(run_in_threadpool(_run_flow))
    flow_future.add_done_callback(_swallow_outcome)
    try:
        async for frame in bridge.events():
            yield frame
        outcome = await flow_future
        if outcome.aborted:
            logger.info("chat turn aborted by the client; nothing persisted")
            return

        timings["generation"] = _elapsed_ms() - timings.get("retrieval", 0)

        prepared = outcome.prepared
        retrieval = outcome.retrieval

        def _persist() -> tuple[str, str]:
            with plan.store_factory() as store:
                conversation_id = plan.payload.conversation_id
                if conversation_id is None:
                    conversation_id = store.create_conversation().conversation_id
                store.append_message(conversation_id, "user", plan.payload.query)
                message_id = store.append_message(
                    conversation_id,
                    "assistant",
                    outcome.answer,
                    retrieval_method=plan.retrieval_label,
                    latency_ms=dict(timings),
                    reasoning=outcome.reasoning or None,
                    # NULL = searched verbatim; the engine NAME is recorded
                    # even then, so a degraded condense_context turn (kill
                    # switch, fallback) stays attributable — spec_v3 §8.
                    condensed_query=(prepared.search_query if prepared.condensed else None),
                    chat_engine=plan.engine_name,
                    sources=outcome.chunks,
                    # Always the *requested* corpus, so a degraded graph turn
                    # stays attributable the same way (ADR-017); the NULL
                    # graph_evidence beside it is what says it degraded.
                    target_corpus=plan.corpus,
                    graph_evidence=None if retrieval is None else graph_evidence(retrieval),
                    graph_sources=() if retrieval is None else graph_snapshots(retrieval),
                )
                return conversation_id, message_id

        conversation_id, message_id = await run_in_threadpool(_persist)
        _auto_title_in_background(plan, conversation_id)
        timings["total"] = _elapsed_ms()
        usage = outcome.usage or {}
        yield done_event(
            DoneEvent(
                message_id=message_id,
                conversation_id=conversation_id,
                answer=outcome.answer,
                usage=UsageInfo(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    latency_ms=dict(timings),
                    tokens_per_second=outcome.tokens_per_second,
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
