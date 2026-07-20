"""Prefect query flow: the spec §10.1 pipeline as tracked task runs.

Condense → embed → retrieve → generate, each a task run, returning the
spec §10.1 state dict. The condense stage (spec_v3 §4.2) runs the
configured chat engine over the turn and its history to decide the *search
query*; retrieval sees ``prepared.search_query`` while the answer prompt
always gets ``prepared.original_query`` — the user's words are never
rewritten for generation. The stage is always in the graph (plan decision
#14): under the ``simple`` engine it is a no-LLM pass-through, so a
tracked run is its only cost. Query embedding — encapsulated inside the
retrievers on the plain path
(:func:`varagity.generation.answer.answer_query`) — is hoisted into its
own tracked stage here via the retrievers'
:meth:`~varagity.retrieval.base.Retriever.encode_query` seam, which also
fills the state's ``query_vector`` (``None`` for ``bm25``: nothing to
encode).

:func:`query_stream_flow` is the flow's streaming twin (spec_v2 §4.3): the
same embed/retrieve stages, with generation swapped for a tracked task that
hands deltas to a callback while it runs — the task boundary (and its run
log) is preserved while tokens flow out to the HTTP API's SSE stream.

Unlike the ingestion flow's model/store tasks, these tasks carry no
Prefect-level retries: the query path is interactive, the clients already
retry transient HTTP failures internally (``tenacity``), and stacking task
retries on top would multiply the wait before a hard failure surfaces at
the prompt. Result caching is disabled (``NO_CACHE``) for the same reasons
as the ingestion tasks: live-service calls with unhashable client/retriever
inputs.

Both flows are Prometheus probe points (spec_v2 §6.2): they
time each stage around its task call, record the retrieved chunks' scores
(+ rerank movement), and count the flow's outcome — the streaming flow
additionally counts the server-reported token usage it already returns.
"""

import time
from collections.abc import Callable, Sequence

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

from varagity.chat import get_chat_engine
from varagity.chat.base import ChatEngine, PreparedQuery, Turn
from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.generation.answer import (
    QueryState,
    StreamedAnswer,
    format_context,
    generate_answer,
    generate_answer_stream,
)
from varagity.models.llm import GenerationTimings, LLMClient
from varagity.models.stream import Kind
from varagity.observability import metrics
from varagity.retrieval import get_retriever
from varagity.retrieval.base import RETRIEVER_REGISTRY, Retriever
from varagity.stores.records import RetrievedChunk


def _method_label(retriever: Retriever) -> str:
    """Resolve a retriever's registry name for metric labels.

    Args:
        retriever: The active retrieval method.

    Returns:
        The registry name (``semantic``/``bm25``/``hybrid``/``reranked``),
        or ``"custom"`` for an implementation the registry doesn't know
        (injected test/eval doubles) — metric labels must stay
        low-cardinality, so arbitrary class names never become labels.
    """
    for name, registered in RETRIEVER_REGISTRY.items():
        if type(retriever) is type(registered):
            return name
    return "custom"


@task(name="condense_query", cache_policy=NO_CACHE)
def condense_query_task(
    engine: ChatEngine,
    query: str,
    *,
    history: Sequence[Turn],
    llm: LLMClient | None,
    verbose: int,
) -> PreparedQuery:
    """Task wrapper over the chat engine's query preparation (spec_v3 §4.2).

    Always in the graph, whatever the engine (plan decision #14): ``simple``
    returns the identity split with no LLM call, so the tracked stage is
    ~free — and the registry stays a seam rather than a special case.

    Args:
        engine: The chat engine (owns the condense decision).
        query: The user's question, verbatim.
        history: Prior conversation turns, oldest first (empty on a first
            turn, and from history-less callers like the CLI).
        llm: Chat client for engines that condense; ``None`` lets them
            resolve one via the model registry.
        verbose: Validated console verbosity.

    Returns:
        The two-string split downstream stages retrieve and answer with.
    """
    prepared = engine.prepare(query, history=history, llm=llm, verbose=verbose)
    logger = get_run_logger()
    if prepared.condensed:
        logger.info("condensed the search query → %r", prepared.search_query)
    else:
        logger.info("search query is the user's words, verbatim")
    return prepared


@task(name="embed_query", cache_policy=NO_CACHE)
def embed_query_task(retriever: Retriever, query: str, *, verbose: int) -> list[float] | None:
    """Task wrapper over query encoding (spec §10.1 step 2).

    Args:
        retriever: The retrieval method (owns its query encoding).
        query: The user's question, unformatted.
        verbose: Validated console verbosity.

    Returns:
        The query embedding, or ``None`` for methods that never encode
        queries (``bm25``).
    """
    vector = retriever.encode_query(query, verbose=verbose)
    logger = get_run_logger()
    if vector is None:
        logger.info("retrieval method encodes no query vector")
    else:
        logger.info("embedded query → %d-dim vector", len(vector))
    return vector


@task(name="retrieve", cache_policy=NO_CACHE)
def retrieve_task(
    retriever: Retriever,
    query: str,
    *,
    k: int,
    query_vector: list[float] | None,
    verbose: int,
) -> list[RetrievedChunk]:
    """Task wrapper over retrieval (spec §10.1 steps 3–4).

    Args:
        retriever: The retrieval method.
        query: The user's question.
        k: Number of chunks to return.
        query_vector: The embed stage's output (reused instead of
            re-encoding; ignored by ``bm25``).
        verbose: Validated console verbosity.

    Returns:
        The top-k chunks, best first.
    """
    chunks = retriever.retrieve(query, k=k, verbose=verbose, query_vector=query_vector)
    logger = get_run_logger()
    if chunks:
        logger.info(
            "retrieved %d chunk(s); best: %s (score %.4f)",
            len(chunks),
            chunks[0].metadata.get("file_name", "<unknown>"),
            chunks[0].score,
        )
    else:
        logger.warning("retrieved no chunks — is the corpus ingested?")
    return chunks


@task(name="generate_answer", cache_policy=NO_CACHE)
def generate_answer_task(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    llm: LLMClient | None,
    formatted_context: str,
    verbose: int,
) -> str:
    """Task wrapper over grounded answer generation (spec §10.1 steps 5–6).

    Args:
        query: The user's question.
        chunks: The retrieved chunks grounding the answer.
        llm: Chat client; resolved via the model registry when ``None``.
        formatted_context: The pre-built context block (spec §10.2).
        verbose: Validated console verbosity.

    Returns:
        The generated, ``<think>``-stripped answer.
    """
    answer = generate_answer(
        query, chunks, llm=llm, formatted_context=formatted_context, verbose=verbose
    )
    get_run_logger().info("generated %d-char answer from %d chunk(s)", len(answer), len(chunks))
    return answer


@task(name="generate_answer_stream", cache_policy=NO_CACHE)
def generate_answer_stream_task(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    llm: LLMClient | None,
    formatted_context: str,
    on_delta: Callable[[Kind, str], None],
    should_abort: Callable[[], bool] | None,
    on_stats: Callable[[GenerationTimings], None] | None,
    verbose: int,
) -> StreamedAnswer:
    """Task wrapper over streamed answer generation (spec_v2 §4.3).

    The task boundary is preserved while deltas flow out through
    ``on_delta`` — the run log records the outcome exactly like the
    non-streaming twin. A client-side abort is a deliberate stop, not a
    failure: the task completes normally with ``aborted=True``.

    Args:
        query: The user's question.
        chunks: The retrieved chunks grounding the answer.
        llm: Chat client; resolved via the model registry when ``None``.
        formatted_context: The pre-built context block (spec §10.2).
        on_delta: Called with each classified ``(kind, text)`` fragment.
        should_abort: Polled between deltas; ``True`` stops generation.
        on_stats: Called with llama.cpp's decode counters as they arrive.
        verbose: Validated console verbosity.

    Returns:
        The completed :class:`~varagity.generation.answer.StreamedAnswer`.
    """
    result = generate_answer_stream(
        query,
        chunks,
        on_delta=on_delta,
        llm=llm,
        formatted_context=formatted_context,
        should_abort=should_abort,
        on_stats=on_stats,
        verbose=verbose,
    )
    logger = get_run_logger()
    if result["aborted"]:
        logger.info("generation aborted by the client after %d chars", len(result["answer"]))
    else:
        logger.info(
            "streamed %d-char answer (%d-char reasoning) from %d chunk(s)",
            len(result["answer"]),
            len(result["reasoning"]),
            len(chunks),
        )
    return result


class StreamedQueryState(QueryState):
    """The spec §10.1 state dict, extended with streaming outcomes.

    Attributes:
        reasoning: The captured ``<think>`` stream (``""`` when none).
        aborted: ``True`` when the client aborted generation mid-stream.
        usage: Server-reported token counts, or ``None`` when unreported.
        tokens_per_second: Final decode throughput reported by llama.cpp,
            or ``None`` when the model server reports no ``timings``.
    """

    reasoning: str
    aborted: bool
    usage: dict[str, int] | None
    tokens_per_second: float | None


@flow(name="query-stream", validate_parameters=False)
def query_stream_flow(
    query: str,
    *,
    history: Sequence[Turn] = (),
    engine: ChatEngine | None = None,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    k: int | None = None,
    verbose: int | None = None,
    on_retrieved: Callable[[list[RetrievedChunk]], None] | None = None,
    on_delta: Callable[[Kind, str], None],
    should_abort: Callable[[], bool] | None = None,
    on_stats: Callable[[GenerationTimings], None] | None = None,
) -> StreamedQueryState:
    """Answer one question with tracked stages, streaming generation deltas.

    :func:`query_flow`'s streaming twin, backing ``POST /api/chat``
    (spec_v2 §4.3): identical condense/embed/retrieve staging
    (``on_retrieved`` still fires before generation — the SSE ``retrieval``
    event), then the streaming generate task hands each classified delta to
    ``on_delta`` while it runs.

    Args:
        query: The user's question.
        history: Prior conversation turns, oldest first; the chat engine
            reads them when preparing the search query (empty by default —
            a first turn, or a history-less caller).
        engine: Chat engine preparing the search query; resolved from
            ``settings.CHAT_ENGINE`` when omitted.
        retriever: Retrieval method; resolved from
            ``settings.RETRIEVAL_METHOD`` when omitted.
        llm: Chat client; resolved via the model registry when omitted.
        k: Chunks to retrieve; defaults to ``settings.TOP_K``.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.
        on_retrieved: Optional hook called with the retrieved chunks before
            generation.
        on_delta: Called with each classified ``(kind, text)`` fragment, in
            stream order (``kind`` is ``"reasoning"`` or ``"answer"``).
        should_abort: Polled between deltas; returning ``True`` stops
            generation early and marks the state ``aborted``.
        on_stats: Called with llama.cpp's cumulative decode counters as
            they arrive (once per chunk — throttle before rendering);
            never fires on a server that reports no ``timings``.

    Returns:
        The completed :class:`StreamedQueryState`.

    Raises:
        ValueError: If ``verbose`` is invalid.
        KeyError: If ``settings.RETRIEVAL_METHOD`` names an unregistered
            retrieval method, or ``settings.CHAT_ENGINE`` an unregistered
            chat engine.
    """
    settings = get_settings()
    verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
    active_retriever = (
        retriever if retriever is not None else get_retriever(settings.RETRIEVAL_METHOD)
    )
    active_engine = engine if engine is not None else get_chat_engine(settings.CHAT_ENGINE)
    top_k = settings.TOP_K if k is None else k

    method = _method_label(active_retriever)
    try:
        stage_started = time.perf_counter()
        prepared = condense_query_task(
            active_engine, query, history=history, llm=llm, verbose=verbose
        )
        metrics.observe_query_stage("condense", method, time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        query_vector = embed_query_task(active_retriever, prepared.search_query, verbose=verbose)
        metrics.observe_query_stage("embed", method, time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        chunks = retrieve_task(
            active_retriever,
            prepared.search_query,
            k=top_k,
            query_vector=query_vector,
            verbose=verbose,
        )
        metrics.observe_query_stage("retrieve", method, time.perf_counter() - stage_started)
        metrics.observe_retrieval(method, chunks)
        if on_retrieved is not None:
            on_retrieved(chunks)
        formatted_context = format_context(chunks)
        stage_started = time.perf_counter()
        result = generate_answer_stream_task(
            prepared.original_query,  # always the user's words (spec_v3 §4.2)
            chunks,
            llm=llm,
            formatted_context=formatted_context,
            on_delta=on_delta,
            should_abort=should_abort,
            on_stats=on_stats,
            verbose=verbose,
        )
        metrics.observe_query_stage("generate", method, time.perf_counter() - stage_started)
    except Exception:
        metrics.count_query(method, "error")
        raise
    metrics.count_query(method, "aborted" if result["aborted"] else "ok")
    usage = result["usage"] or {}
    metrics.count_llm_tokens(usage.get("prompt_tokens"), usage.get("completion_tokens"))
    return StreamedQueryState(
        query=query,
        prepared=prepared,
        query_vector=query_vector,
        retrieved=chunks,
        formatted_context=formatted_context,
        answer=result["answer"],
        reasoning=result["reasoning"],
        aborted=result["aborted"],
        usage=result["usage"],
        tokens_per_second=result["tokens_per_second"],
    )


@flow(name="query", validate_parameters=False)
def query_flow(
    query: str,
    *,
    history: Sequence[Turn] = (),
    engine: ChatEngine | None = None,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    k: int | None = None,
    verbose: int | None = None,
    on_retrieved: Callable[[list[RetrievedChunk]], None] | None = None,
) -> QueryState:
    """Answer one question with every stage tracked as a Prefect task run.

    The tracked twin of :func:`varagity.generation.answer.answer_query`
    (same state dict); the composition differs in query embedding and chat
    engine preparation each running as their own stage. Parameter
    validation is off because callers inject duck-typed retriever/client
    fakes that pydantic would reject; the flow's inputs are
    already-validated internals.

    Args:
        query: The user's question.
        history: Prior conversation turns, oldest first; the chat engine
            reads them when preparing the search query (empty by default —
            a first turn, or a history-less caller).
        engine: Chat engine preparing the search query; resolved from
            ``settings.CHAT_ENGINE`` when omitted.
        retriever: Retrieval method; resolved from
            ``settings.RETRIEVAL_METHOD`` when omitted.
        llm: Chat client; resolved via the model registry when omitted.
        k: Chunks to retrieve; defaults to ``settings.TOP_K``.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.
        on_retrieved: Optional hook called with the retrieved chunks before
            generation (the CLI's matches table — spec §10.1 step 4).

    Returns:
        The completed :class:`~varagity.generation.answer.QueryState`.

    Raises:
        ValueError: If ``verbose`` is invalid.
        KeyError: If ``settings.RETRIEVAL_METHOD`` names an unregistered
            retrieval method, or ``settings.CHAT_ENGINE`` an unregistered
            chat engine.
    """
    settings = get_settings()
    verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
    active_retriever = (
        retriever if retriever is not None else get_retriever(settings.RETRIEVAL_METHOD)
    )
    active_engine = engine if engine is not None else get_chat_engine(settings.CHAT_ENGINE)
    top_k = settings.TOP_K if k is None else k

    method = _method_label(active_retriever)
    try:
        stage_started = time.perf_counter()
        prepared = condense_query_task(
            active_engine, query, history=history, llm=llm, verbose=verbose
        )
        metrics.observe_query_stage("condense", method, time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        query_vector = embed_query_task(active_retriever, prepared.search_query, verbose=verbose)
        metrics.observe_query_stage("embed", method, time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        chunks = retrieve_task(
            active_retriever,
            prepared.search_query,
            k=top_k,
            query_vector=query_vector,
            verbose=verbose,
        )
        metrics.observe_query_stage("retrieve", method, time.perf_counter() - stage_started)
        metrics.observe_retrieval(method, chunks)
        if on_retrieved is not None:
            on_retrieved(chunks)
        formatted_context = format_context(chunks)
        stage_started = time.perf_counter()
        answer = generate_answer_task(
            prepared.original_query,  # always the user's words (spec_v3 §4.2)
            chunks,
            llm=llm,
            formatted_context=formatted_context,
            verbose=verbose,
        )
        metrics.observe_query_stage("generate", method, time.perf_counter() - stage_started)
    except Exception:
        metrics.count_query(method, "error")
        raise
    metrics.count_query(method, "ok")
    return QueryState(
        query=query,
        prepared=prepared,
        query_vector=query_vector,
        retrieved=chunks,
        formatted_context=formatted_context,
        answer=answer,
    )
