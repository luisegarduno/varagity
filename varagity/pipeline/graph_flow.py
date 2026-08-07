"""Prefect graph flows: the build and the query path (spec_graphrag §5.2, §4.3).

**The build** is a thin ``@flow`` shell over
:meth:`varagity.graph.service.GraphService.build`, for the same reason
:mod:`varagity.pipeline.eval_flow` is one: the work itself belongs to the
service (which owns the session, the write lock, and the manifest diff),
while Prefect owns the *record* that a build ran, how long it took, and
whether it failed. A multi-day backfill with no run in the UI would be an
invisible job. Deliberately **one flow run per build attempt, not per
document**: the engine enqueues documents into its own durable status store
and processes them in batches inside a single call, so per-document task
runs would have to reach inside the engine's pipeline. Per-document progress
is reported instead by the runner's status sampler
(:mod:`varagity.api.graph_runner`), which reads the engine's own document
counts — tracking and streaming compose, as they do for ingest.

**The query path** is :func:`varagity.pipeline.query_flow.query_stream_flow`
with one stage swapped: condense → *graph retrieve* → stream the answer. The
condense and generation stages are the **same tasks** the chunk path runs, so
a graph turn inherits the search/answer query split (the original words
always drive the prompt — spec_v3 §4.2), the ``[SOURCE]`` citation contract,
``<think>`` splitting, client-abort handling, and token accounting without a
second implementation of any of them. Only the middle stage differs, because
only the middle stage is actually different.

Parameter validation is off (the service is a live handle, not a
serializable payload) and there are no Prefect-level retries: an engine that
fails mid-backfill must surface, not silently re-run hours of extraction —
and a re-called build resumes from the durable statuses anyway. The query
path carries none for the query-path reason instead: it is interactive, and
the clients already retry transient HTTP failures themselves.
"""

from collections.abc import Callable, Sequence
from typing import TypedDict

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

from varagity.chat.base import ChatEngine, PreparedQuery, Turn
from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.graph.answer import answer_context_max_chars, graph_answer_context
from varagity.graph.records import BuildReport, GraphRetrieval
from varagity.graph.service import GraphService
from varagity.graph.sources.base import MessageBatch
from varagity.models.llm import GenerationTimings, LLMClient
from varagity.models.stream import Kind
from varagity.pipeline.query_flow import condense_query_task, generate_answer_stream_task


@flow(name="graph-build", validate_parameters=False)
def graph_build_flow(
    service: GraphService,
    batches: Sequence[MessageBatch],
    *,
    prune_removed: bool = True,
    verbose: int = 0,
) -> BuildReport:
    """Upsert a parsed message corpus into the graph as a tracked flow run.

    Args:
        service: The process's graph service (holds the session and the
            single-flight write lock).
        batches: Parsed source files, guid-merged by the session before
            rendering.
        prune_removed: Whether ``batches`` render the whole corpus. A bounded
            build passes ``False`` — its render is partial on purpose, and
            pruning on its say-so would delete the rest of the archive.
        verbose: Validated console verbosity (0–2).

    Returns:
        What the build did (messages seen, wall clock, caught failures).
    """
    logger = get_run_logger()
    logger.info(
        "graph build starting: %d source file(s), %s render",
        len(batches),
        "full-corpus" if prune_removed else "bounded",
    )
    report = service.build(batches, verbose=verbose, prune_removed=prune_removed)
    logger.info(
        "graph build finished: %d message(s) in %.1fs, %d failure(s)",
        report.messages_seen,
        report.wall_clock_s,
        len(report.failures),
    )
    return report


@task(name="graph_retrieve", cache_policy=NO_CACHE)
def graph_retrieve_task(
    service: GraphService, query: str, *, mode: str, verbose: int
) -> GraphRetrieval:
    """Task wrapper over one graph retrieval (spec_graphrag §4.3).

    The chunk path's embed+retrieve stages collapse into this one: the
    engine owns its own keyword extraction, embedding, and fusion inside a
    single call, and splitting the wrapper would only invent task boundaries
    the engine does not actually have.

    Args:
        service: The process's graph service (reads take no build lock, so
            this is answered during a backfill).
        query: The search query — the chat engine's ``search_query``.
        mode: The engine query mode (``GRAPH_QUERY_MODE``).
        verbose: Validated console verbosity.

    Returns:
        The evidence and transcript passages the answer will be grounded in.
    """
    retrieval = service.retrieve(query, mode=mode, verbose=verbose)
    logger = get_run_logger()
    if retrieval.excerpts or retrieval.evidence.relations:
        logger.info(
            "graph retrieved %d entities, %d relations, %d transcript(s) (mode %s)",
            len(retrieval.evidence.entities),
            len(retrieval.evidence.relations),
            len(retrieval.excerpts),
            retrieval.mode,
        )
    else:
        logger.warning("the graph returned no evidence — is the corpus built?")
    return retrieval


class StreamedGraphState(TypedDict):
    """Outcome of one streamed graph turn (the chunk path's state, graph-shaped).

    Attributes:
        query: The user's question.
        prepared: The chat engine's search/answer query split (spec_v3
            §4.2) — the same split the chunk path makes.
        retrieval: What the graph returned (the SSE event's payload and the
            turn's persisted evidence both come from here).
        formatted_context: The grounding context the answer was written
            from.
        answer: The generated, ``<think>``-stripped answer.
        reasoning: The captured ``<think>`` stream (``""`` when none).
        aborted: ``True`` when the client aborted generation mid-stream.
        usage: Server-reported token counts, or ``None`` when unreported.
        tokens_per_second: Final decode throughput, or ``None``.
    """

    query: str
    prepared: PreparedQuery
    retrieval: GraphRetrieval
    formatted_context: str
    answer: str
    reasoning: str
    aborted: bool
    usage: dict[str, int] | None
    tokens_per_second: float | None


@flow(name="graph-query-stream", validate_parameters=False)
def graph_query_stream_flow(
    query: str,
    service: GraphService,
    *,
    history: Sequence[Turn] = (),
    engine: ChatEngine | None = None,
    llm: LLMClient | None = None,
    mode: str | None = None,
    verbose: int | None = None,
    on_retrieved: Callable[[GraphRetrieval], None] | None = None,
    on_delta: Callable[[Kind, str], None],
    should_abort: Callable[[], bool] | None = None,
    on_stats: Callable[[GenerationTimings], None] | None = None,
) -> StreamedGraphState:
    """Answer one question from the graph, streaming generation deltas.

    :func:`varagity.pipeline.query_flow.query_stream_flow`'s graph twin,
    backing ``POST /api/chat`` with ``corpus="graph"``: condense (the same
    task, so a follow-up is rewritten into a standalone search query exactly
    as it is for chunk RAG) → graph retrieve → ``on_retrieved`` (the SSE
    ``retrieval`` event: evidence before prose) → the same streaming
    generate task over a graph-shaped context.

    Generation is deliberately the repo's own, not the engine's (ADR-017's
    retrieval-only decision): the answer is grounded in the transcript days
    the graph cited, cites them by ``[SOURCE]``, and streams.

    Args:
        query: The user's question.
        service: The process's graph service.
        history: Prior conversation turns, oldest first; the chat engine
            reads them when preparing the search query.
        engine: Chat engine preparing the search query; resolved from
            ``settings.CHAT_ENGINE`` when omitted.
        llm: Chat client; resolved via the model registry when omitted.
        mode: Engine query mode; defaults to ``settings.GRAPH_QUERY_MODE``.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.
        on_retrieved: Optional hook called with the graph retrieval before
            generation starts.
        on_delta: Called with each classified ``(kind, text)`` fragment.
        should_abort: Polled between deltas; ``True`` stops generation early
            and marks the state ``aborted``.
        on_stats: Called with llama.cpp's cumulative decode counters as they
            arrive.

    Returns:
        The completed :class:`StreamedGraphState`.

    Raises:
        ValueError: If ``verbose`` is invalid.
        KeyError: If ``settings.CHAT_ENGINE`` names an unregistered engine.
        varagity.graph.service.GraphUnavailable: If the engine's session
            cannot be opened (the route resolves this *before* streaming, so
            reaching it here means the session died mid-turn).
    """
    from varagity.chat import get_chat_engine  # deferred: keeps the import graph acyclic

    settings = get_settings()
    verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
    active_engine = engine if engine is not None else get_chat_engine(settings.CHAT_ENGINE)
    prepared = condense_query_task(active_engine, query, history=history, llm=llm, verbose=verbose)
    retrieval = graph_retrieve_task(
        service,
        prepared.search_query,
        mode=mode if mode is not None else settings.GRAPH_QUERY_MODE,
        verbose=verbose,
    )
    if on_retrieved is not None:
        on_retrieved(retrieval)
    formatted_context = graph_answer_context(
        retrieval,
        max_chars=answer_context_max_chars(settings.LLM_CONTEXT_TOKENS, settings.MAX_TOKENS),
    )
    result = generate_answer_stream_task(
        prepared.original_query,  # always the user's words (spec_v3 §4.2)
        [],  # the grounding is the graph's, pre-rendered above
        llm=llm,
        formatted_context=formatted_context,
        on_delta=on_delta,
        should_abort=should_abort,
        on_stats=on_stats,
        verbose=verbose,
    )
    return StreamedGraphState(
        query=query,
        prepared=prepared,
        retrieval=retrieval,
        formatted_context=formatted_context,
        answer=result["answer"],
        reasoning=result["reasoning"],
        aborted=result["aborted"],
        usage=result["usage"],
        tokens_per_second=result["tokens_per_second"],
    )
