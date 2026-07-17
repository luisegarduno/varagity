"""Context prompt & grounded answer generation (spec §10.2).

Each retrieved chunk is formatted with its provenance
(``[SOURCE]/[CONTEXT]/[CONTENT]``), joined into ``formatted_context``, and
fed to the LLM under the grounding prompt: answer from the context only,
admit ignorance otherwise, cite sources. :func:`answer_query` threads the
whole query pipeline through the spec §10.1 state dict.
:func:`generate_answer_stream` is :func:`generate_answer`'s streaming twin
(spec_v2 §4.3): same prompt, but deltas flow to a callback as they arrive.
"""

from collections.abc import Callable
from typing import TypedDict

from openai.types import CompletionUsage

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.models.llm import GenerationTimings, LLMClient, clean_response
from varagity.models.registry import get_model
from varagity.models.stream import Kind, ThinkStreamSplitter
from varagity.retrieval.base import Retriever, get_retriever
from varagity.stores.records import RetrievedChunk

# Per-chunk provenance block (spec §10.2, column-aligned as specified).
# [CONTEXT] renders empty for chunks ingested without a situating blurb
# (CONTEXTUALIZE off) — the format is stable either way.
_CHUNK_BLOCK = "[SOURCE]:  {source}\n[CONTEXT]: {context}\n[CONTENT]: {content}"

# The grounding prompt (spec §10.2, verbatim).
ANSWER_PROMPT = """You are Varagity, a retrieval-augmented assistant.
Answer the user's QUESTION using ONLY the CONTEXT below.
If the answer is not contained in the context, say you don't know — do not fabricate.
Cite the [SOURCE] of any facts you use.

<context>
{formatted_context}
</context>

QUESTION: {query}
ANSWER:"""


class QueryState(TypedDict):
    """State threaded through the query pipeline (spec §10.1).

    Attributes:
        query: The user's question.
        query_vector: The query embedding. ``None`` on the plain path
            (:func:`answer_query`), where the retrieval seam encapsulates
            query encoding (bm25 has no vector at all); the Prefect
            ``query_flow`` runs embedding as its own tracked stage and
            fills it.
        retrieved: The retrieved chunks with scores, best first.
        formatted_context: The provenance-formatted context fed to the LLM.
        answer: The generated, ``<think>``-stripped answer.
    """

    query: str
    query_vector: list[float] | None
    retrieved: list[RetrievedChunk]
    formatted_context: str
    answer: str


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into the context block (spec §10.2).

    Args:
        chunks: The retrieved chunks, best first.

    Returns:
        One ``[SOURCE]/[CONTEXT]/[CONTENT]`` block per chunk, blank-line
        separated. ``[CONTEXT]`` is empty for chunks without a situating
        blurb (ingested with ``CONTEXTUALIZE`` off).
    """
    return "\n\n".join(
        _CHUNK_BLOCK.format(
            source=chunk.metadata.get("source", ""),
            context=chunk.context or "",
            content=chunk.content,
        )
        for chunk in chunks
    )


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    llm: LLMClient | None = None,
    formatted_context: str | None = None,
    verbose: int | None = None,
) -> str:
    """Generate a grounded, cited answer from retrieved chunks.

    The prompt (spec §10.2) instructs the model to answer **only** from the
    provided context, to say it doesn't know when the context lacks the
    answer, and to cite the ``[SOURCE]`` of facts it uses.

    Args:
        query: The user's question.
        chunks: The retrieved chunks to ground the answer in.
        llm: Chat client; resolved via the model registry when omitted.
        formatted_context: Pre-computed :func:`format_context` output, if the
            caller already built it; formatted from ``chunks`` when omitted.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The answer, post-processed with
        :func:`~varagity.models.llm.clean_response`.

    Raises:
        ValueError: If ``verbose`` is invalid.
        openai.APIError: If generation still fails after retries.
    """
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    client = llm if llm is not None else get_model("default")
    if formatted_context is None:
        formatted_context = format_context(chunks)
    prompt = ANSWER_PROMPT.format(formatted_context=formatted_context, query=query)
    response = client.generate([{"role": "user", "content": prompt}], verbose=verbose)
    return clean_response(response)


class StreamedAnswer(TypedDict):
    """Outcome of one streamed generation (spec_v2 §4.3).

    Attributes:
        answer: The final answer — ``clean_response`` over the accumulated
            raw text, so it is exact even when the streamed classification
            was best-effort (the orphaned-``</think>`` shape).
        reasoning: The captured ``<think>`` stream (``""`` when the model
            emitted none) — persisted per spec_v2 §9.1.
        aborted: ``True`` when ``should_abort`` stopped generation early
            (client disconnect); ``answer`` then holds the partial text.
        usage: Server-reported token counts (``prompt_tokens``,
            ``completion_tokens``), or ``None`` when the server sent none.
        tokens_per_second: Final decode throughput reported by llama.cpp,
            or ``None`` on any server that doesn't report ``timings``
            (see :class:`~varagity.models.llm.GenerationTimings`). Kept
            out of ``usage`` deliberately: that dict is token *counts*,
            and this is a rate.
    """

    answer: str
    reasoning: str
    aborted: bool
    usage: dict[str, int] | None
    tokens_per_second: float | None


def generate_answer_stream(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    on_delta: Callable[[Kind, str], None],
    llm: LLMClient | None = None,
    formatted_context: str | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_stats: Callable[[GenerationTimings], None] | None = None,
    verbose: int | None = None,
) -> StreamedAnswer:
    """Generate a grounded answer, streaming deltas to a callback.

    The streaming twin of :func:`generate_answer`: the same spec §10.2
    grounding prompt, but each text delta is classified by
    :class:`~varagity.models.stream.ThinkStreamSplitter` and handed to
    ``on_delta`` as it arrives — reasoning deltas for a collapsible
    "reasoning" surface, answer deltas for the visible answer.

    Args:
        query: The user's question.
        chunks: The retrieved chunks to ground the answer in.
        on_delta: Called with ``(kind, text)`` per classified fragment, in
            stream order (``kind`` is ``"reasoning"`` or ``"answer"``).
        llm: Chat client; resolved via the model registry when omitted.
        formatted_context: Pre-computed :func:`format_context` output, if the
            caller already built it; formatted from ``chunks`` when omitted.
        should_abort: Polled between deltas; returning ``True`` stops
            generation (the underlying HTTP stream closes, which frees the
            model server) and marks the result ``aborted``.
        on_stats: Called with llama.cpp's cumulative decode counters as
            they arrive — once per chunk, so a caller rendering them is
            expected to throttle. Silent on servers that report no
            ``timings``.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The completed :class:`StreamedAnswer`.

    Raises:
        ValueError: If ``verbose`` is invalid.
        openai.APIError: If establishing the stream fails after retries, or
            the stream breaks mid-generation.
    """
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    client = llm if llm is not None else get_model("default")
    if formatted_context is None:
        formatted_context = format_context(chunks)
    prompt = ANSWER_PROMPT.format(formatted_context=formatted_context, query=query)

    usage_holder: list[CompletionUsage] = []
    raw_parts: list[str] = []
    reasoning_parts: list[str] = []
    splitter = ThinkStreamSplitter()
    aborted = False
    # Only the newest reading is worth keeping: the counters are cumulative,
    # so the last one is both the live value and the final total.
    last_timings: GenerationTimings | None = None

    def _dispatch(fragments: list[tuple[Kind, str]]) -> None:
        for kind, text in fragments:
            if kind == "reasoning":
                reasoning_parts.append(text)
            on_delta(kind, text)

    def _record_timings(timings: GenerationTimings) -> None:
        nonlocal last_timings
        last_timings = timings
        if on_stats is not None:
            on_stats(timings)

    deltas = client.generate_stream(
        [{"role": "user", "content": prompt}],
        verbose=verbose,
        on_usage=usage_holder.append,
        on_timings=_record_timings,
    )
    try:
        for delta in deltas:
            if should_abort is not None and should_abort():
                aborted = True
                break
            raw_parts.append(delta)
            _dispatch(splitter.feed(delta))
    finally:
        deltas.close()  # closing aborts the server-side generation
    if not aborted:
        _dispatch(splitter.finalize())

    usage = usage_holder[-1] if usage_holder else None
    return StreamedAnswer(
        answer=clean_response("".join(raw_parts)),
        reasoning="".join(reasoning_parts).strip(),
        aborted=aborted,
        usage=(
            None
            if usage is None
            else {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        ),
        tokens_per_second=(None if last_timings is None else last_timings.tokens_per_second),
    )


def answer_query(
    query: str,
    *,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    k: int | None = None,
    verbose: int | None = None,
    on_retrieved: Callable[[list[RetrievedChunk]], None] | None = None,
) -> QueryState:
    """Run the full query pipeline: retrieve → format → generate (spec §10.1).

    Args:
        query: The user's question.
        retriever: Retrieval method; resolved from
            ``settings.RETRIEVAL_METHOD`` when omitted.
        llm: Chat client; resolved via the model registry when omitted.
        k: Chunks to retrieve; defaults to ``settings.TOP_K``.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.
        on_retrieved: Optional hook called with the retrieved chunks before
            generation — the CLI shows its matches table here (spec §10.1
            step 4: display matches, *then* generate) without this module
            owning any rendering.

    Returns:
        The completed :class:`QueryState`.

    Raises:
        ValueError: If ``verbose`` is invalid.
        KeyError: If ``settings.RETRIEVAL_METHOD`` names an unregistered
            retrieval method.
        openai.APIError: If embedding/generation still fails after retries.
        psycopg.OperationalError: If the vector store is unreachable.
    """
    settings = get_settings()
    verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
    active_retriever = (
        retriever if retriever is not None else get_retriever(settings.RETRIEVAL_METHOD)
    )
    chunks = active_retriever.retrieve(query, k=settings.TOP_K if k is None else k, verbose=verbose)
    if on_retrieved is not None:
        on_retrieved(chunks)
    formatted_context = format_context(chunks)
    answer = generate_answer(
        query, chunks, llm=llm, formatted_context=formatted_context, verbose=verbose
    )
    return QueryState(
        query=query,
        query_vector=None,  # encapsulated by the retrieval seam (see QueryState)
        retrieved=chunks,
        formatted_context=formatted_context,
        answer=answer,
    )
