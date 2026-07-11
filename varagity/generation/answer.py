"""Context prompt & grounded answer generation (spec §10.2).

Each retrieved chunk is formatted with its provenance
(``[SOURCE]/[CONTEXT]/[CONTENT]``), joined into ``formatted_context``, and
fed to the LLM under the grounding prompt: answer from the context only,
admit ignorance otherwise, cite sources. :func:`answer_query` threads the
whole query pipeline through the spec §10.1 state dict.
"""

from collections.abc import Callable
from typing import TypedDict

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.models.llm import LLMClient, clean_response
from varagity.models.registry import get_model
from varagity.retrieval.base import Retriever, get_retriever
from varagity.stores.records import RetrievedChunk

# Per-chunk provenance block (spec §10.2, column-aligned as specified).
# [CONTEXT] renders empty until Phase 5 populates the situating blurb — the
# format is stable across that change.
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
        query_vector: The query embedding. ``None`` in Phase 4 — the
            retrieval seam encapsulates query encoding (bm25 has no vector at
            all); Phase 8's ``query_flow`` restructures embedding into its
            own tracked stage.
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
        blurb (every chunk until Phase 5).
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
