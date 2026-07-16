"""``situate_context()`` — the heart of Contextual Retrieval (spec §9.4, §11.1).

For each chunk, the LLM produces a short blurb situating the chunk within its
whole document; the blurb is prepended to the chunk before embedding and BM25
indexing, which is what lifts retrieval out of the vanilla-RAG baseline
(Anthropic: ≈35% fewer retrieval failures from contextual embeddings alone).

Callers process a document's chunks **sequentially, grouped per document**
(the loader already iterates this way): every call for the same document
shares an identical prompt prefix, so llama.cpp reuses its KV/prompt cache —
a throughput concern with a local server, not a billing one (spec §9.4).
"""

import logging

from varagity.config import Settings, get_settings
from varagity.debug.show import check_verbose, v_situate_context
from varagity.models.llm import LLMClient, clean_response
from varagity.models.registry import get_model
from varagity.tokens import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)

# The Anthropic cookbook prompt, reproduced verbatim (spec §11.1).
CONTEXTUAL_PROMPT = """<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

# The prompt scaffolding around the document and chunk (tags + instruction,
# ~60 tokens) plus margin for the served tokenizer counting a few percent
# more than the cl100k approximation (plan decision #8).
_PROMPT_OVERHEAD_TOKENS = 640
# Doc-budget allowance for the chunk being situated. A fixed allowance (not
# the actual chunk's size) keeps the truncated document preamble IDENTICAL
# for every chunk of a document — the shared-prefix property llama.cpp's
# prompt cache depends on (module docstring). An outlier chunk larger than
# this is still transport-safe: LLMClient clamps the generation cap so the
# request always fits the window.
_CHUNK_ALLOWANCE_TOKENS = 2048


def doc_token_budget(settings: Settings) -> int:
    """Token budget for the document preamble of one situating prompt.

    What's left of the model's context window after reserving space for the
    blurb generation (``CONTEXTUALIZE_MAX_TOKENS`` — llama.cpp needs prompt
    *plus* generation to fit, or it hard-fails the request), the chunk
    allowance, and the prompt scaffolding.

    Args:
        settings: The application settings (``LLM_CONTEXT_TOKENS`` and
            ``CONTEXTUALIZE_MAX_TOKENS`` are read).

    Returns:
        The budget in (approximate) tokens; can be non-positive when the
        configured window is very small — callers degrade rather than crash.
    """
    return (
        settings.LLM_CONTEXT_TOKENS
        - settings.CONTEXTUALIZE_MAX_TOKENS
        - _CHUNK_ALLOWANCE_TOKENS
        - _PROMPT_OVERHEAD_TOKENS
    )


def situate_context(
    document_text: str,
    chunk_text: str,
    *,
    llm: LLMClient | None = None,
    verbose: int | None = None,
) -> str:
    """Generate the blurb situating one chunk within its parent document.

    The LLM sees the whole document and the chunk under the verbatim cookbook
    prompt (:data:`CONTEXTUAL_PROMPT`); the response is post-processed with
    :func:`~varagity.models.llm.clean_response` to strip reasoning-model
    ``<think>…</think>`` stages.

    A document longer than the :func:`doc_token_budget` (approximate tokens)
    is truncated with a warning rather than crashing ingest — the blurb is
    then situated within the document's head, which still beats no context at
    all. Generation runs under ``CONTEXTUALIZE_MAX_TOKENS`` (not the
    chat-sized ``MAX_TOKENS``): a blurb is short, and reserving a chat-sized
    generation against the context window would reject any document over a
    few thousand tokens. A reasoning preamble that overruns the cap cleans
    to an empty blurb (see Returns) — degraded context, never a failed file.

    Args:
        document_text: The parent document's full extracted text.
        chunk_text: The chunk to situate.
        llm: Chat client; resolved via the model registry when omitted.
            Callers contextualizing many chunks should pass one client.
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The situating blurb (may be empty if the model returned nothing
        usable — logged as a warning, never raised).

    Raises:
        ValueError: If ``verbose`` is invalid.
        openai.APIError: If the LLM request still fails after retries.
    """
    settings = get_settings()
    verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
    client = llm if llm is not None else get_model("default")

    budget = doc_token_budget(settings)
    if budget <= 0:
        logger.warning(
            "LLM_CONTEXT_TOKENS (%d) leaves no room for a document preamble after "
            "the blurb/chunk/scaffolding reserves — situating within the chunk alone",
            settings.LLM_CONTEXT_TOKENS,
        )
        document_text = ""
    else:
        n_doc_tokens = count_tokens(document_text)
        if n_doc_tokens > budget:
            logger.warning(
                "document is ~%d tokens — over the ~%d-token contextualization budget "
                "(LLM_CONTEXT_TOKENS %d minus the blurb, chunk, and scaffolding "
                "reserves); truncating the document preamble",
                n_doc_tokens,
                budget,
                settings.LLM_CONTEXT_TOKENS,
            )
            document_text = truncate_to_tokens(document_text, budget)

    prompt = CONTEXTUAL_PROMPT.format(doc_content=document_text, chunk_content=chunk_text)
    response = client.generate(
        [{"role": "user", "content": prompt}],
        max_tokens=settings.CONTEXTUALIZE_MAX_TOKENS,
        verbose=verbose,
    )
    blurb = clean_response(response)
    if not blurb:
        logger.warning(
            "contextualization returned an empty blurb (raw response length %d) — "
            "the chunk will be embedded without context",
            len(response),
        )
    v_situate_context(chunk_text, blurb, verbose)
    return blurb
