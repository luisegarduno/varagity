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

from varagity.config import get_settings
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

# llama.cpp serves --ctx-size 16384 (docker-compose.yml). The document
# preamble budget leaves headroom for the prompt scaffolding, the chunk
# itself, and reasoning + blurb generation — and margin for tokenizer drift,
# since count_tokens is a cl100k approximation of the served model's own
# tokenizer (plan decision #8).
LLAMACPP_CTX_TOKENS = 16384
DOC_TOKEN_BUDGET = 12288


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

    A document longer than :data:`DOC_TOKEN_BUDGET` tokens (approximate) is
    truncated with a warning rather than crashing ingest — the blurb is then
    situated within the document's head, which still beats no context at all.

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
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    client = llm if llm is not None else get_model("default")

    n_doc_tokens = count_tokens(document_text)
    if n_doc_tokens > DOC_TOKEN_BUDGET:
        logger.warning(
            "document is ~%d tokens — over the ~%d-token contextualization budget "
            "(llama.cpp ctx %d minus headroom); truncating the document preamble",
            n_doc_tokens,
            DOC_TOKEN_BUDGET,
            LLAMACPP_CTX_TOKENS,
        )
        document_text = truncate_to_tokens(document_text, DOC_TOKEN_BUDGET)

    prompt = CONTEXTUAL_PROMPT.format(doc_content=document_text, chunk_content=chunk_text)
    response = client.generate([{"role": "user", "content": prompt}], verbose=verbose)
    blurb = clean_response(response)
    if not blurb:
        logger.warning(
            "contextualization returned an empty blurb (raw response length %d) — "
            "the chunk will be embedded without context",
            len(response),
        )
    v_situate_context(chunk_text, blurb, verbose)
    return blurb
