"""``situate_context()`` — the heart of Contextual Retrieval (spec §9.4, §11.1).

For each chunk, the LLM produces a short blurb situating the chunk within its
whole document; the blurb is prepended to the chunk before embedding and BM25
indexing, which is what lifts retrieval out of the vanilla-RAG baseline
(Anthropic: ≈35% fewer retrieval failures from contextual embeddings alone).

Callers process a document's chunks **sequentially, grouped per document**
(the loader already iterates this way), fitting the document into its prompt
budget **once** with :func:`fit_document_preamble` and handing that preamble
to every :func:`situate_context` call for the document: every call then
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

# Everything in the prompt besides the document and the chunk. Measured on
# the served model: the cookbook tags and instruction are 68 tokens, and the
# chat template adds 91 of its own (an injected default system prompt plus
# role markers). The rest is slack for tokenization at the preamble's edges
# and for a chunk running past its allowance. It was sized when it also had
# to absorb the cl100k approximation's drift (plan decision #8); the
# preamble is now counted with the served tokenizer instead
# (fit_document_preamble), so that share is spare.
_PROMPT_OVERHEAD_TOKENS = 640
# Doc-budget allowance for the chunk being situated. A fixed allowance (not
# the actual chunk's size) keeps the fitted document preamble IDENTICAL
# for every chunk of a document — the shared-prefix property llama.cpp's
# prompt cache depends on (module docstring). A chunk larger than this eats
# into the overhead's slack, then into the generation reserve: a generation
# that reaches the window stops there (finish_reason=length), which degrades
# to an empty blurb rather than a failed file.
_CHUNK_ALLOWANCE_TOKENS = 2048
# When the served tokenizer finds a cut over budget, the next cl100k cut is
# scaled by budget / served count, then pulled 2% under so a document whose
# head is denser than its average still lands inside in one re-cut.
_RECUT_AIM = 0.98
# Re-cuts allowed before the byte-safe fallback. Drift runs close to uniform
# across a document (one re-cut lands it, measured), so this bound only
# catches a pathological one.
_MAX_RECUTS = 3


def doc_token_budget(settings: Settings) -> int:
    """Token budget for the document preamble of one situating prompt.

    What's left of the model's context window after reserving space for the
    blurb generation (``CONTEXTUALIZE_MAX_TOKENS`` — llama.cpp needs prompt
    *plus* generation to fit, or the blurb is cut off), the chunk allowance,
    and the prompt scaffolding.

    Args:
        settings: The application settings (``LLM_CONTEXT_TOKENS`` and
            ``CONTEXTUALIZE_MAX_TOKENS`` are read).

    Returns:
        The budget in the served model's tokens (what
        :func:`fit_document_preamble` holds a preamble to); can be
        non-positive when the configured window is very small — callers
        degrade rather than crash.
    """
    return (
        settings.LLM_CONTEXT_TOKENS
        - settings.CONTEXTUALIZE_MAX_TOKENS
        - _CHUNK_ALLOWANCE_TOKENS
        - _PROMPT_OVERHEAD_TOKENS
    )


def _utf8_head(text: str, max_bytes: int) -> str:
    """Cut ``text`` to its first ``max_bytes`` UTF-8 bytes, on a character boundary.

    Args:
        text: The text to cut.
        max_bytes: The byte ceiling.

    Returns:
        The longest prefix of ``text`` that encodes to at most ``max_bytes``
        bytes (a multi-byte character split by the cut is dropped whole).
    """
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def fit_document_preamble(document_text: str, *, llm: LLMClient | None = None) -> str:
    """Fit one document into the situating prompt's budget — once per document.

    The :func:`doc_token_budget` is held in the **served model's** tokens.
    The cl100k approximation makes the cheap first cut, and the served
    tokenizer has the last word: the served model gives every digit its own
    token where cl100k packs up to three, so a digit-dense document
    (timestamps, coordinates, tables) runs far over its cl100k estimate —
    1.45× on a location-history export, enough for a cl100k-sized preamble
    to overflow the window by itself.

    1. A document of at most ``budget`` UTF-8 bytes is returned whole,
       without a count: every token spans at least one byte, so it fits
       under any tokenizer.
    2. A document over the budget by cl100k is cut to it, with a warning.
       The cut also bounds what step 3 sends to the server.
    3. The (possibly whole) text is counted with
       :meth:`~varagity.models.llm.LLMClient.count_served_tokens` and,
       while still over, re-cut in proportion (up to :data:`_MAX_RECUTS`
       times), each re-cut warned.
    4. If the server cannot count, or the re-cuts never land, the preamble
       is the document's first ``budget`` UTF-8 bytes — step 1's
       any-tokenizer bound, so the fit holds regardless. That costs a
       shorter preamble, never a failed file.

    Pass the result to every :func:`situate_context` call for the
    document's chunks: the preamble is then byte-identical across them
    (llama.cpp's prompt cache), and the served count is paid per document,
    not per chunk.

    Args:
        document_text: The parent document's full extracted text.
        llm: Chat client whose server does the counting; resolved via the
            model registry when omitted, and only once a count is needed.

    Returns:
        The document preamble: the whole document when it fits, otherwise
        its head; ``""`` when the configured window leaves no room for one.
    """
    settings = get_settings()
    budget = doc_token_budget(settings)
    if budget <= 0:
        logger.warning(
            "LLM_CONTEXT_TOKENS (%d) leaves no room for a document preamble after "
            "the blurb/chunk/scaffolding reserves — situating within the chunk alone",
            settings.LLM_CONTEXT_TOKENS,
        )
        return ""
    if len(document_text.encode("utf-8")) <= budget:
        return document_text

    preamble = document_text
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
        preamble = truncate_to_tokens(document_text, budget)

    client = llm if llm is not None else get_model("default")
    n_served = client.count_served_tokens(preamble)
    recuts = 0
    while n_served is not None and n_served > budget and recuts < _MAX_RECUTS:
        n_approx = count_tokens(preamble)
        logger.warning(
            "the served model counts the document preamble at %d tokens (~%d by the "
            "cl100k approximation) — over the %d-token budget; re-cutting it",
            n_served,
            n_approx,
            budget,
        )
        preamble = truncate_to_tokens(document_text, int(n_approx * budget / n_served * _RECUT_AIM))
        n_served = client.count_served_tokens(preamble)
        recuts += 1
    if n_served is None:
        logger.warning(
            "no served token count — cutting the document preamble to its first %d bytes, "
            "which fits under any tokenizer",
            budget,
        )
    elif n_served > budget:
        logger.warning(
            "the document preamble is still %d served tokens after %d re-cuts — cutting it "
            "to its first %d bytes, which fits under any tokenizer",
            n_served,
            recuts,
            budget,
        )
    else:
        return preamble
    return _utf8_head(document_text, budget)


def situate_context(
    preamble: str,
    chunk_text: str,
    *,
    llm: LLMClient | None = None,
    verbose: int | None = None,
) -> str:
    """Generate the blurb situating one chunk within its parent document.

    The LLM sees the document preamble and the chunk under the verbatim
    cookbook prompt (:data:`CONTEXTUAL_PROMPT`); the response is
    post-processed with :func:`~varagity.models.llm.clean_response` to strip
    reasoning-model ``<think>…</think>`` stages.

    The preamble is sent as given, so it must already fit: make it with
    :func:`fit_document_preamble`, once per document. A document over
    the budget is then situated within its head, which still beats no
    context at all. Generation runs under ``CONTEXTUALIZE_MAX_TOKENS`` (not
    the chat-sized ``MAX_TOKENS``): a blurb is short, and reserving a
    chat-sized generation against the context window would reject any
    document over a few thousand tokens. A reasoning preamble that overruns
    the cap cleans to an empty blurb (see Returns) — degraded context,
    never a failed file.

    Args:
        preamble: The parent document's text as fitted by
            :func:`fit_document_preamble` — identical for every chunk of
            the document.
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

    prompt = CONTEXTUAL_PROMPT.format(doc_content=preamble, chunk_content=chunk_text)
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
