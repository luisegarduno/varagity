"""Approximate token counting.

Uses tiktoken's ``cl100k_base`` encoding as a **documented approximation**
(plan decision #8): the e5 embedding model actually tokenizes with an
XLM-RoBERTa SentencePiece vocabulary, but a portable approximation is enough
for the two consumers here — the ``n_tokens`` provenance field on
:class:`~varagity.stores.records.ChunkRecord` and the near-512-token ingest
warning in :class:`~varagity.models.embeddings.EmbeddingsClient`.

tiktoken downloads its BPE ranks file on first use (cached afterwards). If
that download fails (offline host), counting falls back to a chars/4 estimate
with a one-time warning instead of failing ingestion.
"""

import logging
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)

# Rule-of-thumb English average used only when the tiktoken encoding cannot
# be loaded (e.g. offline first run).
_FALLBACK_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding | None:
    """Load the cl100k_base encoding once, tolerating offline failure.

    Returns:
        The tiktoken encoding, or ``None`` when it cannot be loaded (the
        failure is logged once; callers fall back to a character estimate).
    """
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 — any load failure (network, cache perms) degrades the same way
        logger.warning(
            "tiktoken cl100k_base unavailable (offline?); token counts fall back to "
            "a chars/%d approximation",
            _FALLBACK_CHARS_PER_TOKEN,
            exc_info=True,
        )
        return None


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` (approximate; see module docstring).

    Args:
        text: The text to count.

    Returns:
        The cl100k_base token count, or a chars/4 estimate if the encoding
        could not be loaded.
    """
    encoding = _encoding()
    if encoding is None:
        return len(text) // _FALLBACK_CHARS_PER_TOKEN
    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens (approximate).

    Used by the contextualizer to keep an oversized document preamble inside
    the llama.cpp context budget instead of crashing ingest.

    Args:
        text: The text to truncate.
        max_tokens: Maximum tokens to keep (must be non-negative).

    Returns:
        ``text`` unchanged when it fits, otherwise its first ``max_tokens``
        tokens decoded back to a string (chars/4 estimate if the encoding
        could not be loaded).

    Raises:
        ValueError: If ``max_tokens`` is negative.
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be non-negative; got {max_tokens}")
    encoding = _encoding()
    if encoding is None:
        return text[: max_tokens * _FALLBACK_CHARS_PER_TOKEN]
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
