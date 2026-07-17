"""Markdown de-decoration for preview text matching (ADR-010).

Chunk text is docling markdown — headings, emphasis, GFM table pipes,
``<!-- image -->``-style placeholder comments, list bullets, dot leaders —
while a PDF's text layer carries none of that decoration. These pure
functions reduce a chunk to the words that actually appear on the page, so
page scoring (:func:`words`) and pdfium snippet search (:func:`snippets`)
can match against the extracted page text.
"""

import re

# docling drops placeholder comments (<!-- image -->,
# <!-- formula-not-decoded -->) into its markdown export.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# docling escapes markdown specials in literal text (``doc\_id``); unwind
# them *before* emphasis stripping so an identifier's inner underscore is
# never mistaken for an (un)wrapping marker.
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()<>#+.!|~-])")

# A GFM separator row (|---|:---:|) — or any line of only pipes, dashes,
# colons, and whitespace — carries no page text at all.
_TABLE_SEPARATOR_ROW = re.compile(r"^[\s:|-]+$", re.MULTILINE)

# Line-initial heading markers and list bullets (unordered + ordered).
_HEADING_MARKER = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_LIST_BULLET = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+", re.MULTILINE)

# Emphasis markers: every asterisk run, and underscore runs only where they
# open or close a word — ``doc_id``'s inner underscore is real page text.
_EMPHASIS = re.compile(r"\*{1,3}|(?<!\w)_{1,3}(?=\w)|(?<=\w)_{1,3}(?!\w)")

# Dot leaders ("Intro ..... 4"): docling may render a different dot count
# than the page shows, so they are dropped from needles rather than trusted.
_DOT_LEADER = re.compile(r"\.{3,}")

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalize_chunk_text(text: str) -> str:
    """Strip markdown decoration, keeping the words as the page shows them.

    Inter-word punctuation (``"Firefly,"``, ``"12,400"``) survives — pdfium
    search needs the needle's characters to match the page text exactly,
    whitespace aside — while pure decoration (markers, pipes, placeholder
    comments) is removed and whitespace collapses to single spaces.

    Args:
        text: The chunk's markdown content.

    Returns:
        The de-decorated text, whitespace-collapsed and stripped.
    """
    text = _HTML_COMMENT.sub(" ", text)
    text = _MARKDOWN_ESCAPE.sub(r"\1", text)
    text = _TABLE_SEPARATOR_ROW.sub(" ", text)
    text = _HEADING_MARKER.sub("", text)
    text = _LIST_BULLET.sub("", text)
    text = text.replace("|", " ")
    text = _EMPHASIS.sub("", text)
    text = _DOT_LEADER.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    r"""Tokenize text into lowercase word tokens for page scoring.

    Punctuation-insensitive by construction (``\w+`` runs), so chunk and
    page text tokenize identically regardless of markdown-vs-layout
    punctuation differences.

    Args:
        text: Any text (normalized chunk text or raw pdfium page text).

    Returns:
        The lowercase word tokens, in order.
    """
    return [match.group(0).lower() for match in _WORD.finditer(text)]


def snippets(text: str, size: int = 8, stride: int = 4) -> list[str]:
    """Break normalized text into short pdfium-searchable needles.

    Sentences of at most ``size`` whitespace tokens search whole; longer
    ones become overlapping ``size``-token windows every ``stride`` tokens
    (the overlap closes the highlight gaps non-overlapping windows leave),
    plus a tail window so the sentence's end is always covered. Tokens keep
    their punctuation — the needle must match the page's characters.

    Args:
        text: Normalized chunk text (:func:`normalize_chunk_text` output).
        size: Window length in whitespace tokens.
        stride: Window step in whitespace tokens.

    Returns:
        The needles, deduplicated, in reading order.
    """
    needles: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        tokens = sentence.split()
        if not tokens:
            continue
        if len(tokens) <= size:
            needles.append(" ".join(tokens))
            continue
        for start in range(0, len(tokens) - size + 1, stride):
            needles.append(" ".join(tokens[start : start + size]))
        if (len(tokens) - size) % stride:
            needles.append(" ".join(tokens[-size:]))
    return list(dict.fromkeys(needles))
