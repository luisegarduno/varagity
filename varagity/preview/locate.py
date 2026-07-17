"""Find the page and highlight rects of a chunk's text (ADR-010).

Locate-at-preview-time: no per-chunk page/bbox provenance exists at ingest
(ADR-005 §5 deferred it), so the preview path scores every page of the
source PDF by word-trigram containment and computes highlight rectangles
via pdfium text search — deterministic, format-agnostic, and working
retroactively for every already-ingested corpus.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from varagity.preview.normalize import normalize_chunk_text, snippets, words

# PDFium is not thread-safe (pypdfium2 METADATA, "Incompatibility with
# Threading"): FastAPI runs the sync preview routes in a threadpool, so
# every pdfium document/page/textpage/render call in this process
# serializes behind this one lock (render.py shares it).
PDFIUM_LOCK = threading.Lock()

# Highlight-rect cap: a pathological chunk (one giant table) must not ship
# thousands of overlay divs to the browser.
_MAX_RECTS = 300

# Chunks shorter than this many words score by unigram containment —
# too few words to form a meaningful trigram set.
_TRIGRAM_MIN_WORDS = 8


@dataclass(frozen=True)
class LocateResult:
    """Where a chunk's text lives in its source document.

    Attributes:
        page: Best-matching page, 1-based; ``None`` when the best coverage
            stayed below the caller's floor (the ``no_match`` outcome).
        page_count: Total pages in the document.
        rects: Highlight rectangles ``(x0, y0, x1, y1)``, normalized to
            ``[0, 1]`` with a **top-left** origin (y-flipped from PDF
            coordinates server-side, so the client does no coordinate math
            beyond percentages).
        coverage: The winning page's containment score in ``[0, 1]``.
    """

    page: int | None
    page_count: int
    rects: list[tuple[float, float, float, float]]
    coverage: float


def locate(pdf_path: Path, chunk_text: str, *, min_coverage: float) -> LocateResult:
    """Find the best-matching page for a chunk and its highlight rects.

    Pages are scored by word-trigram containment (unigram for very short
    chunks); ties keep the lowest page. On the winning page, the chunk's
    sentences become short search needles whose pdfium match rectangles —
    deduplicated and capped — form the highlight overlay.

    Args:
        pdf_path: A pdfium-openable PDF (the source itself, or a converted
            rendition).
        chunk_text: The chunk's markdown content, as both wire shapes
            deliver it to the client.
        min_coverage: Coverage floor below which the result reports no page
            (``PREVIEW_MIN_COVERAGE``).

    Returns:
        The located page, its highlight rects, and the coverage score.

    Raises:
        pypdfium2.PdfiumError: If the file cannot be opened as a PDF.
    """
    normalized = normalize_chunk_text(chunk_text)
    chunk_words = words(normalized)
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            page_texts = _page_texts(pdf)
            page_index, coverage = _best_page(chunk_words, page_texts)
            if page_index is None or coverage < min_coverage:
                return LocateResult(
                    page=None, page_count=len(page_texts), rects=[], coverage=coverage
                )
            rects = _highlight_rects(pdf, page_index, normalized)
            return LocateResult(
                page=page_index + 1, page_count=len(page_texts), rects=rects, coverage=coverage
            )
        finally:
            pdf.close()


def _page_texts(pdf: pdfium.PdfDocument) -> list[str]:
    """Extract every page's raw text (caller holds :data:`PDFIUM_LOCK`).

    Args:
        pdf: The open document.

    Returns:
        One text string per page, in page order.
    """
    texts: list[str] = []
    for index in range(len(pdf)):
        page = pdf[index]
        textpage = page.get_textpage()
        try:
            texts.append(textpage.get_text_bounded())
        finally:
            textpage.close()
            page.close()
    return texts


def _grams(tokens: list[str], use_trigrams: bool) -> set[tuple[str, ...]]:
    """Build the containment-scoring gram set for one token list.

    Args:
        tokens: Lowercase word tokens.
        use_trigrams: Word trigrams when ``True``, unigrams otherwise.

    Returns:
        The gram set (possibly empty).
    """
    if not use_trigrams:
        return {(token,) for token in tokens}
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def _best_page(chunk_words: list[str], page_texts: list[str]) -> tuple[int | None, float]:
    """Score every page by gram containment of the chunk.

    ``score = |chunk_grams ∩ page_grams| / |chunk_grams|`` — how much of
    the chunk the page contains (not the reverse: pages are usually bigger
    than chunks). Ties keep the lowest page (strict ``>``).

    Args:
        chunk_words: The chunk's word tokens.
        page_texts: Raw text per page.

    Returns:
        ``(best_page_index, best_coverage)``; index is ``None`` when no
        page scored above zero (or the chunk has no words at all).
    """
    use_trigrams = len(chunk_words) >= _TRIGRAM_MIN_WORDS
    chunk_grams = _grams(chunk_words, use_trigrams)
    if not chunk_grams:
        return None, 0.0
    best_index: int | None = None
    best_coverage = 0.0
    for index, text in enumerate(page_texts):
        page_grams = _grams(words(text), use_trigrams)
        coverage = len(chunk_grams & page_grams) / len(chunk_grams)
        if coverage > best_coverage:
            best_index, best_coverage = index, coverage
    return best_index, best_coverage


def _highlight_rects(
    pdf: pdfium.PdfDocument, page_index: int, normalized_text: str
) -> list[tuple[float, float, float, float]]:
    """Compute normalized highlight rects on one page (caller holds the lock).

    Each snippet needle is searched once (first match); pdfium merges each
    match into per-line rectangles (``count_rects``/``get_rect``), which are
    y-flipped to a top-left origin, normalized by the page size, deduplicated
    on rounded coordinates (the stride overlap re-finds line fragments), and
    capped at :data:`_MAX_RECTS`.

    Args:
        pdf: The open document.
        page_index: 0-based page to search.
        normalized_text: The chunk's de-decorated text.

    Returns:
        The ``(x0, y0, x1, y1)`` rects, ``[0, 1]``-normalized, top-left
        origin, in discovery order.
    """
    page = pdf[page_index]
    textpage = page.get_textpage()
    try:
        width, height = page.get_size()
        if width <= 0 or height <= 0:
            return []
        rects: list[tuple[float, float, float, float]] = []
        seen: set[tuple[float, ...]] = set()
        for needle in snippets(normalized_text):
            match = textpage.search(needle, match_case=False).get_next()
            if match is None:
                continue
            char_index, char_count = match
            for rect_index in range(textpage.count_rects(char_index, char_count)):
                left, bottom, right, top = textpage.get_rect(rect_index)
                if right <= left or top <= bottom:  # degenerate: nothing to tint
                    continue
                rect = (
                    left / width,
                    (height - top) / height,
                    right / width,
                    (height - bottom) / height,
                )
                key = tuple(round(value, 4) for value in rect)
                if key in seen:
                    continue
                seen.add(key)
                rects.append(rect)
                if len(rects) >= _MAX_RECTS:
                    return rects
        return rects
    finally:
        textpage.close()
        page.close()
