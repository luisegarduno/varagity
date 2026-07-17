"""Evidence-panel document page preview (ADR-010).

On-demand, server-side locate + render: given a chunk's text and its
parent document, find the one page/slide containing it (word-trigram
coverage over pdfium page text), compute highlight rectangles (pdfium
text search over de-decorated snippets), and render that page to a PNG.
No ingest-time provenance, no schema changes, no reingest — everything
keys off the content-addressed ``doc_id``, so results are immutable and
cacheable forever.
"""

from varagity.preview.convert import (
    ConversionFailed,
    ConversionUnavailable,
    conversion_cache_path,
    ensure_pdf,
)
from varagity.preview.locate import PDFIUM_LOCK, LocateResult, locate
from varagity.preview.normalize import normalize_chunk_text, snippets, words
from varagity.preview.render import render_page_png
from varagity.preview.source import PreviewSource, resolve_preview_source

__all__ = [
    "PDFIUM_LOCK",
    "ConversionFailed",
    "ConversionUnavailable",
    "LocateResult",
    "PreviewSource",
    "conversion_cache_path",
    "ensure_pdf",
    "locate",
    "normalize_chunk_text",
    "render_page_png",
    "resolve_preview_source",
    "snippets",
    "words",
]
