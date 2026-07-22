"""Web-document parser: ``.html`` / ``.htm`` / ``.xhtml`` (spec_v2 §8).

A thin registration over the shared Docling core: web pages carry digital
text, so a single no-OCR conversion through the same markdown/table/
provenance pipeline as the PDF parser suffices. Docling's HTML backend is
lightweight (``beautifulsoup4``) — no layout-model downloads.

Provenance: HTML has no pagination concept, so ``page`` stays ``None``
(spec_v2 §8.2) — the same graceful degradation as ``.txt`` / ``.md``.
"""

from varagity.ingest.parsers.base import register
from varagity.ingest.parsers.docling_base import DoclingParser


@register("web")
class WebParser(DoclingParser):
    """Parser for the ``web`` bucket (``.html`` / ``.htm`` / ``.xhtml``)."""
