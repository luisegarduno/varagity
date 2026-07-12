"""Office-document parser: ``.docx`` / ``.pptx`` / ``.xlsx`` (spec_v2 §8).

A thin registration over the shared Docling core: office formats carry
digital text, so a single no-OCR conversion through the same markdown/
table/provenance pipeline as the PDF parser suffices (the two-pass OCR
fallback is PDF-only). Docling's office backends are lightweight
(``python-docx`` / ``python-pptx`` / ``openpyxl``) — no layout-model
downloads.

Provenance: ``.pptx`` maps each slide to a Docling page and ``.xlsx`` maps
each sheet to one, so ``page`` carries the slide/sheet identity; ``.docx``
exposes no reliable pagination, so ``page`` stays ``None`` (spec_v2 §8.2).
"""

from varagity.ingest.parsers.base import register
from varagity.ingest.parsers.docling_base import DoclingParser


@register("office")
class OfficeParser(DoclingParser):
    """Parser for the ``office`` bucket (``.docx`` / ``.pptx`` / ``.xlsx``)."""
