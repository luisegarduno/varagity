"""Office-document parser: the OOXML families, CSV, and OpenDocument (spec_v2 §8).

A thin registration over the shared Docling core: office formats carry
digital text, so a single no-OCR conversion through the same markdown/
table/provenance pipeline as the PDF parser suffices (the two-pass OCR
fallback is PDF-only). The bucket covers the ``.docx``/``.pptx``/``.xlsx``
families (including macro-enabled and template variants — Docling's
backends open them like their base formats), single-table ``.csv``, and
OpenDocument ``.odt``/``.ods``/``.odp``. All backends are lightweight
(``python-docx`` / ``python-pptx`` / ``openpyxl`` / ``odfdo``) — no
layout-model downloads.

Provenance: ``.pptx`` maps each slide to a Docling page and
``.xlsx``/``.ods`` map each sheet to one, so ``page`` carries the
slide/sheet identity; ``.docx``, ``.csv``, ``.odt``, and ``.odp`` expose no
per-item pagination, so ``page`` stays ``None`` (spec_v2 §8.2).
"""

from varagity.ingest.parsers.base import register
from varagity.ingest.parsers.docling_base import DoclingParser


@register("office")
class OfficeParser(DoclingParser):
    """Parser for the ``office`` bucket (OOXML families / CSV / OpenDocument)."""
