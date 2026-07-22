"""Shared Docling conversion core for every rich-document parser (spec_v2 §8.2).

The generic machinery ``parsers/pdf.py`` proved in v1 — ``DocumentConverter``
conversion → structure-aware markdown export → hyphen repair → page/character
provenance → :class:`~varagity.ingest.parsers.base.RawDocument` assembly —
lifted out so the office (``.docx``/``.pptx``/``.xlsx``) and web
(``.html``/``.htm``) parsers reuse one markdown/table/provenance pipeline.

**No OCR lives here.** The two-pass OCR fallback is PDF-specific (scanned
pages) and stays in ``pdf.py``; the formats parsed through
:class:`DoclingParser` carry digital text, so a single no-OCR conversion
suffices and every document is ``extraction="text"``.

Page semantics per format (verified against Docling's backends): ``.pptx``
maps each slide to a page and ``.xlsx`` maps each sheet to a page, so
``page`` carries the slide/sheet identity; ``.docx`` and ``.html`` expose no
pagination (``document.pages`` is empty), so ``page`` stays ``None`` — the
same graceful degradation the text parser uses.

Docling imports are deferred to call time: importing this module (which
happens on every CLI start via parser self-registration) must not pay for
Docling's model machinery.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.ingest.parsers.base import RawDocument
from varagity.ingest.parsers.text import remove_hyphen_space

if TYPE_CHECKING:  # heavy imports, type-only (runtime imports are lazy)
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import DoclingDocument


def page_char_counts(document: "DoclingDocument") -> dict[int, int]:
    """Count extracted non-whitespace characters per page.

    Uses Docling item provenance: every content item (including table
    cells) is attributed to the page(s) it appears on. Pages that no item
    references keep a zero count — the textless-page signal the PDF OCR
    trigger reads. Formats without pagination (``.docx``/``.html``) yield an
    empty mapping.

    Args:
        document: The converted Docling document.

    Returns:
        ``{page_no: char_count}`` covering every page of the document.
    """
    counts: dict[int, int] = dict.fromkeys(document.pages, 0)
    for item, _level in document.iterate_items():
        chars = 0
        text = getattr(item, "text", None)
        if text:
            chars += len(text.strip())
        data = getattr(item, "data", None)  # tables carry text in cells
        table_cells = getattr(data, "table_cells", None) if data is not None else None
        if table_cells:
            chars += sum(len(cell.text.strip()) for cell in table_cells if cell.text)
        if not chars:
            continue
        for prov in getattr(item, "prov", None) or []:
            counts[prov.page_no] = counts.get(prov.page_no, 0) + chars
    return counts


def export_markdown(document: "DoclingDocument") -> str:
    r"""Export a converted document as normalized markdown.

    Docling's structure-aware export (headings, GFM tables) with newlines
    normalized to ``\n`` and hyphen-broken words repaired — the exact text
    shape the chunkers expect from every parser.

    Args:
        document: The converted Docling document.

    Returns:
        The normalized markdown text.
    """
    text = document.export_to_markdown().replace("\r\n", "\n").replace("\r", "\n")
    return remove_hyphen_space(text)


def raw_document(
    path: Path, text: str, page_counts: dict[int, int], *, extraction: str
) -> RawDocument:
    """Assemble the parser result with provenance metadata.

    Args:
        path: The source file.
        text: The normalized extracted text.
        page_counts: Per-page character counts (for page attribution;
            empty for non-paginated formats).
        extraction: ``"text"``, ``"ocr"``, or ``"ocr_fallback"``.

    Returns:
        The :class:`~varagity.ingest.parsers.base.RawDocument` handed to the
        chunker. ``source_meta`` carries ``page`` — the first page (slide
        for ``.pptx``, sheet for ``.xlsx``) that contributed text, ``None``
        when the format has no pagination — and ``extraction``.
    """
    pages_with_text = [page for page, count in sorted(page_counts.items()) if count > 0]
    source_meta: dict[str, Any] = {
        "source": str(path.resolve()),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "page": pages_with_text[0] if pages_with_text else None,
        "extraction": extraction,
    }
    return RawDocument(text=text, source_meta=source_meta)


class DoclingParser:
    """Single-pass Docling parser for digital-text formats (no OCR).

    The shared base the ``office`` and ``web`` registrations subclass: one
    plain ``DocumentConverter`` conversion through the same markdown/table/
    provenance pipeline as the PDF parser, minus the two-pass OCR fallback
    (these formats carry digital text — spec_v2 §8.2). The converter is
    built lazily and cached on the instance (the registry instantiates one
    parser per process), and the office/web backends are lightweight — no
    layout-model downloads.
    """

    def __init__(self) -> None:
        """Initialize the converter cache (nothing heavy happens here)."""
        self._converter: DocumentConverter | None = None

    def extract(self, path: Path, verbose: int | None = None) -> RawDocument:
        """Extract a document's text and provenance via one Docling pass.

        Args:
            path: The file to convert (any format Docling handles without
                OCR; discovery routes only this parser's bucket here).
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The extracted document. ``source_meta`` carries ``page`` (slide
            for ``.pptx``, sheet for ``.xlsx``, ``None`` for formats
            without pagination) and ``extraction="text"`` (never OCR).

        Raises:
            ValueError: If ``verbose`` is invalid.
            docling.exceptions.ConversionError: If conversion fails (a
                malformed file) — the loader counts the file as failed and
                the run continues.
        """
        settings = get_settings()
        check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        document = self._convert(path)
        text = export_markdown(document)
        return raw_document(path, text, page_char_counts(document), extraction="text")

    def _convert(self, path: Path) -> "DoclingDocument":
        """Run the (cached) Docling converter on one file.

        Args:
            path: The file to convert.

        Returns:
            The converted Docling document.
        """
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter.convert(path).document  # raises ConversionError on failure
