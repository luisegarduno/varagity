"""PDF parser: Docling two-pass extraction with an automatic OCR fallback.

Pass 1 converts with OCR disabled — the fast path, since OCR is a
multi-× slowdown. If the result has (near-)no text, too many textless
pages, or pass 1 raised, pass 2 re-converts with OCR enabled (plan
decision #10). Both passes share the ``docling_base`` markdown/table/
provenance core (spec_v2 §8.2), so the fallback changes *how* text is
recovered, never its downstream shape — and the office/web parsers
share the exact same pipeline.

The OCR engine is pluggable via the ``OCR_ENGINE`` setting: a small
factory maps engine names to Docling ``ocr_options`` (EasyOCR is the
benchmark-decided default — ADR-004; Tesseract needs only the system
binary). Documents recovered by pass 2 carry
``extraction="ocr_fallback"`` provenance on every chunk.

Docling imports are deferred to call time: importing this module (which
happens on every CLI start via parser self-registration) must not pay
for Docling's model machinery.

Pattern adapted from the docling-ocr-pipeline reference notebook, kept
Docling-native (no side pdf2image/pytesseract path) so both passes share
one markdown/table/provenance pipeline and the engine swap is pure config.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.ingest.parsers.base import RawDocument, register
from varagity.ingest.parsers.docling_base import (
    export_markdown,
    page_char_counts,
    raw_document,
)

if TYPE_CHECKING:  # heavy imports, type-only (runtime imports are lazy)
    from docling.datamodel.pipeline_options import OcrOptions
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# EasyOCR takes ISO 639-1 codes verbatim; Tesseract wants ISO 639-2
# (`en` → `eng`). Unknown codes pass through so native Tesseract codes
# (and scripts like `chi_sim`) keep working.
_TESSERACT_LANG = {
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "nl": "nld",
    "pt": "por",
}


def _easyocr_options(languages: list[str], force_full_page: bool) -> "OcrOptions":
    """Build EasyOCR options (the default engine — ADR-004).

    Model storage is pinned inside Docling's cache directory (instead of
    EasyOCR's own ``~/.EasyOCR``) so one cache volume covers every model
    the parser downloads.

    Args:
        languages: ISO 639-1 language codes, primary first.
        force_full_page: OCR whole pages even where a text layer exists.

    Returns:
        The populated ``EasyOcrOptions``.
    """
    from docling.datamodel.pipeline_options import EasyOcrOptions
    from docling.datamodel.settings import settings as docling_settings

    return EasyOcrOptions(
        lang=languages,
        force_full_page_ocr=force_full_page,
        model_storage_directory=str(docling_settings.cache_dir / "models" / "EasyOcr"),
    )


def _tesseract_options(languages: list[str], force_full_page: bool) -> "OcrOptions":
    """Build Tesseract CLI options (needs the ``tesseract`` system binary).

    Args:
        languages: ISO 639-1 language codes, primary first (mapped to
            Tesseract's ISO 639-2 codes; unknown codes pass through).
        force_full_page: OCR whole pages even where a text layer exists.

    Returns:
        The populated ``TesseractCliOcrOptions``.
    """
    from docling.datamodel.pipeline_options import TesseractCliOcrOptions

    return TesseractCliOcrOptions(
        lang=[_TESSERACT_LANG.get(code, code) for code in languages],
        force_full_page_ocr=force_full_page,
    )


OCR_ENGINE_FACTORIES: dict[str, Callable[[list[str], bool], "OcrOptions"]] = {
    "easyocr": _easyocr_options,
    "tesseract": _tesseract_options,
}


def get_ocr_options(
    engine: str, languages: list[str], force_full_page: bool = False
) -> "OcrOptions":
    """Resolve an ``OCR_ENGINE`` name to Docling OCR options.

    Args:
        engine: Registry key (``"easyocr"`` or ``"tesseract"``; adding an
            engine means adding a factory above — no caller edits).
        languages: ISO 639-1 language codes, primary first.
        force_full_page: OCR whole pages even where a text layer exists.

    Returns:
        Engine-specific Docling ``ocr_options``.

    Raises:
        KeyError: If no factory is registered under ``engine`` (message
            lists the available ones).
    """
    if engine not in OCR_ENGINE_FACTORIES:
        raise KeyError(f"Unknown OCR engine {engine!r}. Available: {list(OCR_ENGINE_FACTORIES)}")
    return OCR_ENGINE_FACTORIES[engine](languages, force_full_page)


@dataclass(frozen=True)
class ExtractionStats:
    """What pass 1 recovered, as inputs to the fallback trigger.

    Attributes:
        non_ws_chars: Non-whitespace characters in the exported text.
        total_pages: Page count of the source document.
        textless_pages: Pages that contributed no text to the export.
    """

    non_ws_chars: int
    total_pages: int
    textless_pages: int


def extraction_stats(text: str, page_char_counts: dict[int, int]) -> ExtractionStats:
    """Summarize an extraction pass for the fallback trigger.

    Args:
        text: The exported document text.
        page_char_counts: Non-whitespace character count per page number.

    Returns:
        The pass's :class:`ExtractionStats`.
    """
    return ExtractionStats(
        non_ws_chars=sum(1 for char in text if not char.isspace()),
        total_pages=len(page_char_counts),
        textless_pages=sum(1 for count in page_char_counts.values() if count == 0),
    )


def needs_ocr_fallback(
    stats: ExtractionStats, *, min_chars: int, textless_page_ratio: float
) -> bool:
    """Decide whether pass 1's result warrants the OCR pass (plan decision #10).

    Evaluated once per document, as in the reference notebook. Either
    trigger suffices: a (near-)empty extraction, or a high share of
    textless pages — the latter catches mixed scanned/digital documents
    whose digital pages alone pass a document-level length check.

    Args:
        stats: Pass 1's extraction summary.
        min_chars: Below this many non-whitespace characters the document
            counts as having no text layer (``PDF_OCR_MIN_CHARS``).
        textless_page_ratio: Textless-page share at or above which the
            fallback fires (``PDF_OCR_TEXTLESS_PAGE_RATIO``).

    Returns:
        True if the document should be re-converted with OCR.
    """
    if stats.non_ws_chars < min_chars:
        return True
    return stats.total_pages > 0 and stats.textless_pages / stats.total_pages >= textless_page_ratio


@register("pdf")
class PdfParser:
    """Parser for the ``pdf`` bucket: Docling with the two-pass OCR fallback.

    Converters are built lazily and cached on the instance (the registry
    instantiates one parser per process), so Docling's models load once
    per pass type, not once per file.
    """

    def __init__(self) -> None:
        """Initialize the converter caches (nothing heavy happens here)."""
        self._fast_converter: DocumentConverter | None = None
        # Keyed by (engine, languages, force_full_page): tests and
        # long-lived processes may change OCR settings between calls.
        self._ocr_converters: dict[tuple[str, tuple[str, ...], bool], DocumentConverter] = {}

    def extract(self, path: Path, verbose: int | None = None) -> RawDocument:
        """Extract a PDF's text, falling back to OCR when warranted.

        Args:
            path: The ``.pdf`` file to convert.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The extracted document. ``source_meta`` carries ``page`` (the
            first page that contributed text — chunk-level attribution
            would need offset support in the shared chunker, a deferred
            enhancement) and ``extraction`` (``"ocr_fallback"`` when pass
            2 produced the text, else ``"text"``).

        Raises:
            ValueError: If ``verbose`` is invalid.
            docling.exceptions.ConversionError: If conversion fails — for
                pass 1 only when the fallback is disabled; a pass 2
                failure always propagates (the loader counts the file as
                failed and re-attempts it on the next run, rather than
                recording a permanently half-extracted document).
        """
        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)

        if settings.PDF_OCR_FORCE_FULL_PAGE:
            # Escape hatch for corrupt-text-layer PDFs: garbage embedded
            # text passes the content triggers, so skip straight to a
            # forced full-page OCR conversion.
            logger.info("%s: PDF_OCR_FORCE_FULL_PAGE is on — full-page OCR conversion", path.name)
            text, page_counts = self._convert(path, ocr=True)
            return self._raw_document(path, text, page_counts, extraction="ocr_fallback")

        try:
            text, page_counts = self._convert(path, ocr=False)
        except Exception:
            if not settings.PDF_OCR_FALLBACK:
                raise
            logger.warning(
                "%s: text-layer conversion raised — retrying with OCR", path.name, exc_info=True
            )
            text, page_counts = self._convert(path, ocr=True)
            return self._raw_document(path, text, page_counts, extraction="ocr_fallback")

        stats = extraction_stats(text, page_counts)
        if settings.PDF_OCR_FALLBACK and needs_ocr_fallback(
            stats,
            min_chars=settings.PDF_OCR_MIN_CHARS,
            textless_page_ratio=settings.PDF_OCR_TEXTLESS_PAGE_RATIO,
        ):
            logger.info(
                "%s: OCR fallback triggered (%d non-whitespace chars, %d/%d textless pages)",
                path.name,
                stats.non_ws_chars,
                stats.textless_pages,
                stats.total_pages,
            )
            text, page_counts = self._convert(path, ocr=True)
            return self._raw_document(path, text, page_counts, extraction="ocr_fallback")

        return self._raw_document(path, text, page_counts, extraction="text")

    def _convert(self, path: Path, *, ocr: bool) -> tuple[str, dict[int, int]]:
        """Run one Docling conversion pass.

        Args:
            path: The PDF to convert.
            ocr: Whether this is the OCR pass (pass 2) or the fast
                text-layer pass (pass 1).

        Returns:
            The normalized markdown text and per-page character counts.
        """
        result = self._converter(ocr=ocr).convert(path)  # raises ConversionError on failure
        document = result.document
        return export_markdown(document), page_char_counts(document)

    def _converter(self, *, ocr: bool) -> "DocumentConverter":
        """Return the cached converter for a pass type, building it lazily.

        Args:
            ocr: Whether OCR is enabled for this converter.

        Returns:
            The (possibly cached) Docling converter.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        settings = get_settings()
        if not ocr:
            if self._fast_converter is None:
                options = PdfPipelineOptions(do_ocr=False)
                self._fast_converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
                )
            return self._fast_converter

        languages = settings.ocr_language_list
        key = (settings.OCR_ENGINE, tuple(languages), settings.PDF_OCR_FORCE_FULL_PAGE)
        if key not in self._ocr_converters:
            options = PdfPipelineOptions(
                do_ocr=True,
                ocr_options=get_ocr_options(
                    settings.OCR_ENGINE,
                    languages,
                    force_full_page=settings.PDF_OCR_FORCE_FULL_PAGE,
                ),
            )
            self._ocr_converters[key] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._ocr_converters[key]

    def _raw_document(
        self, path: Path, text: str, page_counts: dict[int, int], *, extraction: str
    ) -> RawDocument:
        """Assemble the parser result with provenance metadata.

        Args:
            path: The source PDF.
            text: The normalized extracted text.
            page_counts: Per-page character counts (for page attribution).
            extraction: ``"text"`` or ``"ocr_fallback"``.

        Returns:
            The :class:`RawDocument` handed to the chunker (assembled by the
            shared :func:`~varagity.ingest.parsers.docling_base.raw_document`).
        """
        return raw_document(path, text, page_counts, extraction=extraction)
