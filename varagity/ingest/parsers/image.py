"""Image parser: single-pass Docling conversion with OCR always on.

Bitmap formats carry no text layer, so unlike the PDF parser's two-pass
fallback there is nothing to try first: every conversion runs the
configured ``OCR_ENGINE`` in full-page mode through the same markdown/
table/provenance pipeline as the other Docling-backed parsers
(``docling_base``). Documents carry ``extraction="ocr"`` on every chunk —
distinct from the PDF parser's ``"ocr_fallback"``, because OCR is this
format's only extraction path, not a recovery.

Docling imports are deferred to call time: importing this module (which
happens on every CLI start via parser self-registration) must not pay for
Docling's model machinery.
"""

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
from varagity.ingest.parsers.pdf import get_ocr_options

if TYPE_CHECKING:  # heavy imports, type-only (runtime imports are lazy)
    from docling.document_converter import DocumentConverter


@register("image")
class ImageParser:
    """Parser for the ``image`` bucket: Docling's image pipeline, OCR-only.

    Converters are built lazily and cached per OCR configuration (the
    registry instantiates one parser per process; tests and long-lived
    processes may change OCR settings between calls), so the layout/OCR
    models load once per configuration, not once per file.
    """

    def __init__(self) -> None:
        """Initialize the converter cache (nothing heavy happens here)."""
        # Keyed by (engine, languages) — full-page OCR is unconditional
        # for images, so it is not part of the key.
        self._converters: dict[tuple[str, tuple[str, ...]], DocumentConverter] = {}

    def extract(self, path: Path, verbose: int | None = None) -> RawDocument:
        """OCR an image into markdown text with provenance.

        Args:
            path: The image file to convert.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The extracted document. ``source_meta`` carries ``page``
            (``1`` when any text was recognized — Docling models an image
            as a one-page document) and ``extraction="ocr"``.

        Raises:
            ValueError: If ``verbose`` is invalid.
            docling.exceptions.ConversionError: If conversion fails (a
                corrupt or unreadable image) — the loader counts the file
                as failed and the run continues.
        """
        settings = get_settings()
        check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        text, page_counts = self._convert(path)
        return raw_document(path, text, page_counts, extraction="ocr")

    def _convert(self, path: Path) -> tuple[str, dict[int, int]]:
        """Run the (cached) OCR converter on one image.

        Args:
            path: The image to convert.

        Returns:
            The normalized markdown text and per-page character counts.
        """
        result = self._converter().convert(path)  # raises ConversionError on failure
        document = result.document
        return export_markdown(document), page_char_counts(document)

    def _converter(self) -> "DocumentConverter":
        """Return the converter for the current OCR settings, building lazily.

        Returns:
            The (possibly cached) Docling converter, configured for
            full-page OCR — an image has no embedded text layer to
            preserve, so there is nothing for partial-page OCR to merge
            with.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption

        settings = get_settings()
        languages = settings.ocr_language_list
        key = (settings.OCR_ENGINE, tuple(languages))
        if key not in self._converters:
            options = PdfPipelineOptions(
                do_ocr=True,
                ocr_options=get_ocr_options(settings.OCR_ENGINE, languages, force_full_page=True),
            )
            self._converters[key] = DocumentConverter(
                format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)}
            )
        return self._converters[key]
