"""Resolve an ingested document to a pdfium-openable PDF (ADR-010).

The gate both preview routes share: kill switch, format eligibility,
``DOCS_PATH`` containment (mirroring the delete route's rule — a stored
``source`` is data, not authority), on-disk content-hash verification (an
edited-but-not-reingested file must degrade honestly, not preview the
wrong bytes), and — for PPTX — the cached LibreOffice conversion.
"""

from dataclasses import dataclass
from pathlib import Path

from varagity.config import Settings
from varagity.paths import resolve_contained
from varagity.preview.convert import ConversionFailed, ConversionUnavailable, ensure_pdf
from varagity.stores.records import DocumentInfo, content_hash

# The preview-eligible source formats: digital PDFs render directly; PPTX
# converts (slide N ↔ PDF page N under Impress export — the same identity
# docling relies on). Everything else keeps the full-text view.
_PREVIEWABLE_SUFFIXES = frozenset({".pdf", ".pptx"})


@dataclass(frozen=True)
class PreviewSource:
    """A document resolved to an openable PDF, or the reason it degraded.

    Attributes:
        pdf_path: The pdfium-openable PDF (the source itself for ``.pdf``,
            the cached conversion for ``.pptx``); ``None`` when degraded.
        reason: The degradable condition when ``pdf_path`` is ``None``
            (``preview_disabled`` | ``unsupported_type`` | ``file_missing``
            | ``file_changed`` | ``conversion_unavailable`` |
            ``conversion_failed``); ``None`` when ``pdf_path`` is set.
    """

    pdf_path: Path | None
    reason: str | None


def resolve_preview_source(info: DocumentInfo, settings: Settings) -> PreviewSource:
    """Resolve one ingested document to a PDF the preview can open.

    Every failure mode is a *degradable* outcome, never an exception — the
    routes turn ``reason`` into ``available:false`` (locate) or a 404 code
    (page image), and the GUI falls back to the full-text view.

    Args:
        info: The document's stored metadata (``get_document`` row).
        settings: The effective settings (kill switch, ``DOCS_PATH``,
            conversion timeout).

    Returns:
        The openable PDF path, or the reason there isn't one.
    """
    if not settings.PREVIEW_ENABLED:
        return PreviewSource(pdf_path=None, reason="preview_disabled")
    source = Path(info.source)
    suffix = source.suffix.lower()
    if suffix not in _PREVIEWABLE_SUFFIXES:
        return PreviewSource(pdf_path=None, reason="unsupported_type")
    docs_root = Path(settings.DOCS_PATH).resolve()
    resolved = resolve_contained(source, docs_root)
    if resolved is None or not resolved.is_file():
        return PreviewSource(pdf_path=None, reason="file_missing")
    if content_hash(resolved.read_bytes()) != info.content_hash:
        return PreviewSource(pdf_path=None, reason="file_changed")
    if suffix == ".pptx":
        try:
            converted = ensure_pdf(
                resolved, info.doc_id, timeout_s=settings.PREVIEW_CONVERT_TIMEOUT_S
            )
        except ConversionUnavailable:
            return PreviewSource(pdf_path=None, reason="conversion_unavailable")
        except ConversionFailed:
            return PreviewSource(pdf_path=None, reason="conversion_failed")
        return PreviewSource(pdf_path=converted, reason=None)
    return PreviewSource(pdf_path=resolved, reason=None)
