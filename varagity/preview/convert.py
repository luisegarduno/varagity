"""PPTX→PDF conversion for previews — Phase 1 contract, Phase 2 wiring (ADR-010).

Phase 1 defines the module's whole surface — the exception types the
resolver maps onto wire reasons and the content-addressed cache location —
while :func:`ensure_pdf` itself reports the conversion unavailable. Phase 2
replaces only its body with the locked, cached, timed-out headless
LibreOffice subprocess (``soffice --headless --convert-to pdf``), touching
no caller.
"""

import tempfile
from pathlib import Path


class ConversionUnavailable(Exception):
    """No converter is available to this process (degrade, never crash).

    Raised when LibreOffice is absent — a host-mode run without it loses
    only PPTX previews (the route answers ``conversion_unavailable``).
    """


class ConversionFailed(Exception):
    """A conversion ran and failed (non-zero exit, timeout, missing output)."""


def conversion_cache_path(doc_id: str) -> Path:
    """Return the cached-PDF path for one document's conversion.

    Content-addressed: ``doc_id`` hashes the source path and its byte
    content, so a cache hit can never be stale. The cache lives in the
    process's temp directory — container-ephemeral by design (a restart
    re-pays one conversion per deck).

    Args:
        doc_id: The document's stable id.

    Returns:
        The cache path (existence not implied).
    """
    return Path(tempfile.gettempdir()) / "varagity-preview" / f"{doc_id}.pdf"


def ensure_pdf(source: Path, doc_id: str, *, timeout_s: int) -> Path:
    """Return a PDF rendition of ``source``, converting on first use.

    Phase 1 stub: the LibreOffice wiring lands in Phase 2 (in the api
    image); until then every call reports the conversion unavailable and
    the preview routes degrade to the full-text view.

    Args:
        source: The source document (a ``.pptx``).
        doc_id: The document's stable id (the cache key).
        timeout_s: Conversion timeout in seconds (``PREVIEW_CONVERT_TIMEOUT_S``;
            unused until Phase 2).

    Returns:
        The pdfium-openable converted PDF.

    Raises:
        ConversionUnavailable: Always, in Phase 1.
        ConversionFailed: Never in Phase 1 (Phase 2: exit code / timeout /
            missing output).
    """
    raise ConversionUnavailable(
        "PPTX→PDF conversion is not wired yet (Phase 2 adds LibreOffice to the api image)"
    )
