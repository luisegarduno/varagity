"""Render one document page to a PNG (ADR-010).

The image side of the preview pair: the locate route names a page, this
renders it at ``PREVIEW_RENDER_WIDTH`` for the evidence panel's ``<img>``.
Every pdfium call — the PIL encode included, since the bitmap buffer
belongs to pdfium — runs under :data:`varagity.preview.locate.PDFIUM_LOCK`.
"""

import io
from pathlib import Path

import pypdfium2 as pdfium

from varagity.preview.locate import PDFIUM_LOCK


def render_page_png(pdf_path: Path, page: int, *, width: int) -> bytes:
    """Rasterize one page (1-based) to a PNG of the given pixel width.

    Height follows the page's aspect ratio (``scale = width / page_width``
    in PDF canvas units).

    Args:
        pdf_path: A pdfium-openable PDF.
        page: 1-based page number.
        width: Target image width in pixels.

    Returns:
        The PNG bytes.

    Raises:
        IndexError: If ``page`` is outside ``1..page_count``.
        pypdfium2.PdfiumError: If the file cannot be opened as a PDF.
    """
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            page_count = len(pdf)
            if not 1 <= page <= page_count:
                raise IndexError(f"page {page} out of range 1–{page_count}")
            page_obj = pdf[page - 1]
            try:
                page_width, _ = page_obj.get_size()
                bitmap = page_obj.render(scale=width / page_width)
                image = bitmap.to_pil()
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue()
            finally:
                page_obj.close()
        finally:
            pdf.close()
