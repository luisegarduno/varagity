"""Unit tests for the image parser (OCR-only Docling pipeline).

The extraction/provenance logic is exercised with a stubbed ``_convert``
(mirroring the PDF suite); the real-conversion test OCRs the committed
fixture through the tesseract CLI engine and skips where the binary is
absent (CI — mirrors the LibreOffice-gated preview conversion test).
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from varagity.ingest.parsers import PARSER_REGISTRY, get_parser
from varagity.ingest.parsers.image import ImageParser

FORMATS = Path(__file__).parents[1] / "fixtures" / "formats"


def _stubbed_parser(
    monkeypatch: pytest.MonkeyPatch, text: str, pages: dict[int, int]
) -> ImageParser:
    parser = ImageParser()
    monkeypatch.setattr(parser, "_convert", lambda path: (text, pages))
    return parser


class TestRegistry:
    def test_image_parser_is_registered(self) -> None:
        assert isinstance(get_parser("image"), ImageParser)
        assert "image" in PARSER_REGISTRY


class TestStubbedExtract:
    """Extraction/provenance logic with ``_convert`` stubbed out."""

    def test_extract_carries_ocr_provenance_and_page(
        self, settings_env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings_env()
        parser = _stubbed_parser(monkeypatch, "## BERTH 12 CLOSED", {1: 14})
        raw = parser.extract(Path("sign.png"), verbose=0)
        assert raw.text == "## BERTH 12 CLOSED"
        assert raw.source_meta["extraction"] == "ocr"
        assert raw.source_meta["page"] == 1
        assert raw.source_meta["file_type"] == "png"
        assert raw.source_meta["file_name"] == "sign.png"

    def test_blank_image_yields_empty_text_for_the_loader_guard(
        self, settings_env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings_env()
        raw = _stubbed_parser(monkeypatch, "", {1: 0}).extract(Path("blank.png"), verbose=0)
        assert raw.text == ""
        assert raw.source_meta["page"] is None  # no page contributed text
        assert raw.source_meta["extraction"] == "ocr"

    def test_invalid_verbose_raises(self, settings_env: Callable[..., None]) -> None:
        settings_env()
        with pytest.raises(ValueError, match="verbose"):
            ImageParser().extract(Path("x.png"), verbose=9)


class TestConverterCache:
    """The converter cache keys on the OCR configuration, not per-file."""

    def test_cache_keyed_by_engine_and_languages(self, settings_env: Callable[..., None]) -> None:
        settings_env(OCR_ENGINE="tesseract", OCR_LANGUAGES="en")
        parser = ImageParser()
        first = parser._converter()
        assert parser._converter() is first  # same settings → cached
        settings_env(OCR_ENGINE="tesseract", OCR_LANGUAGES="en,fr")
        assert parser._converter() is not first  # new languages → new converter


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")
class TestRealImageOcr:
    """Real Docling image conversion through the tesseract CLI engine."""

    def test_png_is_ocred_with_page_and_ocr_provenance(
        self, settings_env: Callable[..., None]
    ) -> None:
        settings_env(OCR_ENGINE="tesseract", OCR_LANGUAGES="en")
        raw = get_parser("image").extract(FORMATS / "berth_sign.png", verbose=0)
        assert raw.source_meta["file_type"] == "png"
        assert raw.source_meta["extraction"] == "ocr"
        assert raw.source_meta["page"] == 1
        assert "BERTH 12 CLOSED" in raw.text.upper()
        assert "DREDGING" in raw.text.upper()
