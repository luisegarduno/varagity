"""Unit tests for the PDF parser (no real OCR execution).

The fallback-trigger logic is exercised as a pure function, the two-pass
control flow with the conversion passes stubbed, and the ``OCR_ENGINE``
factory by dispatch. One test runs a real Docling pass-1 conversion on the
digital fixture (no OCR) — first run downloads Docling's layout models.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from varagity.ingest.parsers import PARSER_REGISTRY, get_parser
from varagity.ingest.parsers.pdf import (
    OCR_ENGINE_FACTORIES,
    ExtractionStats,
    PdfParser,
    extraction_stats,
    get_ocr_options,
    needs_ocr_fallback,
)

CORPUS = Path(__file__).parents[1] / "fixtures" / "corpus"


@pytest.fixture
def pdf_settings(settings_env: Callable[..., None]) -> Callable[..., None]:
    """Pin the OCR-fallback settings to the plan defaults."""
    settings_env(
        PDF_OCR_FALLBACK="true",
        PDF_OCR_MIN_CHARS=50,
        PDF_OCR_TEXTLESS_PAGE_RATIO=0.2,
        PDF_OCR_FORCE_FULL_PAGE="false",
        OCR_ENGINE="easyocr",
        OCR_LANGUAGES="en",
    )
    return settings_env


class TestFallbackTrigger:
    """The trigger as a pure function (char count / textless ratio)."""

    def test_below_min_chars_triggers(self) -> None:
        stats = ExtractionStats(non_ws_chars=49, total_pages=1, textless_pages=0)
        assert needs_ocr_fallback(stats, min_chars=50, textless_page_ratio=0.2) is True

    def test_enough_chars_and_no_textless_pages_passes(self) -> None:
        stats = ExtractionStats(non_ws_chars=50, total_pages=2, textless_pages=0)
        assert needs_ocr_fallback(stats, min_chars=50, textless_page_ratio=0.2) is False

    def test_textless_ratio_at_threshold_triggers(self) -> None:
        # 1/5 pages textless == the 0.2 default → the mixed-document case.
        stats = ExtractionStats(non_ws_chars=5_000, total_pages=5, textless_pages=1)
        assert needs_ocr_fallback(stats, min_chars=50, textless_page_ratio=0.2) is True

    def test_textless_ratio_below_threshold_passes(self) -> None:
        stats = ExtractionStats(non_ws_chars=5_000, total_pages=10, textless_pages=1)
        assert needs_ocr_fallback(stats, min_chars=50, textless_page_ratio=0.2) is False

    def test_zero_pages_cannot_divide_and_min_chars_decides(self) -> None:
        stats = ExtractionStats(non_ws_chars=500, total_pages=0, textless_pages=0)
        assert needs_ocr_fallback(stats, min_chars=50, textless_page_ratio=0.2) is False

    def test_extraction_stats_counts(self) -> None:
        stats = extraction_stats(" ab\n\tc ", {1: 120, 2: 0, 3: 44, 4: 0})
        assert stats == ExtractionStats(non_ws_chars=3, total_pages=4, textless_pages=2)


class StubConvert:
    """Records `_convert` calls and returns scripted pass results."""

    def __init__(
        self,
        pass1: tuple[str, dict[int, int]] | Exception,
        pass2: tuple[str, dict[int, int]] | Exception = ("OCR RECOVERED TEXT " * 5, {1: 90}),
    ) -> None:
        self.pass1 = pass1
        self.pass2 = pass2
        self.calls: list[bool] = []

    def __call__(self, path: Path, *, ocr: bool) -> tuple[str, dict[int, int]]:
        self.calls.append(ocr)
        outcome = self.pass2 if ocr else self.pass1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


DIGITAL_PASS = ("A digital text layer with plenty of extractable characters.", {1: 59})
EMPTY_PASS = ("", {1: 0})


def _stubbed_parser(monkeypatch: pytest.MonkeyPatch, stub: StubConvert) -> PdfParser:
    parser = PdfParser()
    monkeypatch.setattr(parser, "_convert", stub)
    return parser


class TestTwoPassControlFlow:
    """extract() routing across pass 1, the trigger, and pass 2 (stubbed)."""

    def test_digital_document_stays_on_the_fast_path(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = StubConvert(pass1=DIGITAL_PASS)
        raw = _stubbed_parser(monkeypatch, stub).extract(Path("doc.pdf"), verbose=0)
        assert stub.calls == [False]  # never OCR'd
        assert raw.source_meta["extraction"] == "text"
        assert raw.source_meta["page"] == 1
        assert raw.source_meta["file_type"] == "pdf"

    def test_empty_pass1_falls_back_to_ocr(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = StubConvert(pass1=EMPTY_PASS)
        raw = _stubbed_parser(monkeypatch, stub).extract(Path("scan.pdf"), verbose=0)
        assert stub.calls == [False, True]
        assert raw.source_meta["extraction"] == "ocr_fallback"
        assert "OCR RECOVERED" in raw.text

    def test_textless_page_ratio_falls_back_to_ocr(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mixed document: page 1 digital, page 2 scanned → ratio 0.5 ≥ 0.2.
        stub = StubConvert(pass1=("Digital page one text, long enough to pass.", {1: 44, 2: 0}))
        raw = _stubbed_parser(monkeypatch, stub).extract(Path("mixed.pdf"), verbose=0)
        assert stub.calls == [False, True]
        assert raw.source_meta["extraction"] == "ocr_fallback"

    def test_pass1_exception_falls_back_to_ocr(
        self,
        pdf_settings: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        stub = StubConvert(pass1=RuntimeError("text-layer conversion exploded"))
        with caplog.at_level(logging.WARNING):
            raw = _stubbed_parser(monkeypatch, stub).extract(Path("odd.pdf"), verbose=0)
        assert stub.calls == [False, True]
        assert raw.source_meta["extraction"] == "ocr_fallback"
        assert any("retrying with OCR" in record.message for record in caplog.records)

    def test_fallback_disabled_returns_pass1_and_never_ocrs(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf_settings(PDF_OCR_FALLBACK="false")
        stub = StubConvert(pass1=EMPTY_PASS)
        raw = _stubbed_parser(monkeypatch, stub).extract(Path("scan.pdf"), verbose=0)
        assert stub.calls == [False]
        assert raw.source_meta["extraction"] == "text"
        assert raw.text == ""  # the loader's empty-extraction guard takes it from here

    def test_fallback_disabled_propagates_pass1_exception(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf_settings(PDF_OCR_FALLBACK="false")
        stub = StubConvert(pass1=RuntimeError("text-layer conversion exploded"))
        with pytest.raises(RuntimeError, match="exploded"):
            _stubbed_parser(monkeypatch, stub).extract(Path("odd.pdf"), verbose=0)
        assert stub.calls == [False]

    def test_pass2_exception_propagates_as_file_failure(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed OCR pass must fail the file, not half-record it.

        The loader counts the failure and re-attempts the file next run.
        """
        stub = StubConvert(pass1=EMPTY_PASS, pass2=RuntimeError("OCR exploded"))
        with pytest.raises(RuntimeError, match="OCR exploded"):
            _stubbed_parser(monkeypatch, stub).extract(Path("scan.pdf"), verbose=0)
        assert stub.calls == [False, True]

    def test_still_empty_after_ocr_returns_empty_for_the_loader_guard(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = StubConvert(pass1=EMPTY_PASS, pass2=EMPTY_PASS)
        raw = _stubbed_parser(monkeypatch, stub).extract(Path("blank.pdf"), verbose=0)
        assert stub.calls == [False, True]
        assert raw.source_meta["extraction"] == "ocr_fallback"
        assert raw.source_meta["page"] is None  # no page contributed text
        assert raw.text == ""

    def test_force_full_page_skips_pass1_entirely(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The corrupt-text-layer escape hatch: straight to full-page OCR."""
        pdf_settings(PDF_OCR_FORCE_FULL_PAGE="true")
        stub = StubConvert(pass1=DIGITAL_PASS)
        raw = _stubbed_parser(monkeypatch, stub).extract(Path("corrupt.pdf"), verbose=0)
        assert stub.calls == [True]
        assert raw.source_meta["extraction"] == "ocr_fallback"

    def test_invalid_verbose_raises(
        self, pdf_settings: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError, match="verbose"):
            _stubbed_parser(monkeypatch, StubConvert(pass1=DIGITAL_PASS)).extract(
                Path("doc.pdf"), verbose=3
            )


class TestOcrEngineFactory:
    """OCR_ENGINE dispatch → engine-specific docling options."""

    def test_easyocr_options(self) -> None:
        from docling.datamodel.pipeline_options import EasyOcrOptions

        options = get_ocr_options("easyocr", ["en", "de"])
        assert isinstance(options, EasyOcrOptions)
        assert options.lang == ["en", "de"]  # ISO 639-1, verbatim
        assert options.force_full_page_ocr is False
        # Models pinned inside docling's cache (one volume covers everything).
        assert options.model_storage_directory is not None
        assert options.model_storage_directory.endswith("models/EasyOcr")

    def test_tesseract_options_map_language_codes(self) -> None:
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions

        options = get_ocr_options("tesseract", ["en", "de", "chi_sim"], force_full_page=True)
        assert isinstance(options, TesseractCliOcrOptions)
        assert options.lang == ["eng", "deu", "chi_sim"]  # mapped; native codes pass through
        assert options.force_full_page_ocr is True

    def test_unknown_engine_raises_listing_available(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_ocr_options("rapidocr", ["en"])
        assert "rapidocr" in str(excinfo.value)
        for engine in OCR_ENGINE_FACTORIES:
            assert engine in str(excinfo.value)


class TestRegistry:
    """The pdf bucket resolves via the parser registry (spec §5.1)."""

    def test_pdf_parser_is_registered(self) -> None:
        assert isinstance(get_parser("pdf"), PdfParser)
        assert "pdf" in PARSER_REGISTRY


class TestRealPass1Extraction:
    """Real Docling conversion of the digital fixture (no OCR involved)."""

    def test_digital_fixture_extracts_structure_and_pages(
        self, pdf_settings: Callable[..., None]
    ) -> None:
        raw = get_parser("pdf").extract(CORPUS / "saltmere_observatory.pdf", verbose=0)

        # Fast path: never OCR'd, first content page recorded.
        assert raw.source_meta["extraction"] == "text"
        assert raw.source_meta["page"] == 1
        assert raw.source_meta["file_name"] == "saltmere_observatory.pdf"
        assert raw.source_meta["file_type"] == "pdf"

        # Structure-aware markdown: the heading and the bordered table survive.
        assert "Saltmere Coastal Observatory" in raw.text
        assert raw.text.lstrip().startswith("#")  # heading markup
        table_lines = [line for line in raw.text.splitlines() if line.startswith("|")]
        assert any("Ceilometer" in line and "SC-201" in line for line in table_lines)

        # Planted facts from both pages made it into the text.
        assert "12,400 meters" in raw.text  # page 1
        assert "7.3 terabytes" in raw.text  # page 2
