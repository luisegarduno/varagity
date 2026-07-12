"""Unit tests for the shared Docling core and the office/web parsers.

The pure helpers (`export_markdown`, `raw_document`, `page_char_counts`)
are exercised with light stub documents; the real-conversion tests run
Docling's lightweight office/web backends on the committed fixtures — no
layout models, no OCR (mirrors the PDF suite's real pass-1 test).
"""

from pathlib import Path

import pytest

from varagity.ingest.parsers import PARSER_REGISTRY, get_parser
from varagity.ingest.parsers.docling_base import (
    DoclingParser,
    export_markdown,
    page_char_counts,
    raw_document,
)
from varagity.ingest.parsers.office import OfficeParser
from varagity.ingest.parsers.web import WebParser

CORPUS = Path(__file__).parents[1] / "fixtures" / "corpus"


class StubDocument:
    """Just enough of a DoclingDocument for the pure helpers."""

    def __init__(self, markdown: str = "", pages: dict[int, object] | None = None) -> None:
        self.markdown = markdown
        self.pages = pages or {}

    def export_to_markdown(self) -> str:
        return self.markdown

    def iterate_items(self) -> list:
        return []


class TestExportMarkdown:
    """Newline normalization + hyphen repair over the raw export."""

    def test_normalizes_newlines_and_repairs_hyphens(self) -> None:
        document = StubDocument(markdown="a frame-\r\nwork and a net- work\rdone")
        assert export_markdown(document) == "a framework and a network\ndone"  # type: ignore[arg-type]

    def test_plain_markdown_passes_through(self) -> None:
        document = StubDocument(markdown="# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |")
        assert export_markdown(document) == document.markdown  # type: ignore[arg-type]


class TestRawDocument:
    """Provenance assembly shared by every Docling-backed parser."""

    def test_first_contributing_page_wins(self) -> None:
        raw = raw_document(Path("deck.pptx"), "text", {1: 0, 2: 40, 3: 12}, extraction="text")
        assert raw.source_meta["page"] == 2
        assert raw.source_meta["file_type"] == "pptx"
        assert raw.source_meta["file_name"] == "deck.pptx"
        assert raw.source_meta["extraction"] == "text"

    def test_no_pages_means_page_none(self) -> None:
        raw = raw_document(Path("doc.docx"), "text", {}, extraction="text")
        assert raw.source_meta["page"] is None

    def test_all_textless_pages_means_page_none(self) -> None:
        raw = raw_document(Path("blank.pdf"), "", {1: 0, 2: 0}, extraction="ocr_fallback")
        assert raw.source_meta["page"] is None
        assert raw.source_meta["extraction"] == "ocr_fallback"


class TestPageCharCounts:
    """The provenance counter degrades gracefully without pagination."""

    def test_unpaginated_document_yields_empty_counts(self) -> None:
        assert page_char_counts(StubDocument(pages={})) == {}  # type: ignore[arg-type]


class TestRegistry:
    """office/web resolve via the parser registry (spec §5.1)."""

    def test_office_and_web_parsers_are_registered(self) -> None:
        assert isinstance(get_parser("office"), OfficeParser)
        assert isinstance(get_parser("web"), WebParser)
        assert {"office", "web"} <= set(PARSER_REGISTRY)

    def test_both_share_the_docling_core(self) -> None:
        assert isinstance(get_parser("office"), DoclingParser)
        assert isinstance(get_parser("web"), DoclingParser)

    def test_invalid_verbose_raises(self) -> None:
        with pytest.raises(ValueError, match="verbose"):
            get_parser("office").extract(CORPUS / "gullwing_ferry_manual.docx", verbose=9)


class TestRealOfficeConversion:
    """Real Docling conversions of the committed office fixtures (no OCR)."""

    def test_docx_extracts_structure_with_no_page(self) -> None:
        raw = get_parser("office").extract(CORPUS / "gullwing_ferry_manual.docx", verbose=0)
        assert raw.source_meta["file_type"] == "docx"
        assert raw.source_meta["page"] is None  # no reliable docx pagination
        assert raw.source_meta["extraction"] == "text"  # never OCR'd
        assert "Gullwing Ferry Operations Manual" in raw.text
        assert raw.text.lstrip().startswith("#")  # heading markup survives
        # The planted fact.
        assert "800 meters before docking" in raw.text

    def test_pptx_records_the_first_contributing_slide_as_page(self) -> None:
        raw = get_parser("office").extract(CORPUS / "petrel_turbine_briefing.pptx", verbose=0)
        assert raw.source_meta["file_type"] == "pptx"
        assert raw.source_meta["page"] == 1  # slide → page (document-level, first with text)
        assert raw.source_meta["extraction"] == "text"
        # Slide titles and the slide-2 planted fact all reach the markdown.
        assert "Petrel-6 Tidal Turbine" in raw.text
        assert "Performance figures" in raw.text
        assert "3.4 megawatts" in raw.text

    def test_xlsx_exports_sheet_tables_with_sheet_as_page(self) -> None:
        raw = get_parser("office").extract(CORPUS / "quayside_inventory.xlsx", verbose=0)
        assert raw.source_meta["file_type"] == "xlsx"
        assert raw.source_meta["page"] == 1  # sheet → page (sheet identity)
        assert raw.source_meta["extraction"] == "text"
        # GFM tables with the planted table fact.
        table_lines = [line for line in raw.text.splitlines() if line.startswith("|")]
        assert any("Mooring bollard" in line and "148" in line for line in table_lines)
        # Content from the second sheet is extracted too.
        assert "Capstan winch" in raw.text


class TestRealWebConversion:
    """Real Docling conversion of the committed HTML fixture."""

    def test_html_extracts_structure_with_no_page(self) -> None:
        raw = get_parser("web").extract(CORPUS / "seagrass_survey.html", verbose=0)
        assert raw.source_meta["file_type"] == "html"
        assert raw.source_meta["page"] is None  # HTML has no pagination
        assert raw.source_meta["extraction"] == "text"
        assert "Wrenhaven Seagrass Survey 2025" in raw.text
        assert raw.text.lstrip().startswith("#")
        # The planted fact.
        assert "12 hectares" in raw.text
