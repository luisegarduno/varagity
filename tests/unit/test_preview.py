"""Unit tests for the preview package (ADR-010): normalize, locate, render, convert.

Locate/render run against the real fixture PDFs (pdfium is a library, not
a service — no mocking layer): ``saltmere_observatory.pdf`` (2 digital
pages), ``breakwater_survey.pdf`` (mixed: page 1 digital, page 2 scanned),
``blank_pages.pdf`` (no text layer at all).
"""

from pathlib import Path

import pytest

from varagity.preview import (
    ConversionUnavailable,
    conversion_cache_path,
    ensure_pdf,
    locate,
    normalize_chunk_text,
    render_page_png,
    snippets,
    words,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"
SALTMERE = FIXTURES / "saltmere_observatory.pdf"
BREAKWATER = FIXTURES / "breakwater_survey.pdf"
BLANK = FIXTURES / "blank_pages.pdf"

# Markdown-decorated chunks the way docling exports them, matching the
# fixtures' real page text.
PAGE_1_TABLE_CHUNK = """## Instrument schedule

| Instrument           | Serial | Calibration interval |
|----------------------|--------|----------------------|
| Ceilometer "Firefly" | SC-201 | 90 days              |
| Anemometer array     | SC-114 | 30 days              |
| Salinity probe       | SC-377 | 45 days              |"""

PAGE_2_CHUNK = """## Data archive

During each winter season the observatory archives **7.3 terabytes** of lidar returns
to vaults on the mainland. The archive is mirrored to the Thalassic Institute every
14 days by microwave link, and a sealed tape copy travels out with the quarterly
supply boat.

<!-- image -->"""


class TestNormalizeChunkText:
    def test_table_chunk_loses_pipes_separators_and_heading(self) -> None:
        normalized = normalize_chunk_text(PAGE_1_TABLE_CHUNK)
        assert "|" not in normalized
        assert "#" not in normalized
        assert "---" not in normalized
        assert 'Ceilometer "Firefly" SC-201 90 days' in normalized

    def test_placeholder_comments_vanish(self) -> None:
        text = "Before\n\n<!-- image -->\n\nAfter <!-- formula-not-decoded --> end"
        assert normalize_chunk_text(text) == "Before After end"

    def test_heading_markers_stripped_but_words_kept(self) -> None:
        assert normalize_chunk_text("### Data archive\nBody text") == "Data archive Body text"

    def test_emphasis_unwraps_without_eating_identifiers(self) -> None:
        assert normalize_chunk_text("**bold** and *italic* and _wrapped_") == (
            "bold and italic and wrapped"
        )
        # An identifier's inner underscore is page text, not emphasis.
        assert normalize_chunk_text("the doc_id field") == "the doc_id field"

    def test_markdown_escapes_unwind(self) -> None:
        assert normalize_chunk_text(r"the doc\_id field \[sic\]") == "the doc_id field [sic]"

    def test_dot_leaders_and_bullets_drop(self) -> None:
        assert normalize_chunk_text("- Intro ..... 4\n- Body .... 9") == "Intro 4 Body 9"

    def test_ordered_list_markers_drop(self) -> None:
        assert normalize_chunk_text("1. First step\n2) Second step") == "First step Second step"

    def test_whitespace_collapses(self) -> None:
        assert normalize_chunk_text("a\r\n b\t\tc\n\n\nd") == "a b c d"


class TestWords:
    def test_lowercases_and_drops_punctuation(self) -> None:
        assert words('Ceilometer "Firefly", 12,400 m!') == [
            "ceilometer",
            "firefly",
            "12",
            "400",
            "m",
        ]

    def test_empty_text_is_empty(self) -> None:
        assert words("  \n ") == []


class TestSnippets:
    def test_short_sentence_searches_whole(self) -> None:
        assert snippets("Data archive holds tapes.") == ["Data archive holds tapes."]

    def test_long_sentence_windows_overlap_and_cover_the_tail(self) -> None:
        text = " ".join(f"w{i}" for i in range(11))  # 11 tokens, no sentence end
        needles = snippets(text, size=8, stride=4)
        assert needles[0] == "w0 w1 w2 w3 w4 w5 w6 w7"
        assert needles[-1] == "w3 w4 w5 w6 w7 w8 w9 w10"  # tail window
        joined = " ".join(needles)
        assert all(f"w{i}" in joined.split() for i in range(11))  # every token covered

    def test_exact_multiple_has_no_duplicate_tail(self) -> None:
        text = " ".join(f"w{i}" for i in range(12))  # (12-8) % 4 == 0
        needles = snippets(text, size=8, stride=4)
        assert needles == ["w0 w1 w2 w3 w4 w5 w6 w7", "w4 w5 w6 w7 w8 w9 w10 w11"]

    def test_sentences_split_before_windowing(self) -> None:
        assert snippets("One two. Three four!") == ["One two.", "Three four!"]

    def test_duplicates_dedupe(self) -> None:
        assert snippets("Same line. Same line.") == ["Same line."]


class TestLocate:
    def test_page_two_excerpt_locates_page_two_with_rects(self) -> None:
        result = locate(SALTMERE, PAGE_2_CHUNK, min_coverage=0.3)
        assert result.page == 2
        assert result.page_count == 2
        assert result.coverage > 0.9
        assert result.rects
        for x0, y0, x1, y1 in result.rects:
            assert 0.0 <= x0 < x1 <= 1.0
            assert 0.0 <= y0 < y1 <= 1.0  # top-left origin: y0 is above y1

    def test_page_one_table_chunk_locates_page_one(self) -> None:
        result = locate(SALTMERE, PAGE_1_TABLE_CHUNK, min_coverage=0.3)
        assert result.page == 1
        assert result.rects

    def test_rects_are_deduplicated(self) -> None:
        result = locate(SALTMERE, PAGE_2_CHUNK, min_coverage=0.3)
        rounded = [tuple(round(v, 4) for v in rect) for rect in result.rects]
        assert len(rounded) == len(set(rounded))  # stride overlap re-finds, dedupe drops

    def test_text_bearing_page_beats_the_scanned_one(self) -> None:
        """breakwater_survey.pdf: page 1 is digital, page 2 image-only."""
        result = locate(
            BREAKWATER,
            "The Halcyon Breakwater shelters the ferry terminal",
            min_coverage=0.3,
        )
        assert result.page == 1
        assert result.page_count == 2

    def test_gibberish_reports_no_match(self) -> None:
        result = locate(
            SALTMERE, "zorblatt quuxification fnord manifold retrograde", min_coverage=0.3
        )
        assert result.page is None
        assert result.coverage < 0.3
        assert result.rects == []
        assert result.page_count == 2  # still reported (diagnostics)

    def test_textless_document_reports_no_match(self) -> None:
        """blank_pages.pdf has no text layer on any page — nothing can match."""
        result = locate(BLANK, PAGE_2_CHUNK, min_coverage=0.3)
        assert result.page is None
        assert result.coverage == 0.0

    def test_decoration_only_chunk_reports_no_match(self) -> None:
        result = locate(SALTMERE, "<!-- image -->\n\n|---|---|", min_coverage=0.3)
        assert result.page is None


class TestRenderPagePng:
    def test_renders_png_at_the_requested_width(self) -> None:
        png = render_page_png(SALTMERE, 1, width=640)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        # Width rides in the IHDR chunk (bytes 16-20) — no PIL round-trip needed.
        assert int.from_bytes(png[16:20], "big") == 640

    @pytest.mark.parametrize("page", [0, 3, -1])
    def test_out_of_range_page_raises_index_error(self, page: int) -> None:
        with pytest.raises(IndexError, match="out of range"):
            render_page_png(SALTMERE, page, width=640)


class TestConvertStub:
    def test_ensure_pdf_reports_conversion_unavailable(self, tmp_path: Path) -> None:
        deck = tmp_path / "deck.pptx"
        deck.write_bytes(b"not really a deck")
        with pytest.raises(ConversionUnavailable):
            ensure_pdf(deck, "a" * 16, timeout_s=5)

    def test_cache_path_is_content_addressed_by_doc_id(self) -> None:
        path = conversion_cache_path("a398491c7441925f")
        assert path.name == "a398491c7441925f.pdf"
        assert path.parent.name == "varagity-preview"
