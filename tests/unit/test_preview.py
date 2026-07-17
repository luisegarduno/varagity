"""Unit tests for the preview package (ADR-010): normalize, locate, render, convert.

Locate/render run against the real fixture PDFs (pdfium is a library, not
a service — no mocking layer): ``saltmere_observatory.pdf`` (2 digital
pages), ``breakwater_survey.pdf`` (mixed: page 1 digital, page 2 scanned),
``blank_pages.pdf`` (no text layer at all). Conversion paths run with a
stubbed ``subprocess.run``; one real LibreOffice conversion of
``petrel_turbine_briefing.pptx`` runs wherever ``soffice`` exists (this
host, the api image) and skips on bare CI.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from varagity.preview import (
    ConversionFailed,
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
PETREL = FIXTURES / "petrel_turbine_briefing.pptx"

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


class TestEnsurePdf:
    @pytest.fixture
    def cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Isolate the conversion cache under this test's tmp dir.

        ``conversion_cache_path`` keys off ``tempfile.gettempdir()``; without
        this, tests would share (and pollute) the host's real ``/tmp`` cache.
        """
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        return tmp_path / "varagity-preview"

    @pytest.fixture
    def deck(self, tmp_path: Path) -> Path:
        deck = tmp_path / "deck.pptx"
        deck.write_bytes(b"not really a deck")
        return deck

    def test_cache_path_is_content_addressed_by_doc_id(self) -> None:
        path = conversion_cache_path("a398491c7441925f")
        assert path.name == "a398491c7441925f.pdf"
        assert path.parent.name == "varagity-preview"

    def test_missing_soffice_reports_conversion_unavailable(
        self, cache_dir: Path, deck: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ConversionUnavailable):
            ensure_pdf(deck, "d0c0000000000001", timeout_s=5)

    def test_cache_hit_needs_neither_soffice_nor_a_runner(
        self, cache_dir: Path, deck: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A converted deck keeps answering even if LibreOffice disappears."""
        cache_dir.mkdir(parents=True)
        (cache_dir / "d0c0000000000002.pdf").write_bytes(b"%PDF-cached")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.setattr(subprocess, "run", _never_run)
        result = ensure_pdf(deck, "d0c0000000000002", timeout_s=5)
        assert result == cache_dir / "d0c0000000000002.pdf"
        assert result.read_bytes() == b"%PDF-cached"

    def test_conversion_command_and_atomic_cache_move(
        self, cache_dir: Path, deck: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def fake_run(
            command: list[str], *, timeout: int, capture_output: bool, check: bool
        ) -> subprocess.CompletedProcess[bytes]:
            seen["command"], seen["timeout"] = command, timeout
            out_dir = Path(command[command.index("--outdir") + 1])
            (out_dir / f"{deck.stem}.pdf").write_bytes(b"%PDF-converted")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(shutil, "which", lambda name: "/stub/soffice")
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = ensure_pdf(deck, "d0c0000000000003", timeout_s=42)
        assert result == cache_dir / "d0c0000000000003.pdf"
        assert result.read_bytes() == b"%PDF-converted"
        assert seen["timeout"] == 42
        command = seen["command"]
        assert isinstance(command, list)
        assert command[0] == "/stub/soffice"
        assert "--headless" in command
        assert "--norestore" in command
        assert any(arg.startswith("-env:UserInstallation=file://") for arg in command)
        assert command[command.index("--convert-to") + 1] == "pdf"
        assert command[-1] == str(deck)
        # The scratch dir is gone: only the finished artifact lives in the cache.
        assert [path.name for path in cache_dir.iterdir()] == ["d0c0000000000003.pdf"]

    def test_timeout_reports_conversion_failed_and_caches_nothing(
        self, cache_dir: Path, deck: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(
            command: list[str], *, timeout: int, capture_output: bool, check: bool
        ) -> subprocess.CompletedProcess[bytes]:
            raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

        monkeypatch.setattr(shutil, "which", lambda name: "/stub/soffice")
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ConversionFailed, match="timed out"):
            ensure_pdf(deck, "d0c0000000000004", timeout_s=5)
        assert not (cache_dir / "d0c0000000000004.pdf").exists()

    def test_nonzero_exit_reports_conversion_failed_with_stderr(
        self, cache_dir: Path, deck: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(
            command: list[str], *, timeout: int, capture_output: bool, check: bool
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(command, 77, stdout=b"", stderr=b"no filter found")

        monkeypatch.setattr(shutil, "which", lambda name: "/stub/soffice")
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ConversionFailed, match="exited 77.*no filter found"):
            ensure_pdf(deck, "d0c0000000000005", timeout_s=5)

    def test_missing_output_reports_conversion_failed(
        self, cache_dir: Path, deck: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LibreOffice can exit 0 without producing anything (bad input filter)."""

        def fake_run(
            command: list[str], *, timeout: int, capture_output: bool, check: bool
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(shutil, "which", lambda name: "/stub/soffice")
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ConversionFailed, match="produced no PDF"):
            ensure_pdf(deck, "d0c0000000000006", timeout_s=5)

    @pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice not installed")
    def test_real_deck_converts_and_slide_n_is_page_n(self, cache_dir: Path) -> None:
        """End-to-end on the fixture deck: convert, then locate slide 2's text."""
        pdf = ensure_pdf(PETREL, "d0cpetre1b41ef1n", timeout_s=120)
        assert pdf == cache_dir / "d0cpetre1b41ef1n.pdf"
        assert pdf.read_bytes()[:5] == b"%PDF-"
        result = locate(
            pdf,
            "The Petrel-6 tidal turbine produces 3.4 megawatts at peak flow. "
            "Availability last quarter held at 96 percent.",
            min_coverage=0.3,
        )
        assert result.page == 2  # slide 2 of the deck IS page 2 of the PDF
        assert result.page_count == 3
        assert result.rects
        for x0, y0, x1, y1 in result.rects:
            assert 0.0 <= x0 < x1 <= 1.0
            assert 0.0 <= y0 < y1 <= 1.0


def _never_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    raise AssertionError("subprocess.run must not be invoked on a cache hit")
