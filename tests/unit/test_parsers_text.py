"""Unit tests for the text parser and the parser registry (spec §15.2)."""

from pathlib import Path

import pytest

from varagity.ingest.parsers import PARSER_REGISTRY, get_parser
from varagity.ingest.parsers.text import TextParser, remove_hyphen_space


class TestRemoveHyphenSpace:
    """The reference implementation's regex cases, carried over."""

    def test_line_broken_word(self) -> None:
        assert remove_hyphen_space("frame-\nwork") == "framework"

    def test_hyphen_space_word(self) -> None:
        assert remove_hyphen_space("frame- work") == "framework"

    def test_line_break_with_surrounding_spaces(self) -> None:
        assert remove_hyphen_space("frame- \n work") == "framework"

    def test_plain_hyphenated_compound_is_kept(self) -> None:
        assert remove_hyphen_space("state-of-the-art") == "state-of-the-art"

    def test_standalone_dash_is_kept(self) -> None:
        assert remove_hyphen_space("a - b") == "a - b"


class TestTextParser:
    def test_reads_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode.txt"
        path.write_text("naïve café — ünïcodé ✓", encoding="utf-8")
        raw = TextParser().extract(path, verbose=0)
        assert "naïve café" in raw.text
        assert "✓" in raw.text

    def test_normalizes_newlines(self, tmp_path: Path) -> None:
        path = tmp_path / "crlf.txt"
        path.write_bytes(b"line one\r\nline two\rline three\n")
        raw = TextParser().extract(path, verbose=0)
        assert raw.text == "line one\nline two\nline three\n"

    def test_repairs_hyphenation(self, tmp_path: Path) -> None:
        path = tmp_path / "hyphen.txt"
        path.write_text("A useful frame-\nwork for testing.")
        raw = TextParser().extract(path, verbose=0)
        assert "framework" in raw.text

    def test_source_meta(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.MD"
        path.write_text("hello")
        meta = TextParser().extract(path, verbose=0).source_meta
        assert meta["source"] == str(path.resolve())
        assert meta["file_name"] == "notes.MD"
        assert meta["file_type"] == "md"  # extension lowered
        assert meta["page"] is None

    def test_invalid_utf8_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "binary.txt"
        path.write_bytes(b"\xff\xfe\x00broken")
        with pytest.raises(UnicodeDecodeError):
            TextParser().extract(path, verbose=0)


class TestParserRegistry:
    def test_text_parser_registered(self) -> None:
        assert isinstance(get_parser("text"), TextParser)
        assert "text" in PARSER_REGISTRY

    def test_unknown_name_lists_available(self) -> None:
        with pytest.raises(KeyError, match="'text'"):
            get_parser("bogus")
