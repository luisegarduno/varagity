"""Unit tests for varagity.ingest.discovery (spec §15.2 "discovery" row)."""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from varagity.ingest.discovery import Buckets, discover_documents


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("alpha")
    (root / "b.md").write_text("bravo")
    (root / "sub" / "c.md").write_text("charlie (nested)")
    (root / "d.pdf").write_bytes(b"%PDF-fake")
    (root / "e.png").write_bytes(b"\x89PNG")
    (root / "f.TXT").write_text("case-insensitive extension")
    (root / "g.docx").write_bytes(b"PK-fake-docx")
    (root / "h.pptx").write_bytes(b"PK-fake-pptx")
    (root / "i.xlsx").write_bytes(b"PK-fake-xlsx")
    (root / "j.html").write_text("<h1>web</h1>")
    (root / "sub" / "k.htm").write_text("<p>nested, short extension</p>")
    (root / "l.PPTX").write_bytes(b"PK-case-insensitive")
    return root


def test_bucketing(corpus: Path, settings_env: Callable[..., None]) -> None:
    settings_env(ALLOWED_EXTENSIONS=".pdf,.txt,.md,.docx,.pptx,.xlsx,.html,.htm")
    buckets = discover_documents(str(corpus), verbose=0)
    assert [p.name for p in buckets.text_like] == ["a.txt", "b.md", "f.TXT", "c.md"]
    assert [p.name for p in buckets.pdf] == ["d.pdf"]
    assert [p.name for p in buckets.office] == ["g.docx", "h.pptx", "i.xlsx", "l.PPTX"]
    assert [p.name for p in buckets.web] == ["j.html", "k.htm"]
    assert buckets.total == 11


def test_v1_whitelist_keeps_office_and_web_out(
    corpus: Path, settings_env: Callable[..., None]
) -> None:
    """A narrow ALLOWED_EXTENSIONS still gates the new buckets (whitelist first)."""
    settings_env(ALLOWED_EXTENSIONS=".pdf,.txt,.md")
    buckets = discover_documents(str(corpus), verbose=0)
    assert buckets.office == []
    assert buckets.web == []
    assert buckets.total == 5


def test_recursive_glob_finds_nested_files(corpus: Path, settings_env: Callable[..., None]) -> None:
    settings_env(ALLOWED_EXTENSIONS=".md")
    buckets = discover_documents(str(corpus), verbose=0)
    assert {p.name for p in buckets.text_like} == {"b.md", "c.md"}


def test_disallowed_extensions_ignored_at_debug(
    corpus: Path,
    settings_env: Callable[..., None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_env(ALLOWED_EXTENSIONS=".txt,.md")
    with caplog.at_level(logging.DEBUG, logger="varagity.ingest.discovery"):
        buckets = discover_documents(str(corpus), verbose=0)
    assert buckets.pdf == []  # .pdf not in the whitelist here
    ignored = [
        r.getMessage() for r in caplog.records if "not in ALLOWED_EXTENSIONS" in r.getMessage()
    ]
    assert any("e.png" in message for message in ignored)
    assert any("d.pdf" in message for message in ignored)


def test_missing_directory_warns_and_returns_empty(
    tmp_path: Path,
    settings_env: Callable[..., None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_env(ALLOWED_EXTENSIONS=".txt")
    with caplog.at_level(logging.WARNING):
        buckets = discover_documents(str(tmp_path / "does-not-exist"), verbose=0)
    assert buckets.total == 0
    assert any("does not exist" in r.message for r in caplog.records)


def test_allowed_but_unbucketed_extension_warns(
    tmp_path: Path,
    settings_env: Callable[..., None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.rtf").write_text("not a bucketed format")
    settings_env(ALLOWED_EXTENSIONS=".rtf")
    with caplog.at_level(logging.WARNING):
        buckets = discover_documents(str(root), verbose=0)
    assert buckets.total == 0
    assert any("no ingestion bucket" in r.message for r in caplog.records)


def test_invalid_verbose_raises(corpus: Path, settings_env: Callable[..., None]) -> None:
    settings_env(ALLOWED_EXTENSIONS=".txt")
    with pytest.raises(ValueError, match="verbose"):
        discover_documents(str(corpus), verbose=7)


def test_buckets_total() -> None:
    assert Buckets().total == 0
    assert (
        Buckets(
            text_like=[Path("a.txt")],
            pdf=[Path("b.pdf")],
            office=[Path("c.docx"), Path("d.xlsx")],
            web=[Path("e.html")],
        ).total
        == 5
    )


def test_by_bucket_enumerates_every_bucket() -> None:
    """The render seam names each bucket exactly once, in a stable order."""
    buckets = Buckets(office=[Path("c.docx")])
    named = dict(buckets.by_bucket())
    assert list(named) == ["text_like", "pdf", "office", "web"]
    assert named["office"] == [Path("c.docx")]
