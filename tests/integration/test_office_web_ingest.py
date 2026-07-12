"""Integration tests for office/web ingestion (``-m integration``).

Real Docling conversions of the committed ``.docx``/``.pptx``/``.xlsx``/
``.html`` fixtures through the full ``ingest_corpus`` path — parse → chunk
→ store — so the planted facts land in stored chunks with the right
provenance. Stores/embeddings are the in-memory fakes: what's under test
is the office/web extraction path, not the databases. No OCR engines are
involved (digital text — spec_v2 §8.2).
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.unit.test_loader import FakeBM25, FakeEmbeddings, FakeStore
from varagity.ingest.loader import ingest_corpus
from varagity.stores.records import ChunkRecord

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).parents[1] / "fixtures" / "corpus"

# Fixture → (file_type, expected document-level page, a planted fact).
OFFICE_WEB_FIXTURES = {
    "gullwing_ferry_manual.docx": ("docx", None, "800 meters before docking"),
    "petrel_turbine_briefing.pptx": ("pptx", 1, "3.4 megawatts"),
    "quayside_inventory.xlsx": ("xlsx", 1, "Mooring bollard"),
    "seagrass_survey.html": ("html", None, "12 hectares"),
}


@pytest.fixture
def office_web_settings(settings_env: Callable[..., None]) -> Callable[..., None]:
    """Pin the ingest settings for the office/web path (no contextualization)."""
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md,.docx,.pptx,.xlsx,.html,.htm",
        CHUNKING_STRATEGY="recursive_character",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        CONTEXTUALIZE="false",
        EMBEDDING_MODEL="test-model",
    )
    return settings_env


@pytest.fixture
def office_web_corpus(tmp_path: Path) -> Path:
    """A corpus directory holding only the four new-format fixtures."""
    root = tmp_path / "docs"
    root.mkdir()
    for name in OFFICE_WEB_FIXTURES:
        (root / name).write_bytes((CORPUS / name).read_bytes())
    return root


def _records_for(store: FakeStore, file_name: str) -> list[ChunkRecord]:
    return [record for record in store.records if record.file_name == file_name]


def test_all_four_formats_ingest_with_provenance(
    office_web_settings: Callable[..., None], office_web_corpus: Path
) -> None:
    """Each format converts, chunks, and stores with format-true provenance."""
    store = FakeStore()
    summary = ingest_corpus(
        str(office_web_corpus), store=store, bm25=FakeBM25(), embeddings=FakeEmbeddings(), verbose=0
    )

    assert summary.discovered == 4
    assert summary.ingested == 4
    assert summary.failed == summary.no_text == summary.unsupported == 0
    assert {doc["file_type"] for doc in store.documents.values()} == {
        "docx",
        "pptx",
        "xlsx",
        "html",
    }

    for file_name, (file_type, page, planted_fact) in OFFICE_WEB_FIXTURES.items():
        records = _records_for(store, file_name)
        assert records, f"{file_name} produced no chunks"
        for record in records:
            assert record.file_type == file_type
            assert record.page == page  # slide/sheet → page; docx/html → None
            assert record.extraction == "text"  # digital text, never OCR
        stored_text = " ".join(record.content for record in records)
        assert planted_fact in stored_text, f"{file_name}: planted fact missing from chunks"


def test_xlsx_table_fact_survives_as_markdown_table(
    office_web_settings: Callable[..., None], office_web_corpus: Path
) -> None:
    """The spreadsheet's table fact is stored as a searchable GFM table row."""
    store = FakeStore()
    ingest_corpus(
        str(office_web_corpus), store=store, bm25=FakeBM25(), embeddings=FakeEmbeddings(), verbose=0
    )
    stored_text = "\n".join(
        record.content for record in _records_for(store, "quayside_inventory.xlsx")
    )
    bollard_rows = [
        line
        for line in stored_text.splitlines()
        if line.startswith("|") and "Mooring bollard" in line
    ]
    assert bollard_rows and any("148" in row for row in bollard_rows)


def test_malformed_office_file_is_contained(
    office_web_settings: Callable[..., None],
    office_web_corpus: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One corrupt .docx is counted failed; the rest of the run continues."""
    (office_web_corpus / "corrupt.docx").write_bytes(b"this is not a zip archive at all")
    store = FakeStore()
    with caplog.at_level(logging.ERROR):
        summary = ingest_corpus(
            str(office_web_corpus),
            store=store,
            bm25=FakeBM25(),
            embeddings=FakeEmbeddings(),
            verbose=0,
        )

    assert summary.discovered == 5
    assert summary.failed == 1
    assert summary.ingested == 4  # the four good fixtures still land
    assert any("corrupt.docx" in record.message for record in caplog.records)
    assert not _records_for(store, "corrupt.docx")
