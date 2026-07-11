"""Integration tests for the real OCR fallback (``-m integration``).

Real Docling conversions with real OCR engines — model download and CPU
inference are too slow/network-bound for the unit suite. Parameterized over
the installed engines; an engine whose binary/models are missing skips with
a notice. Stores/embeddings are the in-memory fakes: the backing services
under test here are the OCR engines, not the databases.
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.unit.test_loader import FakeBM25, FakeEmbeddings, FakeStore
from varagity.ingest.loader import ingest_corpus
from varagity.ingest.parsers import get_parser

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).parents[1] / "fixtures" / "corpus"
ENGINES = ("easyocr", "tesseract")


def _engine_available(engine: str) -> bool:
    """Whether an OCR engine can run here (easyocr is a package dependency)."""
    if engine == "tesseract":
        return shutil.which("tesseract") is not None
    return True


@pytest.fixture(params=ENGINES)
def ocr_engine(request: pytest.FixtureRequest, settings_env: Callable[..., None]) -> str:
    """Pin the OCR settings to one engine, skipping if it's unavailable."""
    engine: str = request.param
    if not _engine_available(engine):
        pytest.skip(f"OCR engine {engine!r} unavailable (binary/models missing)")
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        CHUNKING_STRATEGY="recursive_character",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        CONTEXTUALIZE="false",
        EMBEDDING_MODEL="test-model",
        PDF_OCR_FALLBACK="true",
        PDF_OCR_MIN_CHARS=50,
        PDF_OCR_TEXTLESS_PAGE_RATIO=0.2,
        PDF_OCR_FORCE_FULL_PAGE="false",
        OCR_ENGINE=engine,
        OCR_LANGUAGES="en",
    )
    return engine


def test_scanned_pdf_ingests_via_fallback_with_provenance(ocr_engine: str, tmp_path: Path) -> None:
    """The image-only fixture is recovered by OCR with fallback provenance.

    The planted fact must land in chunks carrying ``extraction="ocr_fallback"``.
    """
    root = tmp_path / "docs"
    root.mkdir()
    (root / "moorhen_dredging_memo.pdf").write_bytes(
        (CORPUS / "moorhen_dredging_memo.pdf").read_bytes()
    )
    store = FakeStore()
    summary = ingest_corpus(
        str(root), store=store, bm25=FakeBM25(), embeddings=FakeEmbeddings(), verbose=0
    )

    assert summary.ingested == 1
    assert summary.failed == summary.no_text == 0
    assert store.records, "the OCR'd document produced no chunks"
    for record in store.records:
        assert record.extraction == "ocr_fallback"
        assert record.file_type == "pdf"
        assert record.page == 1
    recovered = " ".join(record.content for record in store.records).upper()
    assert "MOORHEN" in recovered
    assert "NINE METERS" in recovered  # the planted fact survived OCR


def test_mixed_pdf_keeps_digital_text_and_ocrs_the_scanned_page(ocr_engine: str) -> None:
    """The mixed fixture keeps its digital page and OCRs the scanned one.

    Embedded text must be preserved verbatim (no full-page re-OCR) while the
    image-only page is recovered by the engine.
    """
    raw = get_parser("pdf").extract(CORPUS / "breakwater_survey.pdf", verbose=0)

    assert raw.source_meta["extraction"] == "ocr_fallback"
    assert raw.source_meta["page"] == 1
    # Digital page 1: exact embedded text, not an OCR re-reading of it.
    assert "It is 1,340 meters long, was completed in 2011" in raw.text
    # Scanned page 2: the dive note recovered by OCR.
    recovered = raw.text.upper()
    assert "FIFTY EIGHT" in recovered
    assert "ANCHOR BLOCKS" in recovered


def test_blank_pdf_ends_in_the_empty_extraction_guard(
    settings_env: Callable[..., None], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A blank PDF ends in the loader's empty-extraction guard.

    Pass 2 also comes up empty, so the loader records a 0-chunk document
    (warned, counted) instead of failing the run.
    """
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        CONTEXTUALIZE="false",
        PDF_OCR_FALLBACK="true",
        OCR_ENGINE="easyocr",
        OCR_LANGUAGES="en",
    )
    root = tmp_path / "docs"
    root.mkdir()
    (root / "blank_pages.pdf").write_bytes((CORPUS / "blank_pages.pdf").read_bytes())
    store = FakeStore()
    with caplog.at_level(logging.WARNING):
        summary = ingest_corpus(
            str(root), store=store, bm25=FakeBM25(), embeddings=FakeEmbeddings(), verbose=0
        )

    assert summary.no_text == 1
    assert summary.ingested == summary.failed == 0
    assert any("no extractable text" in record.message for record in caplog.records)
    assert [doc["n_chunks"] for doc in store.documents.values()] == [0]
    assert store.records == []
