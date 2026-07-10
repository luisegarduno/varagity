"""Unit tests for the ingestion loader (skeleton invariants + guards)."""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from varagity.ingest import loader as loader_module
from varagity.ingest.loader import MIN_EXTRACTED_CHARS, ingest_corpus
from varagity.stores.records import ChunkRecord


class FakeStore:
    """In-memory stand-in for ContextualVectorDB."""

    def __init__(self, start_index: int = 0) -> None:
        self.start_index = start_index
        self.documents: dict[str, dict] = {}
        self.records: list[ChunkRecord] = []
        self.embeddings: list[list[float]] = []
        self.closed = False

    def next_original_index(self) -> int:
        return self.start_index

    def document_exists(self, doc_id: str, content_hash: str) -> bool:
        doc = self.documents.get(doc_id)
        return doc is not None and doc["content_hash"] == content_hash

    def document_n_chunks(self, doc_id: str) -> int | None:
        doc = self.documents.get(doc_id)
        return None if doc is None else doc["n_chunks"]

    def upsert_document(
        self, *, doc_id: str, source: str, file_type: str, content_hash: str, n_chunks: int
    ) -> None:
        self.documents[doc_id] = {
            "source": source,
            "file_type": file_type,
            "content_hash": content_hash,
            "n_chunks": n_chunks,
        }

    def upsert_chunks(self, records: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        self.records.extend(records)
        self.embeddings.extend(embeddings)

    def store_document(
        self,
        *,
        doc_id: str,
        source: str,
        file_type: str,
        content_hash: str,
        records: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        self.upsert_document(
            doc_id=doc_id,
            source=source,
            file_type=file_type,
            content_hash=content_hash,
            n_chunks=len(records),
        )
        self.upsert_chunks(records, embeddings)

    def close(self) -> None:
        self.closed = True


class FakeEmbeddings:
    """Deterministic stand-in for EmbeddingsClient (passage mode only)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_passages(self, texts: list[str], verbose: int | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.25, -0.25, 0.5] for _ in texts]


@pytest.fixture
def pinned_settings(settings_env: Callable[..., None]) -> None:
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        CHUNKING_STRATEGY="recursive_character",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        EMBEDDING_MODEL="test-model",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "aurora.md").write_text(
        "# Aurora\n\nThe Aurora station sits at 2,847 meters and is powered by a "
        "thorium micro-reactor nicknamed Lantern producing 4.2 megawatts at peak."
    )
    (root / "tidal.txt").write_text(
        "The Corvo Tidal Grid consists of 87 underwater turbines in three arrays "
        "named Alba, Bruma, and Cinza, delivering 310 megawatt-hours per day."
    )
    return root


def test_happy_path_skeleton_invariants(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore()
    embeddings = FakeEmbeddings()
    summary = ingest_corpus(str(corpus), store=store, embeddings=embeddings, verbose=0)

    assert summary.discovered == 2
    assert summary.ingested == 2
    assert summary.failed == summary.skipped == summary.no_text == summary.unsupported == 0
    assert summary.chunks == len(store.records) > 0

    for record in store.records:
        assert record.context is None  # contextualization lands in Phase 5
        assert record.contextualized_content == record.content
        assert record.chunk_id == f"{record.doc_id}::{record.chunk_index}"
        assert record.chunk_size == 400
        assert record.chunking_strategy == "recursive_character"
        assert record.embedding_model == "test-model"
        assert record.extraction == "text"

    # embed_passages received exactly the contextualized (== original) texts
    embedded = [text for call in embeddings.calls for text in call]
    assert embedded == [record.contextualized_content for record in store.records]

    # injected store is NOT closed by the loader (caller owns it)
    assert store.closed is False


def test_original_index_monotonic_from_store_watermark(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore(start_index=5)
    ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    indexes = [record.original_index for record in store.records]
    assert indexes == list(range(5, 5 + len(indexes)))
    # (doc_id, original_index) unique across the whole run
    assert len({(r.doc_id, r.original_index) for r in store.records}) == len(store.records)


def test_doc_id_is_relative_path_stable(pinned_settings: None, tmp_path: Path) -> None:
    """Same relative layout under two different roots → identical doc_ids."""
    doc_ids = []
    for root_name in ("first-root", "second-root"):
        root = tmp_path / root_name / "docs"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "a.md").write_text("Identical content, long enough to pass the guard. " * 3)
        store = FakeStore()
        ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)
        doc_ids.append(store.records[0].doc_id)
    assert doc_ids[0] == doc_ids[1]


def test_second_run_skips_unchanged_files(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore()
    first = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert first.ingested == 2

    second = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert second.ingested == 0
    assert second.skipped == 2
    assert second.chunks == 0
    assert len(store.records) == first.chunks  # nothing re-written


def test_changed_file_is_reingested(pinned_settings: None, corpus: Path) -> None:
    store = FakeStore()
    ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    (corpus / "aurora.md").write_text(
        "# Aurora, revised\n\nThe station now hosts 16 researchers and the reactor "
        "was upgraded to 4.5 megawatts of peak output for the coming decade."
    )
    summary = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert summary.ingested == 1  # the changed file
    assert summary.skipped == 1  # the unchanged one


def test_empty_extraction_guard_records_zero_chunk_document(
    pinned_settings: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "empty.txt").write_text("   \n\n  \t ")  # below MIN_EXTRACTED_CHARS
    store = FakeStore()
    with caplog.at_level(logging.WARNING):
        summary = ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)

    assert summary.no_text == 1
    assert summary.ingested == 0
    assert any("no extractable text" in r.message for r in caplog.records)
    # visibly "seen": a documents row with n_chunks = 0, but no chunks
    assert [doc["n_chunks"] for doc in store.documents.values()] == [0]
    assert store.records == []
    assert MIN_EXTRACTED_CHARS == 50


def test_known_empty_file_rewarns_without_reparsing(
    pinned_settings: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "empty.txt").write_text(" \n ")
    store = FakeStore()
    ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)

    # Second run: the parser must not run again for the known-empty file.
    real_get_parser = loader_module.get_parser
    extract_calls = []

    def spying_get_parser(name: str):
        parser = real_get_parser(name)

        class Spy:
            def extract(self, path, verbose=None):
                extract_calls.append(path)
                return parser.extract(path, verbose=verbose)

        return Spy()

    monkeypatch.setattr(loader_module, "get_parser", spying_get_parser)
    with caplog.at_level(logging.WARNING):
        summary = ingest_corpus(str(root), store=store, embeddings=FakeEmbeddings(), verbose=0)

    assert summary.no_text == 1  # re-warned, still surfaced in the summary
    assert extract_calls == []  # …but not re-parsed
    assert any("no extractable text" in r.message for r in caplog.records)


def test_pdfs_counted_unsupported_until_phase_7(
    pinned_settings: None, corpus: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (corpus / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    store = FakeStore()
    with caplog.at_level(logging.WARNING):
        summary = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert summary.discovered == 3
    assert summary.ingested == 2
    assert summary.unsupported == 1
    assert any("no parser registered" in r.message for r in caplog.records)


def test_one_bad_file_does_not_abort_the_run(
    pinned_settings: None, corpus: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (corpus / "broken.txt").write_bytes(b"\xff\xfe\x00 not utf-8 at all \xff" * 20)
    store = FakeStore()
    with caplog.at_level(logging.ERROR):
        summary = ingest_corpus(str(corpus), store=store, embeddings=FakeEmbeddings(), verbose=0)
    assert summary.failed == 1
    assert summary.ingested == 2  # the good files still landed
    assert any("failed to ingest" in r.message for r in caplog.records)


def test_invalid_verbose_raises(pinned_settings: None, corpus: Path) -> None:
    with pytest.raises(ValueError, match="verbose"):
        ingest_corpus(str(corpus), store=FakeStore(), embeddings=FakeEmbeddings(), verbose=3)
