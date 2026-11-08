"""Unit tests for the RAG export (ADR-019).

A fake store hands :func:`varagity.backup.rag.export_rag` a scripted corpus
snapshot (the real one is covered against pgvector in the integration
suite); the bundle it writes is a real zip in ``tmp_path``, read back
member by member.
"""

import json
import logging
import struct
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pgvector import Vector

from varagity.backup.bundle import list_bundles, read_manifest
from varagity.backup.rag import (
    CHUNKS_MEMBER,
    DOCUMENTS_MEMBER,
    EMBEDDINGS_MEMBER,
    NothingToExport,
    export_rag,
    vector_bytes,
)
from varagity.stores.vector_store import CorpusChunkRow, CorpusDocumentRow

DIM = 4
MODEL = "test-model"
INGESTED = datetime(2026, 11, 1, 9, 30, tzinfo=UTC)
SETTINGS = {
    "CHUNKING_STRATEGY": "recursive_character",
    "CHUNK_OVERLAP": "50",
    "CHUNK_SIZE": "400",
    "CONTEXTUALIZE": "true",
    "OCR_ENGINE": "easyocr",
}


class FakeSnapshot:
    def __init__(
        self, documents: list[CorpusDocumentRow], chunks: list[CorpusChunkRow], log: list[str]
    ) -> None:
        self._documents = documents
        self._chunks = chunks
        self._log = log
        self.document_count = len(documents)
        self.chunk_count = len(chunks)

    def documents(self) -> Iterator[CorpusDocumentRow]:
        self._log.append("documents")
        yield from self._documents

    def chunks(self) -> Iterator[CorpusChunkRow]:
        self._log.append("chunks")
        yield from self._chunks


class FakeStore:
    """``iter_corpus`` over scripted rows, recording how the export drove it."""

    def __init__(self, documents: list[CorpusDocumentRow], chunks: list[CorpusChunkRow]) -> None:
        self.documents = documents
        self.chunks = chunks
        self.log: list[str] = []

    @contextmanager
    def iter_corpus(self) -> Iterator[FakeSnapshot]:
        self.log.append("enter")
        try:
            yield FakeSnapshot(self.documents, self.chunks, self.log)
        finally:
            self.log.append("exit")


def document(doc_id: str, source: str, n_chunks: int = 2) -> CorpusDocumentRow:
    return CorpusDocumentRow(
        doc_id=doc_id,
        source=source,
        file_type=Path(source).suffix.lstrip("."),
        content_hash="0" * 64,
        n_chunks=n_chunks,
        ingested_at=INGESTED,
    )


def chunk(
    doc_id: str, index: int, original_index: int, vector: list[float], model: str = MODEL
) -> CorpusChunkRow:
    return CorpusChunkRow(
        chunk_id=f"{doc_id}::{index}",
        doc_id=doc_id,
        original_index=original_index,
        chunk_index=index,
        content=f"chunk {index} of {doc_id} — ünïcode",
        context=None if index % 2 else f"blurb {index}",
        contextualized_content=f"blurb {index}\n\nchunk {index}",
        metadata={"embedding_model": model, "source": f"/corpus/{doc_id}.md", "page": None},
        created_at=INGESTED,
        embedding=Vector(vector),
    )


@pytest.fixture
def docs_root(tmp_path: Path, settings_env: Callable[..., None]) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    settings_env(DOCS_PATH=str(root), EMBEDDING_MODEL=MODEL, EMBEDDING_DIM=DIM)
    return root


def sample_store(docs_root: Path) -> FakeStore:
    vectors = [[float(i), -i / 3, 1e-38 * i, 3.4e38 / (i + 1)] for i in range(5)]
    return FakeStore(
        documents=[
            document("doc-a", str(docs_root / "q3" / "a.md"), n_chunks=3),
            document("doc-b", "/elsewhere/b.md", n_chunks=2),
        ],
        chunks=[
            chunk("doc-a", 0, 0, vectors[0]),
            chunk("doc-a", 1, 1, vectors[1]),
            chunk("doc-a", 2, 2, vectors[2]),
            chunk("doc-b", 0, 3, vectors[3]),
            chunk("doc-b", 1, 4, vectors[4]),
        ],
    )


def read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode().splitlines()]


class TestExport:
    def test_writes_a_bundle_into_the_exports_folder(self, docs_root: Path) -> None:
        dest = docs_root / "varagity-exports"
        result = export_rag(dest, sample_store(docs_root), SETTINGS, stale=False)
        assert result.bundle.path.parent == dest
        assert result.bundle.name.startswith("varagity-rag-")
        assert [listing.name for listing in list_bundles(dest)] == [result.bundle.name]
        assert result.counts == {"documents": 2, "chunks": 5}
        assert result.wall_clock_s >= 0

    def test_the_manifest_records_what_the_restore_will_check(self, docs_root: Path) -> None:
        dest = docs_root / "varagity-exports"
        result = export_rag(dest, sample_store(docs_root), SETTINGS, stale=True)
        manifest = read_manifest(result.bundle.path)
        assert manifest.corpus == "rag"
        assert manifest.embedding_model == MODEL
        assert manifest.embedding_dim == DIM
        assert manifest.stale is True
        assert manifest.ingest_settings == SETTINGS
        assert manifest.counts == {"documents": 2, "chunks": 5}
        assert [member.name for member in manifest.members] == [
            DOCUMENTS_MEMBER,
            CHUNKS_MEMBER,
            EMBEDDINGS_MEMBER,
        ]

    def test_chunks_and_embeddings_stay_row_aligned(self, docs_root: Path) -> None:
        """★ Row i of embeddings.f32 is the vector of chunks.jsonl line i, bit-exact."""
        store = sample_store(docs_root)
        result = export_rag(docs_root / "varagity-exports", store, SETTINGS, stale=False)
        members = read_members(result.bundle.path)
        lines = jsonl(members[CHUNKS_MEMBER])
        matrix = members[EMBEDDINGS_MEMBER]
        assert len(matrix) == len(lines) * DIM * 4
        for index, line in enumerate(lines):
            row = matrix[index * DIM * 4 : (index + 1) * DIM * 4]
            source = store.chunks[index]
            assert line["chunk_id"] == source.chunk_id
            expected = struct.pack(f"<{DIM}f", *source.embedding.to_list())
            assert row == expected, f"row {index} drifted from its chunk line"

    def test_chunk_lines_carry_every_column_but_the_vector(self, docs_root: Path) -> None:
        store = sample_store(docs_root)
        result = export_rag(docs_root / "varagity-exports", store, SETTINGS, stale=False)
        first = jsonl(read_members(result.bundle.path)[CHUNKS_MEMBER])[0]
        assert first == {
            "chunk_id": "doc-a::0",
            "doc_id": "doc-a",
            "original_index": 0,
            "chunk_index": 0,
            "content": "chunk 0 of doc-a — ünïcode",
            "context": "blurb 0",
            "contextualized_content": "blurb 0\n\nchunk 0",
            "metadata": {"embedding_model": MODEL, "source": "/corpus/doc-a.md", "page": None},
            "created_at": INGESTED.isoformat(),
        }

    def test_documents_carry_their_path_relative_to_the_corpus(self, docs_root: Path) -> None:
        result = export_rag(
            docs_root / "varagity-exports", sample_store(docs_root), SETTINGS, stale=False
        )
        documents = jsonl(read_members(result.bundle.path)[DOCUMENTS_MEMBER])
        by_id = {entry["doc_id"]: entry for entry in documents}
        assert by_id["doc-a"]["relative_path"] == "q3/a.md"
        assert by_id["doc-a"]["source"] == str(docs_root / "q3" / "a.md")
        # Ingested from outside DOCS_PATH: no relative path, the absolute
        # source is kept as the row's only provenance.
        assert by_id["doc-b"]["relative_path"] is None
        assert by_id["doc-b"]["source"] == "/elsewhere/b.md"
        assert by_id["doc-a"]["ingested_at"] == INGESTED.isoformat()
        assert by_id["doc-a"]["n_chunks"] == 3

    def test_everything_comes_from_one_snapshot_in_its_order(self, docs_root: Path) -> None:
        store = sample_store(docs_root)
        store.chunks.reverse()  # whatever order the snapshot yields is the bundle's
        result = export_rag(docs_root / "varagity-exports", store, SETTINGS, stale=False)
        assert store.log == ["enter", "documents", "chunks", "exit"]
        lines = jsonl(read_members(result.bundle.path)[CHUNKS_MEMBER])
        assert [line["chunk_id"] for line in lines] == [row.chunk_id for row in store.chunks]

    def test_progress_walks_the_stages_in_order(self, docs_root: Path) -> None:
        frames: list[tuple[str, int | None, int | None]] = []
        export_rag(
            docs_root / "varagity-exports",
            sample_store(docs_root),
            SETTINGS,
            stale=False,
            on_progress=lambda stage, current, total: frames.append((stage, current, total)),
        )
        stages = [stage for stage, _, _ in frames]
        assert stages[0] == "snapshot" and stages[-1] == "finalize"
        assert stages.index("documents") < stages.index("chunks") < stages.index("finalize")
        documents = [(current, total) for stage, current, total in frames if stage == "documents"]
        chunks = [(current, total) for stage, current, total in frames if stage == "chunks"]
        assert documents[0] == (0, 2) and documents[-1] == (2, 2)
        assert chunks[0] == (0, 5) and chunks[-1] == (5, 5)

    def test_progress_is_throttled_on_a_large_corpus(self, docs_root: Path) -> None:
        many = [chunk("doc-a", i, i, [0.0] * DIM) for i in range(1000)]
        store = FakeStore([document("doc-a", str(docs_root / "a.md"), 1000)], many)
        frames: list[str] = []
        export_rag(
            docs_root / "varagity-exports",
            store,
            SETTINGS,
            stale=False,
            on_progress=lambda stage, current, total: frames.append(stage),
        )
        assert 2 < frames.count("chunks") <= 52  # ~50 frames, not one per row


class TestRefusals:
    def test_an_empty_corpus_is_nothing_to_export(self, docs_root: Path) -> None:
        dest = docs_root / "varagity-exports"
        with pytest.raises(NothingToExport):
            export_rag(dest, FakeStore([], []), SETTINGS, stale=False)
        assert not dest.exists() or list(dest.iterdir()) == []

    def test_a_wrong_dimension_fails_and_leaves_no_bundle(self, docs_root: Path) -> None:
        dest = docs_root / "varagity-exports"
        store = FakeStore(
            [document("doc-a", str(docs_root / "a.md"), 1)], [chunk("doc-a", 0, 0, [1.0, 2.0])]
        )
        with pytest.raises(ValueError, match="2-dimensional embedding, but EMBEDDING_DIM is 4"):
            export_rag(dest, store, SETTINGS, stale=False)
        assert list(dest.iterdir()) == []

    def test_rows_from_another_embedding_model_are_flagged(
        self, docs_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = FakeStore(
            [document("doc-a", str(docs_root / "a.md"), 2)],
            [
                chunk("doc-a", 0, 0, [0.0] * DIM),
                chunk("doc-a", 1, 1, [0.0] * DIM, model="BAAI/bge-m3"),
            ],
        )
        with caplog.at_level(logging.WARNING, logger="varagity.backup.rag"):
            export_rag(docs_root / "varagity-exports", store, SETTINGS, stale=False)
        warning = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert "BAAI/bge-m3 ×1" in warning
        assert f"the bundle declares {MODEL}" in warning


class TestVectorBytes:
    def test_rows_are_little_endian_float32_and_bit_exact(self) -> None:
        values = [1.0, -2.5, 3.4028234663852886e38, 1.401298464324817e-45]
        assert vector_bytes(Vector(values), 4, "c") == struct.pack("<4f", *values)

    def test_a_mismatched_dimension_names_the_chunk(self) -> None:
        with pytest.raises(ValueError, match="chunk doc::7 has a 3-dimensional"):
            vector_bytes(Vector([1.0, 2.0, 3.0]), 4, "doc::7")
