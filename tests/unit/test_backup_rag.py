"""Unit tests for the RAG export and restore (ADR-019).

A fake store hands :func:`varagity.backup.rag.export_rag` a scripted corpus
snapshot (the real one is covered against pgvector in the integration
suite); the bundle it writes is a real zip in ``tmp_path``, read back
member by member. The restore half reads such bundles back into a fake
store with the real one's transaction semantics (rows land only when both
streams were consumed cleanly) — the real binary COPY, index rebuild and
advisory lock are the integration suite's.
"""

import hashlib
import json
import logging
import struct
import zipfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pgvector import Vector

from varagity.backup import bundle as bundle_module
from varagity.backup.bundle import (
    MANIFEST_NAME,
    BundleDamaged,
    BundleIncompatible,
    BundleInvalid,
    BundleManifest,
    list_bundles,
    read_manifest,
)
from varagity.backup.rag import (
    CHUNKS_MEMBER,
    DOCUMENTS_MEMBER,
    EMBEDDINGS_MEMBER,
    BM25RebuildFailed,
    NothingToExport,
    RagRestore,
    derive_corpus_stale,
    export_rag,
    ingest_drift,
    restore_rag,
    restored_source,
    vector_bytes,
    vector_from_bytes,
)
from varagity.stores.vector_store import (
    CorpusChunkRow,
    CorpusDocumentRow,
    CorpusIndexRow,
    CorpusLocked,
)

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
        assert [listing.name for listing in list_bundles(dest, "rag")] == [result.bundle.name]
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

    def test_rows_round_trip_through_the_restore_decoder(self) -> None:
        values = [1.0, -2.5, 3.4028234663852886e38, 1.401298464324817e-45]
        vector = Vector(values)
        assert vector_from_bytes(vector_bytes(vector, 4, "c"), 4).to_binary() == (
            vector.to_binary()
        )


# ── restore ───────────────────────────────────────────────────────────────


class FakeRestoreStore:
    """``replace_corpus`` with the real one's all-or-nothing semantics.

    Both streams are consumed (documents first, as the foreign key makes
    the real store do) before anything "commits", so a stream that raises
    leaves the previous rows exactly as they were.
    """

    def __init__(
        self,
        documents: Iterable[CorpusDocumentRow] = (),
        chunks: Iterable[CorpusChunkRow] = (),
        *,
        refuse_lock: bool = False,
    ) -> None:
        self.documents = list(documents)
        self.chunks = list(chunks)
        self.refuse_lock = refuse_lock
        self.log: list[str] = []

    @contextmanager
    def corpus_lock(self, *, exclusive: bool) -> Iterator[None]:
        if self.refuse_lock:
            raise CorpusLocked(exclusive=exclusive)
        self.log.append("lock exclusive" if exclusive else "lock shared")
        try:
            yield
        finally:
            self.log.append("unlock")

    def replace_corpus(
        self,
        documents: Iterable[CorpusDocumentRow],
        chunks: Iterable[CorpusChunkRow],
        *,
        on_indexing: Callable[[], None] | None = None,
    ) -> tuple[int, int]:
        self.log.append("begin")
        staged_documents = list(documents)
        staged_chunks = list(chunks)
        if on_indexing is not None:
            on_indexing()
        self.documents, self.chunks = staged_documents, staged_chunks
        self.log.append("commit")
        return len(staged_documents), len(staged_chunks)

    def iter_index_rows(self) -> Iterator[CorpusIndexRow]:
        for row in self.chunks:
            yield CorpusIndexRow(
                row.chunk_id,
                row.doc_id,
                row.original_index,
                row.content,
                row.contextualized_content,
            )


class FakeRebuildBM25:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.indexed: list[CorpusIndexRow] = []

    def rebuild(
        self, chunks: Iterable[CorpusIndexRow], *, on_indexed: Callable[[int], None] | None = None
    ) -> int:
        if self.fail_with is not None:
            raise self.fail_with
        self.indexed = []
        for chunk in chunks:
            self.indexed.append(chunk)
            if on_indexed is not None:
                on_indexed(len(self.indexed))
        return len(self.indexed)


def exported(docs_root: Path, store: FakeStore | None = None) -> Path:
    """Export a sample corpus and return the bundle's path."""
    return export_rag(
        docs_root / "varagity-exports", store or sample_store(docs_root), SETTINGS, stale=False
    ).bundle.path


def rezip(path: Path, replace: dict[str, bytes], *, reseal: bool) -> Path:
    """Rewrite a bundle with some members replaced.

    ``reseal`` records the new members' sizes and sha256s (a self-consistent
    bundle with odd contents); without it the original ``bundle.json``
    stays, so the bundle reads as damaged.
    """
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents.update(replace)
    if reseal:
        manifest = json.loads(contents[MANIFEST_NAME])
        for member in manifest["members"]:
            data = contents[member["name"]]
            member["size_bytes"] = len(data)
            member["sha256"] = hashlib.sha256(data).hexdigest()
        contents[MANIFEST_NAME] = json.dumps(manifest).encode()
    target = path.with_name(f"rezipped-{path.name}")
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in contents.items():
            if name != MANIFEST_NAME:
                archive.writestr(name, data)
        archive.writestr(MANIFEST_NAME, contents[MANIFEST_NAME])
    return target


class TestRestore:
    def test_a_bundle_round_trips_into_another_corpus_root(
        self, docs_root: Path, tmp_path: Path
    ) -> None:
        """★ Export under one DOCS_PATH, restore under another: identity kept, sources re-rooted."""
        source_store = sample_store(docs_root)
        bundle = exported(docs_root, source_store)
        elsewhere = tmp_path / "host-docs"
        store, bm25 = FakeRestoreStore(), FakeRebuildBM25()

        result = restore_rag(bundle, store, bm25, elsewhere)  # type: ignore[arg-type]

        by_id = {row.doc_id: row for row in store.documents}
        # Inside the exporting corpus: re-rooted under the restoring one.
        assert by_id["doc-a"].source == str((elsewhere / "q3" / "a.md").resolve())
        # From outside it: no relative path, so the absolute source is kept.
        assert by_id["doc-b"].source == "/elsewhere/b.md"
        assert by_id["doc-a"].content_hash == "0" * 64
        assert by_id["doc-a"].ingested_at == INGESTED
        assert by_id["doc-a"].n_chunks == 3

        assert [row.chunk_id for row in store.chunks] == [r.chunk_id for r in source_store.chunks]
        for restored, original in zip(store.chunks, source_store.chunks, strict=True):
            assert restored.embedding.to_binary() == original.embedding.to_binary()  # bit-exact
            assert restored.content == original.content
            assert restored.context == original.context
            assert restored.created_at == original.created_at
            # metadata travels whole, its source following the document's.
            assert restored.metadata["source"] == by_id[restored.doc_id].source
            assert restored.metadata["embedding_model"] == MODEL

        assert [chunk.chunk_id for chunk in bm25.indexed] == [r.chunk_id for r in store.chunks]
        assert store.log == ["lock exclusive", "begin", "commit", "unlock"]
        assert isinstance(result, RagRestore)
        assert result.counts == {"documents": 2, "chunks": 5, "bm25": 5}
        assert result.bundle_name == bundle.name
        assert result.size_bytes == bundle.stat().st_size
        assert result.stale is None  # the flow derives and records it

    def test_progress_walks_validate_load_index_bm25(self, docs_root: Path) -> None:
        frames: list[tuple[str, int | None, int | None]] = []
        restore_rag(
            exported(docs_root),
            FakeRestoreStore(),  # type: ignore[arg-type]
            FakeRebuildBM25(),  # type: ignore[arg-type]
            docs_root,
            on_progress=lambda stage, current, total: frames.append((stage, current, total)),
        )
        stages = [stage for stage, _, _ in frames]
        assert stages[0] == "validate"
        assert stages.index("load") < stages.index("index") < stages.index("bm25")
        loads = [(current, total) for stage, current, total in frames if stage == "load"]
        assert loads[0] == (0, 5) and loads[-1] == (5, 5)
        indexed = [(current, total) for stage, current, total in frames if stage == "bm25"]
        assert indexed[0] == (0, 5) and indexed[-1] == (5, 5)

    def test_a_damaged_member_leaves_the_previous_corpus_untouched(self, docs_root: Path) -> None:
        """★ Same length, valid zip, wrong bytes: refused before the commit."""
        bundle = exported(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            chunks = archive.read(CHUNKS_MEMBER)
        tampered = rezip(
            bundle, {CHUNKS_MEMBER: chunks.replace(b"chunk 1", b"chunk 9")}, reseal=False
        )
        previous = [document("old", str(docs_root / "old.md"), 1)]
        store = FakeRestoreStore(previous)
        bm25 = FakeRebuildBM25()
        with pytest.raises(BundleDamaged, match="chunks.jsonl does not match its recorded sha256"):
            restore_rag(tampered, store, bm25, docs_root)  # type: ignore[arg-type]
        assert store.documents == previous
        assert store.log == ["lock exclusive", "begin", "unlock"]  # never committed
        assert bm25.indexed == []

    def test_an_incompatible_bundle_is_refused_before_the_lock(
        self, docs_root: Path, settings_env: Callable[..., None]
    ) -> None:
        bundle = exported(docs_root)
        settings_env(EMBEDDING_MODEL="BAAI/bge-m3")
        store = FakeRestoreStore()
        with pytest.raises(BundleIncompatible, match="Embedded with test-model"):
            restore_rag(bundle, store, FakeRebuildBM25(), docs_root)  # type: ignore[arg-type]
        assert store.log == []

    def test_a_held_corpus_lock_refuses_the_restore(self, docs_root: Path) -> None:
        store = FakeRestoreStore([document("old", "/x.md", 0)], refuse_lock=True)
        bm25 = FakeRebuildBM25()
        with pytest.raises(CorpusLocked, match="an ingest is running in another process"):
            restore_rag(exported(docs_root), store, bm25, docs_root)  # type: ignore[arg-type]
        assert [row.doc_id for row in store.documents] == ["old"]
        assert bm25.indexed == []

    def test_a_bm25_failure_after_the_commit_says_the_rows_are_restored(
        self, docs_root: Path
    ) -> None:
        store = FakeRestoreStore()
        bm25 = FakeRebuildBM25(fail_with=ConnectionError("es went away"))
        with pytest.raises(BM25RebuildFailed, match="restore the same bundle again") as caught:
            restore_rag(exported(docs_root), store, bm25, docs_root)  # type: ignore[arg-type]
        assert "ConnectionError: es went away" in str(caught.value)
        assert len(store.chunks) == 5  # committed
        assert store.log[-1] == "unlock"

    def test_a_malformed_line_is_invalid_and_rolls_back(self, docs_root: Path) -> None:
        bundle = exported(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            lines = archive.read(CHUNKS_MEMBER).splitlines(keepends=True)
        lines[2] = b'{"chunk_id": 7}\n'
        broken = rezip(bundle, {CHUNKS_MEMBER: b"".join(lines)}, reseal=True)
        store = FakeRestoreStore()
        with pytest.raises(BundleInvalid, match="chunks.jsonl line 3 is not a valid row"):
            restore_rag(broken, store, FakeRebuildBM25(), docs_root)  # type: ignore[arg-type]
        assert "commit" not in store.log

    def test_a_chunk_of_no_document_is_invalid(self, docs_root: Path) -> None:
        bundle = exported(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            chunks = archive.read(CHUNKS_MEMBER)
        orphaned = rezip(
            bundle,
            {CHUNKS_MEMBER: chunks.replace(b'"doc_id":"doc-b"', b'"doc_id":"doc-z"')},
            reseal=True,
        )
        with pytest.raises(BundleInvalid, match="chunk doc-b::0 belongs to no document"):
            restore_rag(orphaned, FakeRestoreStore(), FakeRebuildBM25(), docs_root)  # type: ignore[arg-type]

    def test_more_lines_than_vectors_is_invalid(self, docs_root: Path) -> None:
        bundle = exported(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            chunks = archive.read(CHUNKS_MEMBER)
        longer = rezip(
            bundle, {CHUNKS_MEMBER: chunks + chunks.splitlines(keepends=True)[0]}, reseal=True
        )
        with pytest.raises(BundleInvalid, match="embeddings.f32 ends before chunks.jsonl line 6"):
            restore_rag(longer, FakeRestoreStore(), FakeRebuildBM25(), docs_root)  # type: ignore[arg-type]

    def test_fewer_lines_than_vectors_is_invalid(self, docs_root: Path) -> None:
        bundle = exported(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            chunks = archive.read(CHUNKS_MEMBER)
        shorter = rezip(
            bundle, {CHUNKS_MEMBER: b"".join(chunks.splitlines(keepends=True)[:4])}, reseal=True
        )
        with pytest.raises(BundleInvalid, match="more vectors than chunks.jsonl has lines"):
            restore_rag(shorter, FakeRestoreStore(), FakeRebuildBM25(), docs_root)  # type: ignore[arg-type]

    def test_a_document_count_that_disagrees_with_the_manifest_is_invalid(
        self, docs_root: Path
    ) -> None:
        bundle = exported(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            documents = archive.read(DOCUMENTS_MEMBER)
        doubled = rezip(bundle, {DOCUMENTS_MEMBER: documents + documents}, reseal=True)
        with pytest.raises(BundleInvalid, match="documents.jsonl holds 4 document"):
            restore_rag(doubled, FakeRestoreStore(), FakeRebuildBM25(), docs_root)  # type: ignore[arg-type]


class TestReRooting:
    def test_a_relative_path_lands_under_the_restoring_root(self, tmp_path: Path) -> None:
        root = tmp_path / "docs"
        assert restored_source("/app/docs/q3/a.md", "q3/a.md", root) == str(
            (root / "q3" / "a.md").resolve()
        )

    def test_a_document_from_outside_the_corpus_keeps_its_source(self, tmp_path: Path) -> None:
        assert restored_source("/elsewhere/b.md", None, tmp_path) == "/elsewhere/b.md"

    def test_resolution_follows_a_symlink_like_ingest_does(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        (real / "a.md").write_text("x")
        root = tmp_path / "docs"
        root.mkdir()
        (root / "q3").symlink_to(real)
        assert restored_source("/old/q3/a.md", "q3/a.md", root) == str(real / "a.md")

    @pytest.mark.parametrize("relative", ["../etc/passwd", "q3/../../x.md", "/abs/a.md", ""])
    def test_a_path_that_leaves_the_corpus_is_invalid(self, tmp_path: Path, relative: str) -> None:
        with pytest.raises(BundleInvalid, match="not a path inside the corpus"):
            restored_source("/x", relative, tmp_path)


def manifest_with(ingest_settings: dict[str, str] | None, *, stale: bool = False) -> BundleManifest:
    return BundleManifest(
        format="varagity-backup",
        version=1,
        corpus="rag",
        created_at=INGESTED,
        embedding_model=MODEL,
        embedding_dim=DIM,
        stale=stale,
        counts={"documents": 1, "chunks": 1},
        ingest_settings=ingest_settings,
        members=[],
    )


class TestStaleDerivation:
    def test_matching_settings_and_a_fresh_bundle_are_not_stale(self) -> None:
        manifest = manifest_with(dict(SETTINGS))
        assert ingest_drift(manifest, SETTINGS) == []
        assert derive_corpus_stale(manifest, SETTINGS) is False

    def test_a_bundle_stale_at_export_stays_stale(self) -> None:
        assert derive_corpus_stale(manifest_with(dict(SETTINGS), stale=True), SETTINGS) is True

    def test_differing_settings_are_drift_and_flag_stale(self) -> None:
        current = {**SETTINGS, "CHUNK_SIZE": "800", "CHUNKING_STRATEGY": "markdown_aware"}
        manifest = manifest_with(dict(SETTINGS))
        assert ingest_drift(manifest, current) == ["CHUNKING_STRATEGY", "CHUNK_SIZE"]
        assert derive_corpus_stale(manifest, current) is True

    def test_a_setting_on_only_one_side_is_drift(self) -> None:
        current = {**SETTINGS, "NEW_KNOB": "on"}
        assert ingest_drift(manifest_with(dict(SETTINGS)), current) == ["NEW_KNOB"]
        recorded = {**SETTINGS, "OLD_KNOB": "x"}
        assert ingest_drift(manifest_with(recorded), SETTINGS) == ["OLD_KNOB"]

    def test_a_bundle_without_recorded_settings_drifts_on_all_of_them(self) -> None:
        manifest = manifest_with(None)
        assert ingest_drift(manifest, SETTINGS) == sorted(SETTINGS)
        assert derive_corpus_stale(manifest, SETTINGS) is True


def test_the_verdicts_member_allowlist_is_the_bundles_members() -> None:
    """The allowlist is hardcoded in bundle.py (circular import) — pinned here."""
    assert bundle_module._CORPUS_MEMBERS["rag"] == {
        DOCUMENTS_MEMBER,
        CHUNKS_MEMBER,
        EMBEDDINGS_MEMBER,
    }
