"""Integration tests for ContextualVectorDB against a real pgvector Postgres.

Runs the actual ``schema.sql`` in a throwaway ``pgvector/pgvector:pg16``
container (via the shared :mod:`varagity.eval.containers` helpers), then
exercises upsert, idempotency, the unique ``(doc_id, original_index)``
identity, and cosine search ordering.

Select with ``pytest -m integration`` (needs Docker).
"""

from collections.abc import Iterator

import psycopg
import pytest

from varagity.eval.containers import ephemeral_postgres
from varagity.stores.records import ChunkRecord, content_hash, derive_doc_id
from varagity.stores.vector_store import ContextualVectorDB

pytestmark = pytest.mark.integration

DIM = 1024


@pytest.fixture(scope="session")
def pg_conninfo() -> Iterator[str]:
    """A pgvector Postgres with schema.sql applied, for the whole session."""
    with ephemeral_postgres() as conninfo:
        yield conninfo


@pytest.fixture
def store(pg_conninfo: str) -> Iterator[ContextualVectorDB]:
    """A store on clean tables (truncated per test)."""
    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        conn.execute("TRUNCATE documents CASCADE")
    with ContextualVectorDB(pg_conninfo) as db:
        yield db


def _unit_vector(hot: int, value: float = 1.0) -> list[float]:
    vector = [0.0] * DIM
    vector[hot] = value
    return vector


def _record(doc_id: str, chunk_index: int, original_index: int, content: str) -> ChunkRecord:
    return ChunkRecord.create(
        doc_id=doc_id,
        original_index=original_index,
        chunk_index=chunk_index,
        source=f"/abs/corpus/{doc_id}.md",
        file_name=f"{doc_id}.md",
        file_type="md",
        page=None,
        content=content,
        context=None,
        chunk_size=400,
        chunk_overlap=50,
        chunking_strategy="recursive_character",
        embedding_model="test-model",
        content_hash="hash-" + doc_id,
    )


def _seed_document(store: ContextualVectorDB, doc_id: str, embeddings: list[list[float]]) -> None:
    records = [
        _record(doc_id, i, original_index=i, content=f"{doc_id} chunk {i}")
        for i in range(len(embeddings))
    ]
    store.store_document(
        doc_id=doc_id,
        source=f"/abs/corpus/{doc_id}.md",
        file_type="md",
        content_hash="hash-" + doc_id,
        records=records,
        embeddings=embeddings,
    )


class TestSchema:
    def test_expected_indexes_exist(self, store: ContextualVectorDB, pg_conninfo: str) -> None:
        with psycopg.connect(pg_conninfo) as conn:
            rows = conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"
            ).fetchall()
        names = {row[0] for row in rows}
        assert {"chunks_embedding_hnsw", "chunks_doc_id_idx", "chunks_doc_orig_uidx"} <= names


class TestUpsertAndSearch:
    def test_search_returns_planted_order(self, store: ContextualVectorDB) -> None:
        # Three chunks along different axes; the query aligns with axis 0,
        # is 45° from axis 1, and orthogonal to axis 2.
        exact = _unit_vector(0)
        diagonal = [0.0] * DIM
        diagonal[0] = diagonal[1] = 0.7071
        orthogonal = _unit_vector(2)
        _seed_document(store, "docaaa000000000a", [exact, diagonal, orthogonal])

        results = store.search(_unit_vector(0), k=3, verbose=0)

        assert [r.chunk_id for r in results] == [
            "docaaa000000000a::0",
            "docaaa000000000a::1",
            "docaaa000000000a::2",
        ]
        assert results[0].score == pytest.approx(1.0, abs=1e-4)
        assert results[1].score == pytest.approx(0.7071, abs=1e-3)
        assert results[2].score == pytest.approx(0.0, abs=1e-4)
        assert results[0].score > results[1].score > results[2].score

    def test_k_limits_results(self, store: ContextualVectorDB) -> None:
        _seed_document(store, "docbbb000000000b", [_unit_vector(i) for i in range(5)])
        assert len(store.search(_unit_vector(0), k=2, verbose=0)) == 2

    def test_retrieved_metadata_round_trips(self, store: ContextualVectorDB) -> None:
        _seed_document(store, "docccc000000000c", [_unit_vector(0)])
        result = store.search(_unit_vector(0), k=1, verbose=0)[0]
        assert result.doc_id == "docccc000000000c"
        assert result.original_index == 0
        assert result.context is None
        assert result.content == "docccc000000000c chunk 0"
        metadata = result.metadata
        assert metadata["file_type"] == "md"
        assert metadata["chunking_strategy"] == "recursive_character"
        assert metadata["extraction"] == "text"
        assert metadata["n_tokens"] > 0

    def test_chunk_upsert_is_idempotent_on_chunk_id(self, store: ContextualVectorDB) -> None:
        _seed_document(store, "docddd000000000d", [_unit_vector(0)])
        # Re-store the same document (same chunk_id/original_index): updates, no duplicates.
        _seed_document(store, "docddd000000000d", [_unit_vector(1)])
        results = store.search(_unit_vector(1), k=10, verbose=0)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=1e-4)


class TestIdempotency:
    def test_document_exists_matches_hash(self, store: ContextualVectorDB) -> None:
        file_hash = content_hash(b"original bytes")
        doc_id = derive_doc_id("corpus/a.md", file_hash)
        assert store.document_exists(doc_id, file_hash) is False

        store.upsert_document(
            doc_id=doc_id,
            source="/abs/corpus/a.md",
            file_type="md",
            content_hash=file_hash,
            n_chunks=3,
        )
        assert store.document_exists(doc_id, file_hash) is True
        # A changed file yields a different hash → not "already ingested".
        assert store.document_exists(doc_id, content_hash(b"changed bytes")) is False

    def test_document_n_chunks(self, store: ContextualVectorDB) -> None:
        assert store.document_n_chunks("unknown-doc") is None
        store.upsert_document(
            doc_id="docempty0000000e",
            source="/abs/corpus/empty.txt",
            file_type="txt",
            content_hash="h",
            n_chunks=0,
        )
        assert store.document_n_chunks("docempty0000000e") == 0


class TestIdentity:
    def test_unique_doc_original_index_enforced(self, store: ContextualVectorDB) -> None:
        """Plan decision #9: the fusion identity is unique; collisions are ingest bugs."""
        _seed_document(store, "doceee000000000e", [_unit_vector(0)])
        colliding = _record("doceee000000000e", chunk_index=9, original_index=0, content="dup")
        with pytest.raises(psycopg.errors.UniqueViolation):
            store.upsert_chunks([colliding], [_unit_vector(1)])

    def test_next_original_index_watermark(self, store: ContextualVectorDB) -> None:
        assert store.next_original_index() == 0
        records = [
            _record("docfff000000000f", i, original_index=40 + i, content="c") for i in (0, 1)
        ]
        store.store_document(
            doc_id="docfff000000000f",
            source="/abs/corpus/f.md",
            file_type="md",
            content_hash="hash-f",
            records=records,
            embeddings=[_unit_vector(0), _unit_vector(1)],
        )
        assert store.next_original_index() == 42


class TestDeletion:
    def test_delete_document_cascades_to_chunks(
        self, store: ContextualVectorDB, pg_conninfo: str
    ) -> None:
        """`--reingest` backing: the documents row and all chunks go together."""
        _seed_document(store, "docdel000000000d", [_unit_vector(0), _unit_vector(1)])
        _seed_document(store, "dockeep00000000k", [_unit_vector(2)])

        assert store.delete_document("docdel000000000d") == 1

        assert store.document_n_chunks("docdel000000000d") is None
        with psycopg.connect(pg_conninfo) as conn:
            row = conn.execute(
                "SELECT count(*) FROM chunks WHERE doc_id = %s", ("docdel000000000d",)
            ).fetchone()
            assert row is not None and row[0] == 0
        # the other document is untouched
        assert store.document_n_chunks("dockeep00000000k") == 1

    def test_delete_unknown_document_is_a_noop(self, store: ContextualVectorDB) -> None:
        assert store.delete_document("doc0000000000nil") == 0


class TestFetchByIdentity:
    """Hydration for the bm25/hybrid retrievers (spec §11.4)."""

    def test_returns_full_rows_keyed_by_identity(self, store: ContextualVectorDB) -> None:
        _seed_document(store, "dochyd000000000h", [_unit_vector(0), _unit_vector(1)])
        _seed_document(store, "docoth000000000o", [_unit_vector(2)])

        found = store.fetch_by_identity([("dochyd000000000h", 1), ("docoth000000000o", 0)])

        assert set(found) == {("dochyd000000000h", 1), ("docoth000000000o", 0)}
        chunk = found[("dochyd000000000h", 1)]
        assert chunk.chunk_id == "dochyd000000000h::1"
        assert chunk.content == "dochyd000000000h chunk 1"
        assert chunk.metadata["file_type"] == "md"  # full metadata hydrated
        assert chunk.score == 0.0  # relevance belongs to the caller

    def test_unknown_keys_are_absent(self, store: ContextualVectorDB) -> None:
        _seed_document(store, "dochyd000000000h", [_unit_vector(0)])
        found = store.fetch_by_identity([("dochyd000000000h", 0), ("doc0000000000nil", 9)])
        assert set(found) == {("dochyd000000000h", 0)}

    def test_empty_input_short_circuits(self, store: ContextualVectorDB) -> None:
        assert store.fetch_by_identity([]) == {}


class TestDocumentChunks:
    """Reading-order chunk listing (the chunker sweep's fact scan, spec_v2 §7.4)."""

    def test_returns_chunks_in_reading_order(self, store: ContextualVectorDB) -> None:
        _seed_document(
            store, "docord000000000r", [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
        )
        chunks = store.document_chunks("docord000000000r")
        assert [c.chunk_id for c in chunks] == [f"docord000000000r::{i}" for i in range(3)]
        assert chunks[0].content == "docord000000000r chunk 0"
        assert chunks[0].metadata["file_type"] == "md"  # full metadata present
        assert all(c.score == 0.0 for c in chunks)

    def test_unknown_document_is_empty(self, store: ContextualVectorDB) -> None:
        assert store.document_chunks("doc0000000000nil") == []


class TestCorpusCounts:
    """The store-derived corpus gauges' queries (spec_v3 §6.1a).

    These back the Ingestion dashboard's size panels, so they are asserted
    against real SQL — the ``chunking_strategy`` grouping in particular
    reads a JSONB key rather than a column.
    """

    def _seed_typed(
        self,
        store: ContextualVectorDB,
        doc_id: str,
        *,
        file_type: str,
        chunking_strategy: str,
        n_chunks: int,
    ) -> None:
        records = [
            ChunkRecord.create(
                doc_id=doc_id,
                original_index=i,
                chunk_index=i,
                source=f"/abs/corpus/{doc_id}.{file_type}",
                file_name=f"{doc_id}.{file_type}",
                file_type=file_type,
                page=None,
                content=f"{doc_id} chunk {i}",
                context=None,
                chunk_size=400,
                chunk_overlap=50,
                chunking_strategy=chunking_strategy,
                embedding_model="test-model",
                content_hash="hash-" + doc_id,
            )
            for i in range(n_chunks)
        ]
        store.store_document(
            doc_id=doc_id,
            source=f"/abs/corpus/{doc_id}.{file_type}",
            file_type=file_type,
            content_hash="hash-" + doc_id,
            records=records,
            embeddings=[_unit_vector(i) for i in range(n_chunks)],
        )

    def test_empty_corpus_counts_zero(self, store: ContextualVectorDB) -> None:
        assert store.chunk_count() == 0
        assert store.document_count_by_type() == {}
        assert store.chunk_count_by_strategy() == {}

    def test_counts_group_by_type_and_strategy(self, store: ContextualVectorDB) -> None:
        self._seed_typed(
            store,
            "docmd0000000001a",
            file_type="md",
            chunking_strategy="markdown_aware",
            n_chunks=3,
        )
        self._seed_typed(
            store, "docmd0000000002b", file_type="md", chunking_strategy="semantic", n_chunks=2
        )
        self._seed_typed(
            store, "docpdf000000003c", file_type="pdf", chunking_strategy="semantic", n_chunks=4
        )

        assert store.document_count() == 3
        assert store.chunk_count() == 9
        assert store.document_count_by_type() == {"md": 2, "pdf": 1}
        assert store.chunk_count_by_strategy() == {"markdown_aware": 3, "semantic": 6}

    def test_chunks_without_a_strategy_group_as_unknown(
        self, store: ContextualVectorDB, pg_conninfo: str
    ) -> None:
        """A chunk predating the metadata field must not vanish into a null label."""
        self._seed_typed(
            store, "docold000000004d", file_type="md", chunking_strategy="semantic", n_chunks=2
        )
        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            conn.execute("UPDATE chunks SET metadata = metadata - 'chunking_strategy'")

        assert store.chunk_count_by_strategy() == {"unknown": 2}


class TestAtomicity:
    def test_store_document_rolls_back_on_mismatch(self, store: ContextualVectorDB) -> None:
        """A failed chunk write rolls back the documents row too.

        Otherwise a re-run's idempotency check would skip a half-ingested
        file instead of re-attempting it.
        """
        records = [_record("docggg000000000g", i, original_index=i, content="c") for i in (0, 1)]
        with pytest.raises(ValueError, match="every chunk needs exactly one vector"):
            store.store_document(
                doc_id="docggg000000000g",
                source="/abs/corpus/g.md",
                file_type="md",
                content_hash="hash-g",
                records=records,
                embeddings=[_unit_vector(0)],  # one short → raises inside the transaction
            )
        assert store.document_n_chunks("docggg000000000g") is None
        assert store.next_original_index() == 0  # no chunks landed either
