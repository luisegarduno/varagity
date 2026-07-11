"""Integration tests for ContextualVectorDB against a real pgvector Postgres.

Runs the actual ``schema.sql`` in a throwaway ``pgvector/pgvector:pg16``
container (testcontainers), then exercises upsert, idempotency, the unique
``(doc_id, original_index)`` identity, and cosine search ordering.

Select with ``pytest -m integration`` (needs Docker).
"""

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

from varagity.stores.records import ChunkRecord, content_hash, derive_doc_id
from varagity.stores.vector_store import ContextualVectorDB

pytestmark = pytest.mark.integration

SCHEMA_PATH = Path(__file__).parents[2] / "varagity" / "stores" / "schema.sql"
DIM = 1024


@pytest.fixture(scope="session")
def pg_conninfo() -> Iterator[str]:
    """A pgvector Postgres with schema.sql applied, for the whole session."""
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        conninfo = psycopg.conninfo.make_conninfo(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(5432)),
            dbname=container.dbname,
            user=container.username,
            password=container.password,
        )
        with psycopg.connect(conninfo, autocommit=True) as conn:
            conn.execute(SCHEMA_PATH.read_text())
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
