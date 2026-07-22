"""Unit tests for the pgvector store's SQL shapes and row mapping.

Real Postgres round-trips live in the integration suite; here a scripted
fake connection (the ``test_conversation_store.py`` pattern) verifies each
method's SQL, parameter marshalling (``Vector``/``Json`` adapters), row →
record mapping, and the empty-input short circuits.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from pgvector import Vector
from psycopg.types.json import Json

from varagity.stores import vector_store
from varagity.stores.records import ChunkRecord
from varagity.stores.vector_store import ContextualVectorDB

INGESTED_AT = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)


class FakeCursor:
    def __init__(self, *, row: Any = None, rows: list[Any] | None = None, rowcount: int = 0):
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self) -> Any:
        return self._row

    def fetchall(self) -> list[Any]:
        return self._rows


class FakeConnection:
    """Queue of scripted cursors; records every statement and batch."""

    def __init__(self, results: list[FakeCursor] | None = None) -> None:
        self.results = list(results or [])
        self.queries: list[tuple[str, Any]] = []
        self.batches: list[tuple[str, list[Any]]] = []
        self.transactions = 0
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> FakeCursor:
        self.queries.append((sql, params))
        return self.results.pop(0) if self.results else FakeCursor()

    def cursor(self) -> Any:
        connection = self

        class _Cursor:
            def __enter__(self) -> "_Cursor":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def executemany(self, sql: str, params: Any) -> None:
                connection.batches.append((sql, list(params)))

        return _Cursor()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions += 1
        yield

    def close(self) -> None:
        self.closed = True


def store_with(conn: FakeConnection) -> ContextualVectorDB:
    store = ContextualVectorDB.__new__(ContextualVectorDB)
    store._conn = conn  # type: ignore[assignment]
    return store


def make_record(index: int) -> ChunkRecord:
    return ChunkRecord.create(
        doc_id="docaaa000000000a",
        original_index=index,
        chunk_index=index,
        source="/abs/corpus/a.md",
        file_name="a.md",
        file_type="md",
        page=None,
        content=f"chunk {index}",
        context=None,
        chunk_size=400,
        chunk_overlap=50,
        chunking_strategy="recursive_character",
        embedding_model="intfloat/multilingual-e5-large-instruct",
        content_hash="deadbeef",
    )


_CHUNK_ROW = (
    "docaaa000000000a::0",
    "docaaa000000000a",
    0,
    "chunk text",
    "the blurb",
    {"source": "/abs/corpus/a.md"},
)
_DOCUMENT_ROW = (
    "docaaa000000000a",
    "/abs/corpus/a.md",
    "md",
    "deadbeef",
    3,
    INGESTED_AT,
    {"text": 2, "ocr_fallback": 1},
)


class TestLifecycle:
    def test_init_connects_and_registers_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = FakeConnection()
        connects: list[tuple[str, bool]] = []
        registered: list[Any] = []
        monkeypatch.setattr(
            psycopg,
            "connect",
            lambda conninfo, autocommit: connects.append((conninfo, autocommit)) or conn,
        )
        monkeypatch.setattr(vector_store, "register_vector", registered.append)
        store = ContextualVectorDB("host=example dbname=x")
        assert connects == [("host=example dbname=x", True)]
        assert registered == [conn]
        store.close()
        assert conn.closed
        store.close()  # idempotent — a closed connection is not re-closed

    def test_context_manager_closes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = FakeConnection()
        monkeypatch.setattr(psycopg, "connect", lambda conninfo, autocommit: conn)
        monkeypatch.setattr(vector_store, "register_vector", lambda _conn: None)
        with ContextualVectorDB("host=example") as store:
            assert isinstance(store, ContextualVectorDB)
        assert conn.closed


class TestDocumentQueries:
    def test_document_exists(self) -> None:
        assert store_with(FakeConnection([FakeCursor(row=(1,))])).document_exists("d", "h")
        assert not store_with(FakeConnection([FakeCursor()])).document_exists("d", "h")

    def test_document_n_chunks(self) -> None:
        assert store_with(FakeConnection([FakeCursor(row=(5,))])).document_n_chunks("d") == 5
        assert store_with(FakeConnection([FakeCursor()])).document_n_chunks("d") is None

    def test_counts(self) -> None:
        assert store_with(FakeConnection([FakeCursor(row=(7,))])).document_count() == 7
        assert store_with(FakeConnection([FakeCursor()])).document_count() == 0
        assert store_with(FakeConnection([FakeCursor(row=(31,))])).chunk_count() == 31
        assert store_with(FakeConnection([FakeCursor()])).chunk_count() == 0

    def test_grouped_counts_for_the_corpus_gauges(self) -> None:
        conn = FakeConnection(
            [
                FakeCursor(rows=[("pdf", 3), ("md", 2)]),
                FakeCursor(rows=[("recursive_character", 40), ("unknown", 2)]),
            ]
        )
        store = store_with(conn)
        assert store.document_count_by_type() == {"pdf": 3, "md": 2}
        assert store.chunk_count_by_strategy() == {"recursive_character": 40, "unknown": 2}

    def test_list_documents_maps_rows(self) -> None:
        store = store_with(FakeConnection([FakeCursor(rows=[_DOCUMENT_ROW])]))
        (info,) = store.list_documents()
        assert info.doc_id == "docaaa000000000a"
        assert info.n_chunks == 3
        assert info.ingested_at == INGESTED_AT
        assert info.extraction_mix == {"text": 2, "ocr_fallback": 1}

    def test_get_document(self) -> None:
        conn = FakeConnection([FakeCursor(row=_DOCUMENT_ROW)])
        info = store_with(conn).get_document("docaaa000000000a")
        assert info is not None and info.file_type == "md"
        assert conn.queries[0][1] == {"doc_id": "docaaa000000000a"}
        assert store_with(FakeConnection([FakeCursor()])).get_document("nope") is None

    def test_delete_documents(self) -> None:
        conn = FakeConnection([FakeCursor(rowcount=2)])
        assert store_with(conn).delete_documents(["a", "b"]) == 2
        assert conn.queries[0][1] == (["a", "b"],)
        empty = FakeConnection()
        assert store_with(empty).delete_documents([]) == 0
        assert empty.queries == []  # the statement is skipped entirely

    def test_delete_document_wraps_the_bulk_path(self) -> None:
        conn = FakeConnection([FakeCursor(rowcount=1)])
        assert store_with(conn).delete_document("a") == 1
        assert conn.queries[0][1] == (["a"],)

    def test_next_original_index(self) -> None:
        assert store_with(FakeConnection([FakeCursor(row=(42,))])).next_original_index() == 42
        assert store_with(FakeConnection([FakeCursor()])).next_original_index() == 0


class TestWrites:
    def test_upsert_document_params(self) -> None:
        conn = FakeConnection()
        store_with(conn).upsert_document(
            doc_id="d", source="/abs/a.md", file_type="md", content_hash="h", n_chunks=3
        )
        sql, params = conn.queries[0]
        assert "INSERT INTO documents" in sql
        assert params["doc_id"] == "d" and params["n_chunks"] == 3

    def test_upsert_chunks_marshals_vector_and_json(self) -> None:
        conn = FakeConnection()
        records = [make_record(0), make_record(1)]
        store_with(conn).upsert_chunks(records, [[0.1, 0.2], [0.3, 0.4]])
        ((sql, params),) = conn.batches
        assert "INSERT INTO chunks" in sql
        assert len(params) == 2
        assert isinstance(params[0]["embedding"], Vector)
        assert isinstance(params[0]["metadata"], Json)
        assert params[1]["chunk_id"] == records[1].chunk_id

    def test_upsert_chunks_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="exactly one vector"):
            store_with(FakeConnection()).upsert_chunks([make_record(0)], [])

    def test_store_document_is_one_transaction(self) -> None:
        conn = FakeConnection()
        store_with(conn).store_document(
            doc_id="d",
            source="/abs/a.md",
            file_type="md",
            content_hash="h",
            records=[make_record(0)],
            embeddings=[[0.1]],
        )
        assert conn.transactions == 1
        assert "INSERT INTO documents" in conn.queries[0][0]
        assert len(conn.batches) == 1


class TestReads:
    def test_search_maps_rows_and_binds_the_query_vector(self) -> None:
        conn = FakeConnection([FakeCursor(rows=[(*_CHUNK_ROW, 0.87)])])
        (chunk,) = store_with(conn).search([0.1, 0.2], k=5, verbose=0)
        assert chunk.chunk_id == "docaaa000000000a::0"
        assert chunk.context == "the blurb"
        assert chunk.score == pytest.approx(0.87)
        params = conn.queries[0][1]
        assert isinstance(params["qvec"], Vector)
        assert params["k"] == 5

    def test_document_chunks_carry_zero_scores(self) -> None:
        conn = FakeConnection([FakeCursor(rows=[_CHUNK_ROW])])
        (chunk,) = store_with(conn).document_chunks("docaaa000000000a")
        assert chunk.score == 0.0
        assert conn.queries[0][1] == {"doc_id": "docaaa000000000a"}

    def test_fetch_by_identity_keys_the_result(self) -> None:
        conn = FakeConnection([FakeCursor(rows=[_CHUNK_ROW])])
        found = store_with(conn).fetch_by_identity([("docaaa000000000a", 0), ("missing", 9)])
        assert set(found) == {("docaaa000000000a", 0)}  # unknown keys are absent
        assert found[("docaaa000000000a", 0)].score == 0.0
        params = conn.queries[0][1]
        assert params["doc_ids"] == ["docaaa000000000a", "missing"]
        assert params["original_indexes"] == [0, 9]

    def test_fetch_by_identity_empty_short_circuits(self) -> None:
        conn = FakeConnection()
        assert store_with(conn).fetch_by_identity([]) == {}
        assert conn.queries == []
