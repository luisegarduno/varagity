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
        self.server_cursors: list[tuple[str, bool, int, str]] = []
        self.streams: list[list[Any]] = []
        self.transactions = 0
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> FakeCursor:
        self.queries.append((sql, params))
        return self.results.pop(0) if self.results else FakeCursor()

    def cursor(self, name: str = "", *, binary: bool = False) -> Any:
        connection = self

        class _Cursor:
            itersize = 100

            def __enter__(self) -> "_Cursor":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def executemany(self, sql: str, params: Any) -> None:
                connection.batches.append((sql, list(params)))

            def execute(self, sql: str) -> None:
                # A named (server-side) cursor: record its shape, then serve
                # the next scripted result set when iterated.
                connection.server_cursors.append((name, binary, self.itersize, sql))

            def __iter__(self) -> Iterator[Any]:
                return iter(connection.streams.pop(0))

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


class TestCorpusSnapshot:
    """The backup export's read (ADR-019): one REPEATABLE READ snapshot."""

    def test_one_read_only_repeatable_read_transaction_documents_then_chunks(self) -> None:
        conn = FakeConnection([FakeCursor(), FakeCursor(row=(2, 3))])
        vector = Vector([0.5, 0.25])
        document_rows = [
            ("docaaa000000000a", "/abs/corpus/a.md", "md", "deadbeef", 2, INGESTED_AT),
            ("docbbb000000000b", "/abs/corpus/b.md", "md", "cafebabe", 1, INGESTED_AT),
        ]
        chunk_rows = [
            (
                f"docaaa000000000a::{i}",
                "docaaa000000000a",
                i,
                i,
                f"c{i}",
                None,
                f"c{i}",
                {},
                INGESTED_AT,
                vector,
            )
            for i in range(3)
        ]
        conn.streams = [document_rows, chunk_rows]
        store = store_with(conn)

        with store.iter_corpus() as snapshot:
            assert (snapshot.document_count, snapshot.chunk_count) == (2, 3)
            documents = list(snapshot.documents())
            chunks = list(snapshot.chunks())

        assert conn.transactions == 1  # every read inside one transaction
        # The isolation level must be the transaction's first statement.
        assert conn.queries[0][0] == ("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        assert "count(*) FROM documents" in conn.queries[1][0]
        assert "count(*) FROM chunks" in conn.queries[1][0]
        (doc_cursor, chunk_cursor) = conn.server_cursors
        # Server-side (named), binary (vectors arrive bit-exact), batched.
        assert doc_cursor[0] and doc_cursor[1] is True and doc_cursor[2] == 500
        assert chunk_cursor[0] and chunk_cursor[1] is True and chunk_cursor[2] == 500
        assert "FROM documents" in doc_cursor[3] and "ORDER BY doc_id" in doc_cursor[3]
        assert "embedding" not in doc_cursor[3]
        assert "ORDER BY original_index, doc_id" in chunk_cursor[3]
        assert "embedding" in chunk_cursor[3]

        assert [row.doc_id for row in documents] == ["docaaa000000000a", "docbbb000000000b"]
        assert documents[1].content_hash == "cafebabe"
        assert documents[0].ingested_at == INGESTED_AT
        assert [row.chunk_id for row in chunks] == [f"docaaa000000000a::{i}" for i in range(3)]
        assert chunks[0].embedding is vector
        assert chunks[2].original_index == 2

    def test_an_empty_corpus_counts_zero(self) -> None:
        conn = FakeConnection([FakeCursor(), FakeCursor(row=None)])
        with store_with(conn).iter_corpus() as snapshot:
            assert (snapshot.document_count, snapshot.chunk_count) == (0, 0)
