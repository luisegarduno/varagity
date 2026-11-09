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
from varagity.stores.vector_store import (
    CORPUS_LOCK_KEY,
    ContextualVectorDB,
    CorpusChunkRow,
    CorpusDocumentRow,
    CorpusIndexRow,
    CorpusLocked,
    CorpusTablesBusy,
)

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
        self.copies: list[tuple[str, list[str], list[tuple[Any, ...]]]] = []
        # Every statement, COPY and transaction boundary, in order.
        self.events: list[str] = []
        self.fail_on: dict[str, Exception] = {}
        self.transactions = 0
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> FakeCursor:
        self.queries.append((sql, params))
        self.events.append(" ".join(sql.split()))
        for fragment, error in self.fail_on.items():
            if fragment in sql:
                raise error
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

            @contextmanager
            def copy(self, sql: str) -> Iterator[Any]:
                types: list[str] = []
                rows: list[tuple[Any, ...]] = []
                connection.events.append(sql.split(" (")[0])  # "COPY <table>"

                class _Copy:
                    def set_types(self, names: list[str]) -> None:
                        types.extend(names)

                    def write_row(self, row: tuple[Any, ...]) -> None:
                        rows.append(tuple(row))

                yield _Copy()
                connection.copies.append((sql, types, rows))

        return _Cursor()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions += 1
        self.events.append("BEGIN")
        try:
            yield
        except BaseException:
            self.events.append("ROLLBACK")
            raise
        self.events.append("COMMIT")

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


class TestCorpusLock:
    """The RAG corpus's advisory lock (ADR-019): try-only, paired with its unlock."""

    def test_an_ingest_takes_it_shared_and_gives_it_back(self) -> None:
        conn = FakeConnection([FakeCursor(row=(True,))])
        with store_with(conn).corpus_lock(exclusive=False):
            conn.events.append("work")
        assert conn.events == [
            "SELECT pg_try_advisory_lock_shared(%s)",
            "work",
            "SELECT pg_advisory_unlock_shared(%s)",
        ]
        assert all(params == (CORPUS_LOCK_KEY,) for _, params in conn.queries)

    def test_a_restore_takes_it_exclusive_and_releases_it_even_on_failure(self) -> None:
        conn = FakeConnection([FakeCursor(row=(True,))])
        with (
            pytest.raises(RuntimeError, match="boom"),
            store_with(conn).corpus_lock(exclusive=True),
        ):
            raise RuntimeError("boom")
        assert conn.events == ["SELECT pg_try_advisory_lock(%s)", "SELECT pg_advisory_unlock(%s)"]

    @pytest.mark.parametrize(
        ("exclusive", "message"),
        [
            (True, "an ingest is running in another process"),
            (False, "the corpus is being restored"),
        ],
    )
    def test_a_refused_try_raises_and_unlocks_nothing(self, exclusive: bool, message: str) -> None:
        conn = FakeConnection([FakeCursor(row=(False,))])
        with (
            pytest.raises(CorpusLocked, match=message) as caught,
            store_with(conn).corpus_lock(exclusive=exclusive),
        ):
            pytest.fail("the block must not run")  # pragma: no cover
        assert caught.value.exclusive is exclusive
        assert len(conn.events) == 1  # the try only: never unlock what we don't hold

    def test_an_unlock_on_a_dead_session_is_logged_not_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = FakeConnection([FakeCursor(row=(True,))])
        conn.fail_on["pg_advisory_unlock"] = psycopg.OperationalError(
            "server closed the connection"
        )
        with store_with(conn).corpus_lock(exclusive=True):
            pass
        assert "could not release the corpus lock" in caplog.text

    def test_the_key_is_one_fixed_bigint(self) -> None:
        assert int.from_bytes(b"VARAGITY", "big") == CORPUS_LOCK_KEY
        assert 0 < CORPUS_LOCK_KEY < 2**63

    def test_lock_free_tries_and_gives_back(self) -> None:
        conn = FakeConnection([FakeCursor(row=(True,))])
        assert store_with(conn).corpus_lock_free() is True
        assert conn.events == ["SELECT pg_try_advisory_lock(%s)", "SELECT pg_advisory_unlock(%s)"]
        held = FakeConnection([FakeCursor(row=(False,))])
        assert store_with(held).corpus_lock_free() is False
        assert held.events == ["SELECT pg_try_advisory_lock(%s)"]


def _document(doc_id: str) -> CorpusDocumentRow:
    return CorpusDocumentRow(doc_id, f"/abs/corpus/{doc_id}.md", "md", "deadbeef", 1, INGESTED_AT)


def _chunk(doc_id: str, index: int) -> CorpusChunkRow:
    return CorpusChunkRow(
        chunk_id=f"{doc_id}::{index}",
        doc_id=doc_id,
        original_index=index,
        chunk_index=index,
        content=f"c{index}",
        context=None,
        contextualized_content=f"c{index}",
        metadata={"source": f"/abs/corpus/{doc_id}.md"},
        created_at=INGESTED_AT,
        embedding=Vector([0.5, 0.25]),
    )


_HNSW_DEF = (
    "CREATE INDEX chunks_embedding_hnsw ON public.chunks USING hnsw "
    "(embedding vector_cosine_ops) WITH (m='24')"
)


class TestReplaceCorpus:
    """The restore's write, in one transaction (ADR-019)."""

    def test_statement_order(self) -> None:
        """★ Empty → drop the index → COPY both → rebuild from the saved definition → ANALYZE."""
        conn = FakeConnection([FakeCursor(), FakeCursor(), FakeCursor(row=(_HNSW_DEF,))])
        indexing: list[str] = []
        written = store_with(conn).replace_corpus(
            [_document("docaaa")],
            [_chunk("docaaa", 0), _chunk("docaaa", 1)],
            on_indexing=lambda: indexing.append("now"),
        )
        assert written == (1, 2)
        assert indexing == ["now"]
        assert conn.events == [
            "BEGIN",
            "SET LOCAL lock_timeout = '30s'",
            "TRUNCATE documents, chunks",
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = "
            "'chunks_embedding_hnsw'",
            "DROP INDEX chunks_embedding_hnsw",
            "COPY documents",
            "COPY chunks",
            _HNSW_DEF,  # the saved definition, tuning and all
            "ANALYZE documents",
            "ANALYZE chunks",
            "COMMIT",
        ]

    def test_rows_go_over_binary_copy_with_their_types(self) -> None:
        conn = FakeConnection([FakeCursor(), FakeCursor(), FakeCursor(row=(_HNSW_DEF,))])
        chunk = _chunk("docaaa", 0)
        store_with(conn).replace_corpus([_document("docaaa")], [chunk])
        (doc_sql, doc_types, doc_rows), (chunk_sql, chunk_types, chunk_rows) = conn.copies
        assert "FROM STDIN (FORMAT binary)" in doc_sql and "FROM STDIN (FORMAT binary)" in chunk_sql
        assert doc_types == ["text", "text", "text", "text", "int4", "timestamptz"]
        assert doc_rows == [("docaaa", "/abs/corpus/docaaa.md", "md", "deadbeef", 1, INGESTED_AT)]
        assert chunk_types[7] == "vector" and chunk_types[8] == "jsonb"
        (row,) = chunk_rows
        assert row[0] == "docaaa::0"
        assert row[7] is chunk.embedding  # the Vector itself: dumped binary, bit for bit
        assert row[8] == {"source": "/abs/corpus/docaaa.md"}

    def test_a_lost_index_is_rebuilt_from_the_schema_definition(self) -> None:
        conn = FakeConnection([FakeCursor(), FakeCursor(), FakeCursor(row=None)])
        store_with(conn).replace_corpus([], [])
        assert "DROP INDEX chunks_embedding_hnsw" not in conn.events
        assert (
            "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
            in conn.events
        )

    def test_a_stream_that_raises_rolls_the_whole_replace_back(self) -> None:
        conn = FakeConnection([FakeCursor(), FakeCursor(), FakeCursor(row=(_HNSW_DEF,))])

        def damaged() -> Iterator[CorpusChunkRow]:
            yield _chunk("docaaa", 0)
            raise ValueError("chunks.jsonl does not match its recorded sha256")

        with pytest.raises(ValueError, match="does not match"):
            store_with(conn).replace_corpus([_document("docaaa")], damaged())
        assert conn.events[-1] == "ROLLBACK"
        assert "COMMIT" not in conn.events
        assert not any(event.startswith("CREATE INDEX") for event in conn.events)

    def test_tables_held_past_the_timeout_refuse_cleanly(self) -> None:
        conn = FakeConnection()
        conn.fail_on["TRUNCATE"] = psycopg.errors.LockNotAvailable("lock timeout")
        with pytest.raises(CorpusTablesBusy, match="nothing was replaced"):
            store_with(conn).replace_corpus([_document("docaaa")], [])
        assert conn.events[-1] == "ROLLBACK"
        assert conn.copies == []


class TestIndexRows:
    def test_the_five_bm25_fields_stream_through_a_server_side_cursor(self) -> None:
        conn = FakeConnection()
        conn.streams = [[("d::0", "d", 0, "c0", "x c0"), ("d::1", "d", 1, "c1", "x c1")]]
        rows = list(store_with(conn).iter_index_rows())
        assert rows == [
            CorpusIndexRow("d::0", "d", 0, "c0", "x c0"),
            CorpusIndexRow("d::1", "d", 1, "c1", "x c1"),
        ]
        ((name, binary, itersize, sql),) = conn.server_cursors
        assert name and binary is False and itersize == 500
        assert "embedding" not in sql and "ORDER BY original_index" in sql
        assert conn.events[0] == "BEGIN" and conn.events[-1] == "COMMIT"
