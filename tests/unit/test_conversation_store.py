"""Unit tests for the conversation store's pure logic.

The SQL round-trips run against real Postgres in the integration suite;
here a scripted fake connection covers the snapshot builder, the
auto-title behavior (LLM cleanup, fallback, only-default-title guard),
and each CRUD method's SQL shape, parameter marshalling, and row mapping.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from psycopg.types.json import Json

from varagity.stores.conversation_store import (
    DEFAULT_TITLE,
    ConversationStore,
    _generate_title,
    _source_snapshot,
)
from varagity.stores.records import RetrievalTrace, RetrievedChunk

CREATED_AT = datetime(2026, 7, 21, 9, 0, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 7, 21, 9, 5, 0, tzinfo=UTC)


def make_chunk(*, with_trace: bool = True) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc::3",
        doc_id="doc",
        original_index=3,
        content="chunk text",
        context="the blurb",
        metadata={
            "source": "/docs/report.pdf",
            "file_name": "report.pdf",
            "file_type": "pdf",
            "page": 4,
            "extraction": "ocr_fallback",
            "file_created_at": "2024-01-02T08:00:00Z",
            "file_modified_at": "2024-05-04T12:30:45Z",
        },
        score=0.87,
        trace=(
            RetrievalTrace(
                semantic_rank=1,
                semantic_score=0.91,
                bm25_rank=3,
                bm25_score=7.2,
                fused_score=0.05,
                fused_rank=1,
                rerank_score=0.98,
                rerank_delta=0,
                final_rank=1,
            )
            if with_trace
            else None
        ),
    )


class TestSourceSnapshot:
    def test_snapshot_carries_everything_the_panel_shows(self) -> None:
        snapshot = _source_snapshot(make_chunk())
        assert snapshot["score"] == 0.87
        assert snapshot["content"] == "chunk text"
        assert snapshot["context"] == "the blurb"
        assert snapshot["source"] == "/docs/report.pdf"
        assert snapshot["file_name"] == "report.pdf"
        assert snapshot["file_type"] == "pdf"
        assert snapshot["page"] == 4
        assert snapshot["extraction"] == "ocr_fallback"
        assert snapshot["file_created_at"] == "2024-01-02T08:00:00Z"
        assert snapshot["file_modified_at"] == "2024-05-04T12:30:45Z"
        assert snapshot["trace"]["semantic_rank"] == 1
        assert snapshot["trace"]["rerank_score"] == 0.98

    def test_snapshot_without_trace_is_null(self) -> None:
        assert _source_snapshot(make_chunk(with_trace=False))["trace"] is None

    def test_snapshot_tolerates_pre_timestamp_metadata(self) -> None:
        """Chunks ingested before the fields existed snapshot as None, not KeyError."""
        chunk = make_chunk()
        del chunk.metadata["file_created_at"], chunk.metadata["file_modified_at"]
        snapshot = _source_snapshot(chunk)
        assert snapshot["file_created_at"] is None
        assert snapshot["file_modified_at"] is None


class ScriptedLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls = 0

    def generate(self, messages: Any, **kwargs: Any) -> str:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestGenerateTitle:
    def test_llm_one_liner_cleaned(self) -> None:
        llm = ScriptedLLM('<think>hmm a title</think>\n"Kelp Corridor Length"\n')
        assert _generate_title("How long is it?", llm=llm) == "Kelp Corridor Length"

    def test_multiline_title_collapsed(self) -> None:
        llm = ScriptedLLM("Kelp\nCorridor   Facts")
        assert _generate_title("q", llm=llm) == "Kelp Corridor Facts"

    def test_llm_failure_falls_back_to_question(self) -> None:
        llm = ScriptedLLM(RuntimeError("llm down"))
        assert _generate_title("  What is the kelp corridor?  ", llm=llm) == (
            "What is the kelp corridor?"
        )

    def test_empty_llm_title_falls_back(self) -> None:
        llm = ScriptedLLM("<think>only reasoning</think>")
        assert _generate_title("The question", llm=llm) == "The question"

    def test_long_results_truncated(self) -> None:
        llm = ScriptedLLM("x" * 500)
        assert len(_generate_title("q", llm=llm)) == 80


class FakeTitleConnection:
    """Scripts the SELECT title row; records UPDATEs."""

    def __init__(self, title: str | None) -> None:
        self.title = title
        self.updates: list[Any] = []
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> Any:
        if sql.startswith("SELECT title"):
            return _Cursor(None if self.title is None else (self.title,))
        if sql.startswith("UPDATE conversations SET title"):
            self.updates.append(params)
            return _Cursor(None)
        raise AssertionError(f"unexpected SQL: {sql}")


class _Cursor:
    def __init__(self, row: Any) -> None:
        self.row = row

    def fetchone(self) -> Any:
        return self.row


class FakeTurnsConnection:
    """Scripts the newest-first recent-turns SELECT; records the queries."""

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> Any:
        self.queries.append((sql, params))
        return _RowsCursor(self.rows)


class _RowsCursor:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def fetchall(self) -> list[Any]:
        return self.rows


def store_with(conn: Any) -> ConversationStore:
    store = ConversationStore.__new__(ConversationStore)
    store._conn = conn  # type: ignore[assignment]
    return store


class TestRecentTurns:
    """The chat engine's history feed (spec_v3 §4.4).

    SQL round-trips are integration-tested; here the reversal, the SQL
    shape, and the non-positive-limit guard.
    """

    def test_newest_first_rows_come_back_oldest_first(self) -> None:
        conn = FakeTurnsConnection([("assistant", "a2"), ("user", "q2"), ("assistant", "a1")])
        assert store_with(conn).recent_turns("c1", limit=3) == [
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
        ]

    def test_bounds_in_sql_with_no_sources_join(self) -> None:
        """Decision #8: LIMIT belongs in SQL, and evidence is never hydrated."""
        conn = FakeTurnsConnection([])
        store_with(conn).recent_turns("c1", limit=6)
        sql, params = conn.queries[0]
        assert "LIMIT" in sql
        assert "DESC" in sql  # newest-first, so LIMIT keeps the newest
        assert "message_sources" not in sql  # no snapshot/trace hydration
        assert "trace" not in sql
        assert params == ("c1", 6)

    def test_non_positive_limit_short_circuits(self) -> None:
        conn = FakeTurnsConnection([("user", "q")])
        assert store_with(conn).recent_turns("c1", limit=0) == []
        assert store_with(conn).recent_turns("c1", limit=-1) == []
        assert conn.queries == []  # the database is never touched


class ScriptedCursor:
    """One scripted result: fetchone/fetchall/iteration over the same rows."""

    def __init__(self, *, row: Any = None, rows: list[Any] | None = None, rowcount: int = 0):
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self) -> Any:
        return self._row

    def fetchall(self) -> list[Any]:
        return self._rows

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows)


class ScriptedConnection:
    """Queue of scripted cursors; records every statement and transaction."""

    def __init__(self, results: list[ScriptedCursor] | None = None) -> None:
        self.results = list(results or [])
        self.queries: list[tuple[str, Any]] = []
        self.transactions = 0
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> ScriptedCursor:
        self.queries.append((sql, params))
        return self.results.pop(0) if self.results else ScriptedCursor()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions += 1
        yield

    def close(self) -> None:
        self.closed = True


class TestLifecycle:
    def test_init_connects_and_close_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = ScriptedConnection()
        connects: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            psycopg,
            "connect",
            lambda conninfo, autocommit: connects.append((conninfo, autocommit)) or conn,
        )
        store = ConversationStore("host=example dbname=x")
        assert connects == [("host=example dbname=x", True)]
        store.close()
        assert conn.closed
        store.close()  # already closed — not re-closed


class TestConversationCrud:
    def test_create_conversation_defaults_the_title(self) -> None:
        conn = ScriptedConnection([ScriptedCursor(row=(CREATED_AT, UPDATED_AT))])
        summary = store_with(conn).create_conversation()
        assert summary.title == DEFAULT_TITLE
        assert summary.message_count == 0
        assert summary.created_at == CREATED_AT
        assert len(summary.conversation_id) == 32  # uuid4 hex
        assert conn.queries[0][1] == (summary.conversation_id, DEFAULT_TITLE)

    def test_create_conversation_with_explicit_title(self) -> None:
        conn = ScriptedConnection([ScriptedCursor(row=(CREATED_AT, UPDATED_AT))])
        assert store_with(conn).create_conversation("Kelp").title == "Kelp"

    def test_conversation_exists(self) -> None:
        assert store_with(ScriptedConnection([ScriptedCursor(row=(1,))])).conversation_exists("c")
        assert not store_with(ScriptedConnection([ScriptedCursor()])).conversation_exists("c")

    def test_list_conversations_maps_rows(self) -> None:
        conn = ScriptedConnection(
            [
                ScriptedCursor(
                    rows=[
                        ("c1", "Kelp", CREATED_AT, UPDATED_AT, "g1", 4),
                        ("c2", "Loose", CREATED_AT, UPDATED_AT, None, 0),
                    ]
                )
            ]
        )
        grouped, ungrouped = store_with(conn).list_conversations()
        assert grouped.conversation_id == "c1"
        assert grouped.title == "Kelp"
        assert grouped.message_count == 4
        assert grouped.group_id == "g1"
        assert ungrouped.group_id is None

    def test_delete_conversation_reports_rowcount(self) -> None:
        conn = ScriptedConnection([ScriptedCursor(rowcount=1)])
        assert store_with(conn).delete_conversation("c1") == 1
        assert conn.queries[0][1] == ("c1",)


class TestGroupCrud:
    def test_create_group_returns_the_row(self) -> None:
        conn = ScriptedConnection([ScriptedCursor(row=(CREATED_AT,))])
        created = store_with(conn).create_group("Research")
        assert created.name == "Research"
        assert created.created_at == CREATED_AT
        assert len(created.group_id) == 32  # uuid4 hex
        sql, params = conn.queries[0]
        assert "INSERT INTO conversation_groups" in sql
        assert params == (created.group_id, "Research")

    def test_list_groups_maps_rows_in_name_order(self) -> None:
        conn = ScriptedConnection(
            [ScriptedCursor(rows=[("g1", "Alpha", CREATED_AT), ("g2", "beta", UPDATED_AT)])]
        )
        groups = store_with(conn).list_groups()
        assert [(g.group_id, g.name) for g in groups] == [("g1", "Alpha"), ("g2", "beta")]
        assert "ORDER BY lower(name)" in conn.queries[0][0]  # case-folded folder order

    def test_group_exists(self) -> None:
        assert store_with(ScriptedConnection([ScriptedCursor(row=(1,))])).group_exists("g")
        assert not store_with(ScriptedConnection([ScriptedCursor()])).group_exists("g")

    def test_delete_group_reports_rowcount(self) -> None:
        conn = ScriptedConnection([ScriptedCursor(rowcount=1)])
        assert store_with(conn).delete_group("g1") == 1
        sql, params = conn.queries[0]
        assert "DELETE FROM conversation_groups" in sql
        assert params == ("g1",)

    def test_set_conversation_group_files_and_ungroups(self) -> None:
        conn = ScriptedConnection([ScriptedCursor(rowcount=1), ScriptedCursor(rowcount=1)])
        store = store_with(conn)
        assert store.set_conversation_group("c1", "g1") == 1
        assert store.set_conversation_group("c1", None) == 1
        assert conn.queries[0][1] == ("g1", "c1")
        assert conn.queries[1][1] == (None, "c1")
        # A move must never bump updated_at — it would re-order the sidebar.
        assert "updated_at" not in conn.queries[0][0]


class TestGetConversation:
    def test_unknown_id_returns_none(self) -> None:
        assert store_with(ScriptedConnection([ScriptedCursor()])).get_conversation("nope") is None

    def test_transcript_folds_sources_into_their_messages(self) -> None:
        conversation_row = ScriptedCursor(row=("Kelp", CREATED_AT, UPDATED_AT))
        source_rows = ScriptedCursor(
            rows=[
                ("m2", 1, "doc::0", {"score": 0.9}),
                ("m2", 2, "doc::1", {"score": 0.5}),
            ]
        )
        message_rows = ScriptedCursor(
            rows=[
                ("m1", "user", "How long?", CREATED_AT, None, None, None, None, None),
                (
                    "m2",
                    "assistant",
                    "About 12 km.",
                    UPDATED_AT,
                    "reranked",
                    {"retrieval": 120},
                    "thinking…",
                    "kelp corridor length",
                    "condense_context",
                ),
            ]
        )
        conn = ScriptedConnection([conversation_row, source_rows, message_rows])
        detail = store_with(conn).get_conversation("c1")
        assert detail is not None
        assert detail.title == "Kelp"
        user, assistant = detail.messages
        assert user.sources == []
        assert assistant.retrieval_method == "reranked"
        assert assistant.condensed_query == "kelp corridor length"
        assert assistant.chat_engine == "condense_context"
        assert [source.chunk_id for source in assistant.sources] == ["doc::0", "doc::1"]
        assert assistant.sources[0].trace == {"score": 0.9}


class TestAppendMessage:
    def test_assistant_turn_snapshots_sources_in_one_transaction(self) -> None:
        conn = ScriptedConnection()
        message_id = store_with(conn).append_message(
            "c1",
            "assistant",
            "About 12 km.",
            retrieval_method="reranked",
            latency_ms={"retrieval": 120},
            reasoning="thinking…",
            condensed_query="kelp corridor length",
            chat_engine="condense_context",
            sources=[make_chunk()],
        )
        assert len(message_id) == 32
        assert conn.transactions == 1
        insert, source_insert, bump = conn.queries
        assert "INSERT INTO messages" in insert[0]
        assert insert[1][0] == message_id
        assert isinstance(insert[1][5], Json)  # latency_ms marshalled as JSONB
        assert "INSERT INTO message_sources" in source_insert[0]
        assert source_insert[1][:3] == (message_id, 1, "doc::3")
        assert isinstance(source_insert[1][3], Json)  # the spec_v2 §9.1 snapshot
        assert "UPDATE conversations SET updated_at" in bump[0]
        assert bump[1] == ("c1",)

    def test_user_turn_writes_no_sources_and_null_latency(self) -> None:
        conn = ScriptedConnection()
        store_with(conn).append_message("c1", "user", "How long?")
        insert, bump = conn.queries
        assert insert[1][5] is None  # latency_ms stays NULL, not Json(None)
        assert "UPDATE conversations" in bump[0]


class TestAutoTitle:
    def test_titles_a_default_titled_conversation(self) -> None:
        conn = FakeTitleConnection(DEFAULT_TITLE)
        title = store_with(conn).auto_title("c1", "How long?", llm=ScriptedLLM("Kelp Facts"))
        assert title == "Kelp Facts"
        assert conn.updates == [("Kelp Facts", "c1")]

    def test_leaves_an_already_titled_conversation_alone(self) -> None:
        conn = FakeTitleConnection("Custom title")
        llm = ScriptedLLM("New Title")
        assert store_with(conn).auto_title("c1", "q", llm=llm) is None
        assert conn.updates == []
        assert llm.calls == 0

    def test_unknown_conversation_returns_none(self) -> None:
        conn = FakeTitleConnection(None)
        assert store_with(conn).auto_title("nope", "q", llm=ScriptedLLM("T")) is None
        assert conn.updates == []
