"""Integration tests for the v2 persistence layer and the chat API over it.

Runner, store, and route against a real pgvector Postgres (testcontainers).
Covers the spec_v2 §9.3 convergence claims (fresh vs existing volume), the
§9.1 conversation round-trip with snapshotted sources, and a full
``POST /api/chat`` SSE stream over ``httpx.AsyncClient`` that persists into
the real database (fake embeddings/LLM — the GPU services are not needed).

Select with ``pytest -m integration`` (needs Docker).
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from prefect.testing.utilities import prefect_test_harness

from tests.sse import parse_sse
from tests.unit.test_api_chat import (
    CONDENSED_QUERY,
    CondensingFakeLLM,
    FakeRetriever,
    StreamingFakeLLM,
)
from varagity.eval.containers import ephemeral_postgres
from varagity.stores.conversation_store import DEFAULT_TITLE, ConversationStore
from varagity.stores.migrate import MIGRATIONS_PATH, run_migrations
from varagity.stores.records import RetrievalTrace, RetrievedChunk

pytestmark = pytest.mark.integration

CONVERSATION_TABLES = ("conversations", "messages", "message_sources")

ALL_MIGRATIONS = [
    "001_conversations.sql",
    "002_app_settings.sql",
    "003_condensed_query.sql",
    "004_message_engine.sql",
]


@pytest.fixture(scope="module")
def pg_conninfo() -> Iterator[str]:
    """A pgvector Postgres with schema.sql applied, for the whole module."""
    with ephemeral_postgres() as conninfo:
        yield conninfo


@pytest.fixture
def migrated_conninfo(pg_conninfo: str) -> str:
    """The container database with migrations applied and tables truncated."""
    with psycopg.connect(pg_conninfo, autocommit=True) as conn:
        run_migrations(conn)
        conn.execute("TRUNCATE conversations CASCADE")
    return pg_conninfo


def _table_names(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    return {row[0] for row in rows}


def _message_columns(conn: psycopg.Connection) -> dict[str, tuple[str, str]]:
    """Shape of the messages table: column → (data_type, is_nullable)."""
    rows = conn.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'messages'"
    ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _drop_migrated_state(conn: psycopg.Connection) -> None:
    conn.execute(
        "DROP TABLE IF EXISTS message_sources, messages, conversations, "
        "app_settings, schema_migrations CASCADE"
    )


class TestMigrationRunner:
    def test_fresh_volume_applies_then_noops(self, pg_conninfo: str) -> None:
        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            _drop_migrated_state(conn)
            first = run_migrations(conn)
            assert first == ALL_MIGRATIONS
            assert set(CONVERSATION_TABLES) | {"app_settings"} <= _table_names(conn)
            # The v3 columns land inert: nullable TEXT, nothing writes them.
            columns = _message_columns(conn)
            assert columns["condensed_query"] == ("text", "YES")
            assert columns["chat_engine"] == ("text", "YES")
            # Idempotent: a second run applies nothing and changes nothing.
            assert run_migrations(conn) == []
            applied = conn.execute("SELECT name FROM schema_migrations ORDER BY name").fetchall()
            assert [row[0] for row in applied] == ALL_MIGRATIONS

    def test_existing_v1_volume_converges(self, pg_conninfo: str) -> None:
        """A volume with only schema.sql state (v1) gains the v2 tables."""
        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            _drop_migrated_state(conn)
            assert {"documents", "chunks"} <= _table_names(conn)  # v1 state intact
            run_migrations(conn)
            assert set(CONVERSATION_TABLES) | {"app_settings"} <= _table_names(conn)
            assert {"documents", "chunks"} <= _table_names(conn)  # untouched

    def test_v2_volume_and_fresh_volume_converge_on_v3_columns(
        self, pg_conninfo: str, tmp_path: Path
    ) -> None:
        """003/004 alone reconcile a v2-shaped volume; both paths converge.

        The invariant behind plan decisions #11/#13: an existing volume
        (001+002 applied, rows present) and a fresh volume must end with the
        byte-identical messages shape, and the reconciling run must apply
        exactly the two v3 migrations without touching existing rows.
        """
        v2_dir = tmp_path / "v2_migrations"
        v2_dir.mkdir()
        for name in ("001_conversations.sql", "002_app_settings.sql"):
            (v2_dir / name).write_text((MIGRATIONS_PATH / name).read_text())

        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            # Path A: a v2-shaped volume — only 001+002 known, one turn stored.
            _drop_migrated_state(conn)
            assert run_migrations(conn, v2_dir) == ALL_MIGRATIONS[:2]
            assert "condensed_query" not in _message_columns(conn)  # truly v2
            # Planted with raw SQL, as v2 code wrote it: today's store
            # speaks the v3 schema (it names condensed_query/chat_engine in
            # its INSERT) and rightly cannot write into a pre-003 volume.
            conn.execute(
                "INSERT INTO conversations (conversation_id, title) VALUES ('c-v2', 'Existing')"
            )
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, content) "
                "VALUES ('m-v2', 'c-v2', 'user', 'pre-migration turn')"
            )

            assert run_migrations(conn) == ALL_MIGRATIONS[2:]  # exactly 003/004
            v2_shape = _message_columns(conn)
            row = conn.execute(
                "SELECT content, condensed_query, chat_engine FROM messages"
            ).fetchone()
            assert row == ("pre-migration turn", None, None)  # rows untouched, inert NULLs
            assert run_migrations(conn) == []  # idempotent on the converged volume

            # Path B: a fresh volume, the full runner from scratch.
            _drop_migrated_state(conn)
            run_migrations(conn)
            assert _message_columns(conn) == v2_shape  # the two paths converge


def make_chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc::{index}",
        doc_id="doc",
        original_index=index,
        content=f"content {index}",
        context="blurb",
        metadata={
            "source": "/docs/x.txt",
            "file_name": "x.txt",
            "file_type": "txt",
            "page": None,
            "extraction": "text",
        },
        score=0.9,
        trace=RetrievalTrace(fused_score=0.9, fused_rank=index + 1, final_rank=index + 1),
    )


class TestConversationStoreRoundTrip:
    def test_full_crud_round_trip(self, migrated_conninfo: str) -> None:
        with ConversationStore(migrated_conninfo) as store:
            created = store.create_conversation()
            assert created.title == DEFAULT_TITLE
            assert store.conversation_exists(created.conversation_id)

            store.append_message(created.conversation_id, "user", "what is it?")
            message_id = store.append_message(
                created.conversation_id,
                "assistant",
                "It is 42. [SOURCE]: x.txt",
                retrieval_method="hybrid",
                latency_ms={"retrieval": 120, "generation": 900, "total": 1020},
                reasoning="let me see",
                condensed_query="what is the kelp corridor?",
                chat_engine="condense_context",
                sources=[make_chunk(0), make_chunk(1)],
            )

            detail = store.get_conversation(created.conversation_id)
            assert detail is not None
            assert [m.role for m in detail.messages] == ["user", "assistant"]
            # The user turn carries no engine provenance — NULLs round-trip.
            assert detail.messages[0].condensed_query is None
            assert detail.messages[0].chat_engine is None
            assistant = detail.messages[1]
            assert assistant.message_id == message_id
            assert assistant.retrieval_method == "hybrid"
            assert assistant.latency_ms == {"retrieval": 120, "generation": 900, "total": 1020}
            assert assistant.reasoning == "let me see"
            # The v3 §8 snapshot columns round-trip (migrations 003/004).
            assert assistant.condensed_query == "what is the kelp corridor?"
            assert assistant.chat_engine == "condense_context"
            assert [s.rank for s in assistant.sources] == [1, 2]
            snapshot = assistant.sources[0].trace
            assert snapshot["content"] == "content 0"
            assert snapshot["file_name"] == "x.txt"
            assert snapshot["trace"]["fused_rank"] == 1

            summaries = store.list_conversations()
            assert summaries[0].conversation_id == created.conversation_id
            assert summaries[0].message_count == 2

    def test_delete_cascades_to_messages_and_sources(self, migrated_conninfo: str) -> None:
        with ConversationStore(migrated_conninfo) as store:
            created = store.create_conversation()
            store.append_message(created.conversation_id, "assistant", "a", sources=[make_chunk(0)])
            assert store.delete_conversation(created.conversation_id) == 1
            assert store.get_conversation(created.conversation_id) is None
        with psycopg.connect(migrated_conninfo, autocommit=True) as conn:
            for table in ("messages", "message_sources"):
                count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
                assert count is not None and count[0] == 0

    def test_delete_unknown_is_a_noop(self, migrated_conninfo: str) -> None:
        with ConversationStore(migrated_conninfo) as store:
            assert store.delete_conversation("ghost") == 0

    def test_recent_turns_bounds_in_sql_and_orders_oldest_first(
        self, migrated_conninfo: str
    ) -> None:
        """The chat engine's history feed: newest ``limit`` turns, oldest first."""
        with ConversationStore(migrated_conninfo) as store:
            created = store.create_conversation()
            conversation_id = created.conversation_id
            for index in range(3):
                store.append_message(conversation_id, "user", f"q{index}")
                store.append_message(
                    conversation_id,
                    "assistant",
                    f"a{index}",
                    retrieval_method="hybrid",
                    sources=[make_chunk(index)],  # must never be hydrated here
                )

            # The newest three of six messages, re-ordered oldest first.
            assert store.recent_turns(conversation_id, limit=3) == [
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
            ]
            # A limit beyond the history returns everything, oldest first.
            assert store.recent_turns(conversation_id, limit=100) == [
                ("user", "q0"),
                ("assistant", "a0"),
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
            ]
            assert store.recent_turns(conversation_id, limit=0) == []
            assert store.recent_turns("ghost", limit=5) == []

    def test_auto_title_only_replaces_the_default(self, migrated_conninfo: str) -> None:
        class ScriptedLLM:
            def generate(self, messages: Any, **kwargs: Any) -> str:
                return "Kelp Corridor Facts"

        with ConversationStore(migrated_conninfo) as store:
            created = store.create_conversation()
            assert (
                store.auto_title(created.conversation_id, "How long?", llm=ScriptedLLM())
                == "Kelp Corridor Facts"
            )
            # Second call: no longer default-titled — untouched.
            assert store.auto_title(created.conversation_id, "Other?", llm=ScriptedLLM()) is None

    def test_updated_at_bumps_on_append(self, migrated_conninfo: str) -> None:
        with ConversationStore(migrated_conninfo) as store:
            created = store.create_conversation()
            store.append_message(created.conversation_id, "user", "q")
            detail = store.get_conversation(created.conversation_id)
            assert detail is not None
            assert detail.updated_at >= created.updated_at


@pytest.fixture(scope="module")
def prefect_harness() -> Iterator[None]:
    """Ephemeral Prefect API for the API-over-real-Postgres stream test."""
    with prefect_test_harness():
        yield


class TestChatStreamPersistsToPostgres:
    async def test_full_sse_stream_persists_the_turn(
        self,
        migrated_conninfo: str,
        prefect_harness: None,
        settings_env: Callable[..., None],
    ) -> None:
        params = psycopg.conninfo.conninfo_to_dict(migrated_conninfo)
        settings_env(
            POSTGRES_HOST=str(params["host"]),
            POSTGRES_PORT=str(params["port"]),
            POSTGRES_DB=str(params["dbname"]),
            POSTGRES_USER=str(params["user"]),
            POSTGRES_PASSWORD=str(params["password"]),
            CONDENSE_ENABLED="true",
            CONDENSE_HISTORY_TURNS="6",
            CONDENSE_MAX_CHARS="512",
        )

        from varagity.api.deps import (
            get_llm,
            get_retriever_resolver,
            get_services_preflight,
        )
        from varagity.api.main import create_app

        app = create_app()
        app.dependency_overrides[get_llm] = lambda: StreamingFakeLLM()
        app.dependency_overrides[get_retriever_resolver] = lambda: lambda name: FakeRetriever()

        async def _noop() -> None:
            return None

        app.dependency_overrides[get_services_preflight] = lambda: _noop

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            async with client.stream(
                "POST", "/api/chat", json={"query": "what is it?"}
            ) as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])
            events = parse_sse(body)
            names = [name for name, _ in events]
            assert names[0] == "retrieval"
            assert names[-1] == "done"
            done = events[-1][1]

            # The turn is really in Postgres — reread over the REST route.
            detail = (await client.get(f"/api/conversations/{done['conversation_id']}")).json()
            roles = [m["role"] for m in detail["messages"]]
            assert roles == ["user", "assistant"]
            assert detail["messages"][1]["content"] == done["answer"]
            assert detail["messages"][1]["sources"][0]["trace"]["content"] == "content 0"
            assert detail["messages"][1]["condensed_query"] is None  # simple: verbatim
            assert detail["messages"][1]["chat_engine"] == "simple"

            # Turn 2: a condensed follow-up in the same conversation — the
            # spec_v3 §8 wire round-trip: engine rewrite → real columns →
            # REST transcript (the history the engine read is turn 1's,
            # loaded from the real recent_turns).
            app.dependency_overrides[get_llm] = lambda: CondensingFakeLLM()
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "query": "how long is it?",
                    "conversation_id": done["conversation_id"],
                    "overrides": {"chat_engine": "condense_context"},
                },
            ) as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])
            follow_up = parse_sse(body)
            assert follow_up[0][1]["condensed_query"] == CONDENSED_QUERY

            detail = (await client.get(f"/api/conversations/{done['conversation_id']}")).json()
            assert [m["role"] for m in detail["messages"]] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            condensed_turn = detail["messages"][3]
            assert condensed_turn["condensed_query"] == CONDENSED_QUERY
            assert condensed_turn["chat_engine"] == "condense_context"

        # And it survives a fresh connection (nothing lived only in memory).
        with ConversationStore(migrated_conninfo) as store:
            fresh = store.get_conversation(done["conversation_id"])
            assert fresh is not None
            assert len(fresh.messages) == 4
            assert fresh.messages[3].condensed_query == CONDENSED_QUERY
