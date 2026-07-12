"""Integration tests for the v2 persistence layer and the chat API over it.

Runner, store, and route against a real pgvector Postgres (testcontainers).
Covers the spec_v2 §9.3 convergence claims (fresh vs existing volume), the
§9.1 conversation round-trip with snapshotted sources, and a full
``POST /api/chat`` SSE stream over ``httpx.AsyncClient`` that persists into
the real database (fake embeddings/LLM — the GPU services are not needed).

Select with ``pytest -m integration`` (needs Docker).
"""

from collections.abc import Callable, Iterator
from typing import Any

import httpx
import psycopg
import pytest
from prefect.testing.utilities import prefect_test_harness

from tests.sse import parse_sse
from tests.unit.test_api_chat import FakeRetriever, StreamingFakeLLM
from varagity.eval.containers import ephemeral_postgres
from varagity.stores.conversation_store import DEFAULT_TITLE, ConversationStore
from varagity.stores.migrate import run_migrations
from varagity.stores.records import RetrievalTrace, RetrievedChunk

pytestmark = pytest.mark.integration

CONVERSATION_TABLES = ("conversations", "messages", "message_sources")


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


class TestMigrationRunner:
    def test_fresh_volume_applies_then_noops(self, pg_conninfo: str) -> None:
        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            conn.execute(
                "DROP TABLE IF EXISTS message_sources, messages, conversations, "
                "schema_migrations CASCADE"
            )
            first = run_migrations(conn)
            assert first == ["001_conversations.sql"]
            assert set(CONVERSATION_TABLES) <= _table_names(conn)
            # Idempotent: a second run applies nothing and changes nothing.
            assert run_migrations(conn) == []
            applied = conn.execute("SELECT name FROM schema_migrations").fetchall()
            assert [row[0] for row in applied] == ["001_conversations.sql"]

    def test_existing_v1_volume_converges(self, pg_conninfo: str) -> None:
        """A volume with only schema.sql state (v1) gains the v2 tables."""
        with psycopg.connect(pg_conninfo, autocommit=True) as conn:
            conn.execute(
                "DROP TABLE IF EXISTS message_sources, messages, conversations, "
                "schema_migrations CASCADE"
            )
            assert {"documents", "chunks"} <= _table_names(conn)  # v1 state intact
            run_migrations(conn)
            assert set(CONVERSATION_TABLES) <= _table_names(conn)
            assert {"documents", "chunks"} <= _table_names(conn)  # untouched


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
                sources=[make_chunk(0), make_chunk(1)],
            )

            detail = store.get_conversation(created.conversation_id)
            assert detail is not None
            assert [m.role for m in detail.messages] == ["user", "assistant"]
            assistant = detail.messages[1]
            assert assistant.message_id == message_id
            assert assistant.retrieval_method == "hybrid"
            assert assistant.latency_ms == {"retrieval": 120, "generation": 900, "total": 1020}
            assert assistant.reasoning == "let me see"
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

        # And it survives a fresh connection (nothing lived only in memory).
        with ConversationStore(migrated_conninfo) as store:
            fresh = store.get_conversation(done["conversation_id"])
            assert fresh is not None
            assert len(fresh.messages) == 2
