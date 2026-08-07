"""Unit tests for graph-targeted chat turns (spec_graphrag §4.2, ADR-017).

The same app, the same protocol, the same fakes as ``test_api_chat`` — with
``corpus="graph"`` on the body and a fake graph service behind the seam. What
is under test is the *branch*: which flow runs, what the ``retrieval`` event
says, what the turn persists, and — the ADR's real requirement — that every
way the graph can be unavailable still answers the question and still records
what was asked for.
"""

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from prefect.testing.utilities import prefect_test_harness

from tests.sse import parse_sse
from tests.unit.test_api_chat import (
    CONDENSED_QUERY,
    CondensingFakeLLM,
    FakeConversationStore,
    FakeRetriever,
    StreamingFakeLLM,
)
from varagity.api.deps import (
    get_conversation_store_factory,
    get_graph_chat_preflight,
    get_llm,
    get_retriever_resolver,
    get_services_preflight,
)
from varagity.api.main import create_app
from varagity.graph.records import (
    GraphEntity,
    GraphEvidence,
    GraphRelation,
    GraphRetrieval,
    TranscriptExcerpt,
)
from varagity.graph.service import GraphUnavailable, get_graph_service

THREAD = "iMessage;-;+15125550101"
DOC_KEY = f"{THREAD}::2016-03-04"
TRANSCRIPT = "Thread: Hardware Talk (participants: Bob, Me)\n\n[18:22] Bob: I built the PC"


@pytest.fixture(scope="module", autouse=True)
def prefect_harness() -> Iterator[None]:
    """Ephemeral Prefect API so the graph flow runs tracked, hermetically."""
    with prefect_test_harness():
        yield


def make_retrieval(mode: str = "mix", *, excerpt_text: str = TRANSCRIPT) -> GraphRetrieval:
    return GraphRetrieval(
        evidence=GraphEvidence(
            entities=[GraphEntity(name="Bob", type="person", summary="a friend")],
            relations=[
                GraphRelation(
                    source="Bob",
                    target="mechanical keyboard",
                    label="prefers",
                    description="Bob prefers mechanical keyboards",
                )
            ],
            message_guids=["g1", "g2"],
            raw={"engine": "native payload that must not reach the browser"},
        ),
        excerpts=[
            TranscriptExcerpt(
                doc_key=DOC_KEY,
                thread_name="Hardware Talk",
                span="2016-03-04",
                text=excerpt_text,
                message_guids=["g1", "g2"],
            )
        ],
        mode=mode,
    )


class FakeGraphService:
    """A graph service double: opens (or refuses to) and retrieves."""

    def __init__(self, *, fail_open: bool = False, payload: GraphRetrieval | None = None) -> None:
        self.fail_open = fail_open
        self.payload = payload if payload is not None else make_retrieval()
        self.opens = 0
        self.calls: list[tuple[str, str | None]] = []

    def session(self) -> object:
        self.opens += 1
        if self.fail_open:
            raise GraphUnavailable("the engine storage is locked")
        return object()

    def retrieve(
        self, question: str, *, mode: str | None = None, verbose: int = 0
    ) -> GraphRetrieval:
        self.calls.append((question, mode))
        return self.payload


@pytest.fixture
def store_state() -> dict[str, Any]:
    return {"conversations": {}, "messages": [], "titled": [], "recent_turns_calls": []}


@pytest.fixture
def graph_service() -> FakeGraphService:
    return FakeGraphService()


@pytest.fixture
def preflights() -> dict[str, int]:
    return {"chunk": 0, "graph": 0}


@pytest.fixture
def app(
    store_state: dict[str, Any],
    graph_service: FakeGraphService,
    preflights: dict[str, int],
    settings_env: Any,
) -> FastAPI:
    settings_env(GRAPH_ENABLED="true", GRAPH_QUERY_MODE="mix", CHAT_ENGINE="simple")
    application = create_app()
    retriever = FakeRetriever()
    application.dependency_overrides[get_llm] = lambda: StreamingFakeLLM()
    application.dependency_overrides[get_retriever_resolver] = lambda: lambda name: retriever
    application.dependency_overrides[get_conversation_store_factory] = lambda: (
        lambda: FakeConversationStore(store_state)
    )
    application.dependency_overrides[get_graph_service] = lambda: graph_service

    async def _chunk_preflight() -> None:
        preflights["chunk"] += 1

    async def _graph_preflight() -> None:
        preflights["graph"] += 1

    application.dependency_overrides[get_services_preflight] = lambda: _chunk_preflight
    application.dependency_overrides[get_graph_chat_preflight] = lambda: _graph_preflight
    application.state.fake_retriever = retriever
    return application


async def post_chat(app: FastAPI, body: dict[str, Any]) -> tuple[int, str]:
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://api") as client,
        client.stream("POST", "/api/chat", json=body) as response,
    ):
        text = "".join([chunk async for chunk in response.aiter_text()])
        return response.status_code, text


async def graph_turn(app: FastAPI, **body: Any) -> list[tuple[str | None, Any]]:
    status, text = await post_chat(app, {"query": "what about Bob?", "corpus": "graph", **body})
    assert status == 200
    return parse_sse(text)


def assistant_turn(store_state: dict[str, Any]) -> dict[str, Any]:
    return next(m for m in reversed(store_state["messages"]) if m["role"] == "assistant")


class TestGraphTurn:
    async def test_event_order_is_the_protocol_with_graph_evidence_first(
        self, app: FastAPI
    ) -> None:
        events = await graph_turn(app)
        names = [name for name, _ in events]
        assert names[0] == "retrieval"
        assert names[-1] == "done"
        assert names.index("retrieval") < names.index("token")

    async def test_the_retrieval_event_carries_the_graph_payload(self, app: FastAPI) -> None:
        """★ The wire shape the evidence panel renders (spec_graphrag §4.3)."""
        _, data = (await graph_turn(app))[0]
        assert data["corpus"] == "graph"
        assert data["chunks"] == []
        assert data["method"] == "mix"  # the mode is what retrieved
        graph = data["graph"]
        assert graph["mode"] == "mix"
        assert graph["entities"] == [{"name": "Bob", "type": "person", "summary": "a friend"}]
        assert graph["relations"][0]["target"] == "mechanical keyboard"
        (transcript,) = graph["transcripts"]
        assert transcript["doc_key"] == DOC_KEY
        assert transcript["thread_name"] == "Hardware Talk"
        assert transcript["span"] == "2016-03-04"
        assert transcript["message_count"] == 2
        assert "I built the PC" in transcript["excerpt"]

    async def test_the_engine_native_payload_never_reaches_the_browser(self, app: FastAPI) -> None:
        """``GraphEvidence.raw`` exists for autopsies, not for clients."""
        _, body = await post_chat(app, {"query": "q", "corpus": "graph"})
        assert "native payload that must not reach the browser" not in body

    async def test_an_oversized_transcript_is_clipped_on_the_wire(self, app: FastAPI) -> None:
        app.dependency_overrides[get_graph_service] = lambda: FakeGraphService(
            payload=make_retrieval(excerpt_text="x" * 20_000)
        )
        _, data = (await graph_turn(app))[0]
        excerpt = data["graph"]["transcripts"][0]["excerpt"]
        assert len(excerpt) < 20_000
        assert excerpt.endswith("[…]")

    async def test_the_graph_answered_and_the_chunk_retriever_never_ran(
        self, app: FastAPI, graph_service: FakeGraphService
    ) -> None:
        events = await graph_turn(app)
        assert events[-1][1]["answer"] == "The answer is 42. [SOURCE]: x.txt"
        assert graph_service.calls == [("what about Bob?", "mix")]
        assert app.state.fake_retriever.queries == []

    async def test_the_graph_preflight_replaces_the_chunk_one(
        self, app: FastAPI, preflights: dict[str, int]
    ) -> None:
        """Elasticsearch is not on the graph turn's path (deps.py)."""
        await graph_turn(app)
        assert preflights == {"chunk": 0, "graph": 1}

    async def test_the_turn_persists_its_corpus_evidence_and_day_sources(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        """★ Migration 006's columns, filled by the route (stage-2 decision #12)."""
        await graph_turn(app)
        assistant = assistant_turn(store_state)
        assert assistant["target_corpus"] == "graph"
        assert assistant["retrieval_method"] == "mix"
        assert assistant["graph_evidence"]["mode"] == "mix"
        assert assistant["graph_evidence"]["entities"][0]["name"] == "Bob"
        assert "transcripts" not in assistant["graph_evidence"]  # they are source rows
        assert assistant["sources"] == []  # no chunk evidence on a live graph turn
        (source,) = assistant["graph_sources"]
        assert source.doc_key == DOC_KEY
        assert source.message_guids == ["g1", "g2"]

    async def test_a_chunk_turn_is_untouched_by_any_of_this(
        self, app: FastAPI, store_state: dict[str, Any], preflights: dict[str, int]
    ) -> None:
        """★ The regression guard: corpus defaults to rag and nothing changes."""
        status, body = await post_chat(app, {"query": "what is it?"})
        assert status == 200
        _, retrieval = parse_sse(body)[0]
        assert retrieval["corpus"] == "rag"
        assert retrieval["graph"] is None
        assert retrieval["method"] == "hybrid"
        assert len(retrieval["chunks"]) == 2
        assert preflights == {"chunk": 1, "graph": 0}
        assistant = assistant_turn(store_state)
        assert assistant["target_corpus"] == "rag"
        assert assistant["graph_evidence"] is None
        assert len(assistant["sources"]) == 2

    async def test_an_unknown_corpus_is_rejected_before_anything_runs(
        self, app: FastAPI, graph_service: FakeGraphService
    ) -> None:
        import json

        status, body = await post_chat(app, {"query": "q", "corpus": "notes"})
        assert status == 422
        assert json.loads(body)["error"]["code"] == "validation_error"
        assert graph_service.opens == 0


class TestCondenseOnGraphTurns:
    async def test_the_condensed_query_searches_the_graph_and_is_reported(
        self, app: FastAPI, store_state: dict[str, Any], graph_service: FakeGraphService
    ) -> None:
        """★ Decision #14: the engine seam applies unchanged to graph turns."""
        app.dependency_overrides[get_llm] = lambda: CondensingFakeLLM()
        store_state["conversations"]["known"] = "Existing"
        store_state["messages"] = [
            {"conversation_id": "known", "role": "user", "content": "q0"},
            {"conversation_id": "known", "role": "assistant", "content": "a0"},
        ]
        events = await graph_turn(
            app, conversation_id="known", overrides={"chat_engine": "condense_context"}
        )
        _, retrieval = events[0]
        assert retrieval["condensed_query"] == CONDENSED_QUERY
        assert graph_service.calls == [(CONDENSED_QUERY, "mix")]
        assistant = assistant_turn(store_state)
        assert assistant["condensed_query"] == CONDENSED_QUERY
        assert assistant["chat_engine"] == "condense_context"
        assert assistant["target_corpus"] == "graph"


class TestDegrade:
    async def test_the_kill_switch_answers_from_the_chunk_corpus(
        self,
        app: FastAPI,
        store_state: dict[str, Any],
        graph_service: FakeGraphService,
        settings_env: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """★ ADR-017: an INFO and a chunk answer, never a silent no-op."""
        settings_env(GRAPH_ENABLED="false")
        with caplog.at_level("INFO", logger="varagity.api.routes.chat"):
            events = await graph_turn(app)
        _, retrieval = events[0]
        assert retrieval["corpus"] == "graph"  # what was asked for
        assert retrieval["graph"] is None  # …and what the graph gave: nothing
        assert retrieval["method"] == "hybrid"  # the chunk method really ran
        assert len(retrieval["chunks"]) == 2
        assert events[-1][0] == "done"
        assert graph_service.calls == []
        assert "GRAPH_ENABLED=false" in caplog.text
        assistant = assistant_turn(store_state)
        assert assistant["target_corpus"] == "graph"
        assert assistant["graph_evidence"] is None
        assert len(assistant["sources"]) == 2

    async def test_an_unopenable_engine_degrades_at_warning(
        self, app: FastAPI, store_state: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """★ The other half of the degrade — louder, same outcome."""
        broken = FakeGraphService(fail_open=True)
        app.dependency_overrides[get_graph_service] = lambda: broken
        with caplog.at_level("WARNING", logger="varagity.api.routes.chat"):
            events = await graph_turn(app)
        assert events[-1][0] == "done"
        assert events[0][1]["graph"] is None
        assert broken.opens == 1
        assert broken.calls == []
        assert "storage is locked" in caplog.text
        assert assistant_turn(store_state)["target_corpus"] == "graph"

    async def test_a_degraded_turn_uses_the_chunk_preflight(
        self, app: FastAPI, preflights: dict[str, int], settings_env: Any
    ) -> None:
        """It is about to search Elasticsearch, so ES must be checked."""
        settings_env(GRAPH_ENABLED="false")
        await graph_turn(app)
        assert preflights == {"chunk": 1, "graph": 0}

    async def test_the_graph_session_opens_once_per_turn_not_per_retrieval(
        self, app: FastAPI, graph_service: FakeGraphService
    ) -> None:
        await graph_turn(app)
        assert graph_service.opens == 1


class TestPayloadMapping:
    def test_the_snapshot_and_the_wire_agree_on_the_cited_days(self) -> None:
        """One clip, one label — the panel matches persisted and live evidence."""
        from varagity.api.routes.chat import (
            graph_evidence,
            graph_retrieval_payload,
            graph_snapshots,
        )

        payload = make_retrieval()
        wire = graph_retrieval_payload(payload)
        (snapshot,) = graph_snapshots(payload)
        assert wire.transcripts[0].excerpt == snapshot.excerpt
        assert wire.transcripts[0].doc_key == snapshot.doc_key
        assert wire.transcripts[0].message_count == len(snapshot.message_guids)
        # The persisted evidence blob is the wire payload minus the days.
        assert graph_evidence(payload) == wire.model_dump(mode="json", exclude={"transcripts"})

    def test_an_empty_retrieval_maps_to_an_empty_payload(self) -> None:
        from varagity.api.routes.chat import graph_retrieval_payload, graph_snapshots

        empty = GraphRetrieval(evidence=GraphEvidence(), mode="mix")
        assert graph_retrieval_payload(empty).model_dump() == {
            "mode": "mix",
            "entities": [],
            "relations": [],
            "transcripts": [],
        }
        assert graph_snapshots(empty) == []


def test_the_fakes_match_the_service_surface_the_route_uses() -> None:
    """The doubles above may not drift from what the route actually calls."""
    from varagity.graph.service import GraphService

    for name in ("session", "retrieve"):
        assert callable(getattr(GraphService, name))
        assert callable(getattr(FakeGraphService, name))
