"""Unit tests for POST /api/chat — SSE ordering, persistence, errors.

The app runs over httpx's ASGI transport with every service seam faked
(the flows execute for real under ``prefect_test_harness``, exactly the
production composition). The fakes here are shared with the integration
API suite.
"""

from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from prefect.testing.utilities import prefect_test_harness

from tests.sse import parse_sse
from varagity.api.deps import (
    get_conversation_store_factory,
    get_llm,
    get_retriever_resolver,
    get_services_preflight,
)
from varagity.api.main import create_app
from varagity.models.llm import GenerationTimings
from varagity.stores.conversation_store import ConversationSummary
from varagity.stores.records import RetrievalTrace, RetrievedChunk


@pytest.fixture(scope="module", autouse=True)
def prefect_harness() -> Iterator[None]:
    """Ephemeral Prefect API so the chat flow runs tracked, hermetically."""
    with prefect_test_harness():
        yield


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
        score=0.9 - index / 10,
        trace=RetrievalTrace(
            semantic_rank=index + 1,
            semantic_score=0.9 - index / 10,
            fused_score=0.9 - index / 10,
            fused_rank=index + 1,
            final_rank=index + 1,
        ),
    )


class FakeRetriever:
    """Two scripted chunks; records the ks and queries it was asked for."""

    def __init__(self) -> None:
        self.requested_ks: list[int] = []
        self.queries: list[str] = []

    def encode_query(self, query: str, verbose: int | None = None) -> list[float] | None:
        return None

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        self.requested_ks.append(k)
        self.queries.append(query)
        return [make_chunk(0), make_chunk(1)]


class StreamingFakeLLM:
    """Scripted deltas + optional usage; generate() covers auto-titling."""

    def __init__(
        self,
        deltas: Sequence[str] | None = None,
        timings: Sequence[GenerationTimings] | None = None,
    ) -> None:
        self.deltas = list(
            deltas
            if deltas is not None
            else ["<think>let me see</think>", "The answer", " is 42. [SOURCE]: x.txt"]
        )
        # One reading reported after each delta, llama.cpp-style (cumulative
        # counters); empty = a server that reports no timings.
        self.timings = list(timings or [])

    def generate_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
        on_usage: Callable[[Any], None] | None = None,
        on_timings: Callable[[GenerationTimings], None] | None = None,
    ) -> Iterator[str]:
        def gen() -> Iterator[str]:
            for index, delta in enumerate(self.deltas):
                yield delta
                if on_timings is not None and index < len(self.timings):
                    on_timings(self.timings[index])
            if on_usage is not None:

                class _Usage:
                    prompt_tokens = 11
                    completion_tokens = 7

                on_usage(_Usage())

        return gen()

    def generate(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        return "A Scripted Title"


class BrokenStreamLLM(StreamingFakeLLM):
    """Raises after the first delta — the mid-stream failure shape."""

    def generate_stream(self, messages: Any, **kwargs: Any) -> Iterator[str]:
        def gen() -> Iterator[str]:
            yield "partial "
            raise RuntimeError("llm fell over mid-stream")

        return gen()


# What the condensing fake rewrites the follow-up into (think-wrapped on
# the wire to prove clean_response runs before anything downstream).
CONDENSED_QUERY = "how long is the kelp corridor"


class CondensingFakeLLM(StreamingFakeLLM):
    """generate() plays the condenser; the streamed answer is unchanged."""

    def generate(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        return f"<think>resolving the pronoun</think>{CONDENSED_QUERY}"


class FakeConversationStore:
    """In-memory ConversationStore double (class-level state per make())."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def __enter__(self) -> "FakeConversationStore":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def create_conversation(self, title: str | None = None) -> ConversationSummary:
        conversation_id = f"conv{len(self._state['conversations'])}"
        self._state["conversations"][conversation_id] = title or "New conversation"
        now = datetime.now(UTC)
        return ConversationSummary(
            conversation_id=conversation_id,
            title=title or "New conversation",
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def conversation_exists(self, conversation_id: str) -> bool:
        return conversation_id in self._state["conversations"]

    def recent_turns(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        self._state["recent_turns_calls"].append((conversation_id, limit))
        turns = [
            (m["role"], m["content"])
            for m in self._state["messages"]
            if m["conversation_id"] == conversation_id
        ]
        return turns[-limit:] if limit > 0 else []

    def append_message(self, conversation_id: str, role: str, content: str, **kwargs: Any) -> str:
        self._state["messages"].append(
            {"conversation_id": conversation_id, "role": role, "content": content, **kwargs}
        )
        return f"msg{len(self._state['messages'])}"

    def auto_title(self, conversation_id: str, question: str, llm: Any = None) -> str | None:
        self._state["titled"].append(conversation_id)
        return None


@pytest.fixture
def store_state() -> dict[str, Any]:
    return {"conversations": {}, "messages": [], "titled": [], "recent_turns_calls": []}


@pytest.fixture
def app(store_state: dict[str, Any]) -> FastAPI:
    application = create_app()
    retriever = FakeRetriever()

    def resolve(name: str) -> FakeRetriever:
        if name not in ("semantic", "bm25", "hybrid", "reranked"):
            raise KeyError(f"Unknown retrieval method {name!r}")
        return retriever

    application.dependency_overrides[get_llm] = lambda: StreamingFakeLLM()
    application.dependency_overrides[get_retriever_resolver] = lambda: resolve
    application.dependency_overrides[get_conversation_store_factory] = lambda: (
        lambda: FakeConversationStore(store_state)
    )

    async def _noop_preflight() -> None:
        return None

    application.dependency_overrides[get_services_preflight] = lambda: _noop_preflight
    application.state.fake_retriever = retriever
    return application


async def post_chat(app: FastAPI, body: dict[str, Any]) -> tuple[int, str, str]:
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://api") as client,
        client.stream("POST", "/api/chat", json=body) as response,
    ):
        text = "".join([chunk async for chunk in response.aiter_text()])
        return response.status_code, response.headers.get("content-type", ""), text


async def test_event_order_retrieval_then_deltas_then_done(app: FastAPI) -> None:
    status, content_type, body = await post_chat(app, {"query": "what is it?"})
    assert status == 200
    assert content_type.startswith("text/event-stream")
    events = parse_sse(body)
    names = [name for name, _ in events]
    assert names[0] == "retrieval"
    assert names[-1] == "done"
    assert "token" in names
    assert names.index("retrieval") < names.index("token")
    reasoning_positions = [i for i, n in enumerate(names) if n == "reasoning"]
    assert reasoning_positions and max(reasoning_positions) < names.index("token")


async def test_retrieval_event_carries_chunks_and_traces(app: FastAPI) -> None:
    _, _, body = await post_chat(app, {"query": "q"})
    name, data = parse_sse(body)[0]
    assert name == "retrieval"
    assert data["method"] == "hybrid"  # settings default
    assert data["top_k"] == 10
    assert data["reranked_to"] is None
    assert data["condensed_query"] is None  # simple searches verbatim
    assert len(data["chunks"]) == 2
    first = data["chunks"][0]
    assert first["content"] == "content 0"
    assert first["trace"]["semantic_rank"] == 1
    assert first["metadata"]["file_name"] == "x.txt"


async def test_done_carries_ids_answer_usage_and_latency(
    app: FastAPI, store_state: dict[str, Any]
) -> None:
    _, _, body = await post_chat(app, {"query": "q"})
    name, data = parse_sse(body)[-1]
    assert name == "done"
    assert data["answer"] == "The answer is 42. [SOURCE]: x.txt"
    assert data["conversation_id"] in store_state["conversations"]
    assert data["message_id"] == "msg2"
    assert data["usage"]["prompt_tokens"] == 11
    assert data["usage"]["completion_tokens"] == 7
    assert {"retrieval", "generation", "total"} <= set(data["usage"]["latency_ms"])
    # The default fake reports no timings — the non-llama.cpp shape.
    assert data["usage"]["tokens_per_second"] is None


async def test_no_stats_frames_when_the_server_reports_no_timings(app: FastAPI) -> None:
    _, _, body = await post_chat(app, {"query": "q"})
    names = [name for name, _ in parse_sse(body)]
    assert "stats" not in names


async def test_stats_streamed_and_done_carries_the_final_rate(app: FastAPI) -> None:
    # Cumulative llama.cpp-style readings, all past the warmup gate.
    app.dependency_overrides[get_llm] = lambda: StreamingFakeLLM(
        timings=[
            GenerationTimings(predicted_n=10, predicted_ms=100.0),
            GenerationTimings(predicted_n=20, predicted_ms=250.0),
            GenerationTimings(predicted_n=30, predicted_ms=500.0),
        ]
    )
    _, _, body = await post_chat(app, {"query": "q"})
    events = parse_sse(body)
    names = [name for name, _ in events]

    stats = [data for name, data in events if name == "stats"]
    # At least the first qualifying reading becomes a frame; later ones may
    # be swallowed by the 250 ms throttle (real time, so not pinned here).
    assert stats
    assert stats[0] == {"tokens_per_second": 100.0, "completion_tokens": 10}
    assert names.index("retrieval") < names.index("stats") < names.index("done")

    _, done = events[-1]
    # The done rate is the *last* reading (30 tok / 500 ms), throttle-exempt.
    assert done["usage"]["tokens_per_second"] == pytest.approx(60.0)


async def test_stats_suppressed_below_the_warmup_gate(app: FastAPI) -> None:
    # llama.cpp's first readings (tiny predicted_n) compute absurd rates;
    # the gate keeps them off the wire while done still gets the real total.
    app.dependency_overrides[get_llm] = lambda: StreamingFakeLLM(
        timings=[
            GenerationTimings(predicted_n=1, predicted_ms=0.001),
            GenerationTimings(predicted_n=3, predicted_ms=60.0),
        ]
    )
    _, _, body = await post_chat(app, {"query": "q"})
    events = parse_sse(body)
    assert "stats" not in [name for name, _ in events]
    _, done = events[-1]
    assert done["usage"]["tokens_per_second"] == pytest.approx(50.0)


async def test_turn_persisted_user_then_assistant(
    app: FastAPI, store_state: dict[str, Any]
) -> None:
    await post_chat(app, {"query": "what is it?"})
    roles = [(m["role"], m["content"]) for m in store_state["messages"]]
    assert roles[0] == ("user", "what is it?")
    assert roles[1][0] == "assistant"
    assert roles[1][1] == "The answer is 42. [SOURCE]: x.txt"
    assistant = store_state["messages"][1]
    assert assistant["retrieval_method"] == "hybrid"
    assert assistant["reasoning"] == "let me see"
    assert assistant["condensed_query"] is None  # searched verbatim
    assert assistant["chat_engine"] == "simple"  # the engine is still recorded
    assert len(assistant["sources"]) == 2


async def test_existing_conversation_is_reused(app: FastAPI, store_state: dict[str, Any]) -> None:
    store_state["conversations"]["known"] = "Existing"
    _, _, body = await post_chat(app, {"query": "q", "conversation_id": "known"})
    _, data = parse_sse(body)[-1]
    assert data["conversation_id"] == "known"
    assert len(store_state["conversations"]) == 1  # nothing new created


async def test_history_loaded_for_an_existing_conversation(
    app: FastAPI, store_state: dict[str, Any]
) -> None:
    """The route loads the newest turns (simple ignores them)."""
    store_state["conversations"]["known"] = "Existing"
    store_state["messages"] = [
        {"conversation_id": "known", "role": "user", "content": "q0"},
        {"conversation_id": "known", "role": "assistant", "content": "a0"},
    ]
    _, _, body = await post_chat(app, {"query": "follow-up?", "conversation_id": "known"})
    assert parse_sse(body)[-1][0] == "done"  # unchanged behavior under simple
    assert store_state["recent_turns_calls"] == [("known", 6)]


async def test_no_history_lookup_without_a_conversation_id(
    app: FastAPI, store_state: dict[str, Any]
) -> None:
    """A fresh conversation has no history to load — no store round-trip."""
    await post_chat(app, {"query": "q"})
    assert store_state["recent_turns_calls"] == []


async def test_unknown_conversation_404_before_stream(app: FastAPI) -> None:
    status, content_type, body = await post_chat(app, {"query": "q", "conversation_id": "ghost"})
    assert status == 404
    assert "text/event-stream" not in content_type
    import json

    assert json.loads(body)["error"]["code"] == "conversation_not_found"


async def test_unknown_retrieval_method_422(app: FastAPI) -> None:
    status, _, body = await post_chat(
        app, {"query": "q", "overrides": {"retrieval_method": "made_up"}}
    )
    import json

    assert status == 422
    assert json.loads(body)["error"]["code"] == "unknown_retrieval_method"


async def test_top_k_override_reaches_the_retriever(app: FastAPI) -> None:
    await post_chat(app, {"query": "q", "overrides": {"top_k": 3}})
    assert app.state.fake_retriever.requested_ks == [3]


async def test_midstream_failure_emits_error_event(
    app: FastAPI, store_state: dict[str, Any]
) -> None:
    app.dependency_overrides[get_llm] = lambda: BrokenStreamLLM()
    status, _, body = await post_chat(app, {"query": "q"})
    assert status == 200  # headers were long gone — failure is in-band
    events = parse_sse(body)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "pipeline_error"
    assert "RuntimeError" in events[-1][1]["message"]
    assert store_state["messages"] == []  # nothing persisted without done


@pytest.fixture
def condense_settings(settings_env: Callable[..., None]) -> None:
    """Pin the condense knobs so the machine's .env can't leak in."""
    settings_env(
        CONDENSE_ENABLED="true",
        CONDENSE_MODEL_TYPE="default",
        CONDENSE_HISTORY_TURNS="6",
        CONDENSE_MAX_TOKENS="128",
        CONDENSE_MAX_CHARS="512",
    )


async def test_condense_rewrite_reaches_event_retriever_and_persistence(
    app: FastAPI, store_state: dict[str, Any], condense_settings: None
) -> None:
    """★ The spec_v3 §4.7 wire, end to end at the route.

    With history and the condense_context override, the retrieval event
    carries the think-stripped rewrite, the retriever searches with it,
    and the persisted turn snapshots rewrite + engine.
    """
    app.dependency_overrides[get_llm] = lambda: CondensingFakeLLM()
    store_state["conversations"]["known"] = "Existing"
    store_state["messages"] = [
        {"conversation_id": "known", "role": "user", "content": "q0"},
        {"conversation_id": "known", "role": "assistant", "content": "a0"},
    ]
    _, _, body = await post_chat(
        app,
        {
            "query": "how long is it?",
            "conversation_id": "known",
            "overrides": {"chat_engine": "condense_context"},
        },
    )
    events = parse_sse(body)
    names = [name for name, _ in events]
    # Evidence before prose survives the extra pre-retrieval stage.
    assert names[0] == "retrieval"
    assert names.index("retrieval") < names.index("token")
    assert names[-1] == "done"

    retrieval = events[0][1]
    assert retrieval["condensed_query"] == CONDENSED_QUERY  # think-stripped
    assert app.state.fake_retriever.queries == [CONDENSED_QUERY]

    assistant = store_state["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["condensed_query"] == CONDENSED_QUERY
    assert assistant["chat_engine"] == "condense_context"


async def test_condense_context_first_turn_searches_verbatim(
    app: FastAPI, store_state: dict[str, Any], condense_settings: None
) -> None:
    """No conversation ⇒ no history ⇒ no condense call, null on the wire."""
    app.dependency_overrides[get_llm] = lambda: CondensingFakeLLM()
    _, _, body = await post_chat(
        app, {"query": "how long is it?", "overrides": {"chat_engine": "condense_context"}}
    )
    retrieval = parse_sse(body)[0][1]
    assert retrieval["condensed_query"] is None
    assert app.state.fake_retriever.queries == ["how long is it?"]
    assert store_state["messages"][-1]["condensed_query"] is None
    assert store_state["messages"][-1]["chat_engine"] == "condense_context"


async def test_unknown_chat_engine_override_422(app: FastAPI) -> None:
    status, _, body = await post_chat(app, {"query": "q", "overrides": {"chat_engine": "made_up"}})
    import json

    assert status == 422
    assert json.loads(body)["error"]["code"] == "unknown_chat_engine"


async def test_typoed_override_field_422(app: FastAPI) -> None:
    """ChatOverrides is extra="forbid" — a misspelled knob fails loudly."""
    status, _, _ = await post_chat(app, {"query": "q", "overrides": {"chat_enginee": "simple"}})
    assert status == 422


async def test_history_limit_reads_condense_history_turns(
    app: FastAPI, store_state: dict[str, Any], settings_env: Callable[..., None]
) -> None:
    """The hardcoded history-turns literal is gone: the load bound is the setting."""
    settings_env(CONDENSE_HISTORY_TURNS="2")
    store_state["conversations"]["known"] = "Existing"
    await post_chat(app, {"query": "q", "conversation_id": "known"})
    assert store_state["recent_turns_calls"] == [("known", 2)]
