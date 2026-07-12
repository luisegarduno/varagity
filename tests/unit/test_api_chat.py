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
    """Two scripted chunks; records the ks it was asked for."""

    def __init__(self) -> None:
        self.requested_ks: list[int] = []

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
        return [make_chunk(0), make_chunk(1)]


class StreamingFakeLLM:
    """Scripted deltas + optional usage; generate() covers auto-titling."""

    def __init__(self, deltas: Sequence[str] | None = None) -> None:
        self.deltas = list(
            deltas
            if deltas is not None
            else ["<think>let me see</think>", "The answer", " is 42. [SOURCE]: x.txt"]
        )

    def generate_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> Iterator[str]:
        def gen() -> Iterator[str]:
            yield from self.deltas
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
    return {"conversations": {}, "messages": [], "titled": []}


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
    assert len(assistant["sources"]) == 2


async def test_existing_conversation_is_reused(app: FastAPI, store_state: dict[str, Any]) -> None:
    store_state["conversations"]["known"] = "Existing"
    _, _, body = await post_chat(app, {"query": "q", "conversation_id": "known"})
    _, data = parse_sse(body)[-1]
    assert data["conversation_id"] == "known"
    assert len(store_state["conversations"]) == 1  # nothing new created


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
