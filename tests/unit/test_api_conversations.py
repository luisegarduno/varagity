"""Unit tests for the conversation CRUD routes over a fake store."""

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from varagity.api.deps import get_conversation_store
from varagity.api.main import create_app
from varagity.stores.conversation_store import (
    ConversationDetail,
    ConversationSummary,
    MessageRecord,
    StoredSource,
)

NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)


def summary(
    conversation_id: str, title: str = "T", count: int = 0, group_id: str | None = None
) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=conversation_id,
        title=title,
        created_at=NOW,
        updated_at=NOW,
        message_count=count,
        group_id=group_id,
    )


class FakeStore:
    def __init__(self) -> None:
        self.summaries: list[ConversationSummary] = []
        self.details: dict[str, ConversationDetail] = {}
        self.deleted: list[str] = []
        self.created_titles: list[str | None] = []
        self.known_groups: set[str] = set()
        self.moves: list[tuple[str, str | None]] = []

    def list_conversations(self) -> list[ConversationSummary]:
        return self.summaries

    def create_conversation(self, title: str | None = None) -> ConversationSummary:
        self.created_titles.append(title)
        return summary("new-id", title or "New conversation")

    def get_conversation(self, conversation_id: str) -> ConversationDetail | None:
        return self.details.get(conversation_id)

    def delete_conversation(self, conversation_id: str) -> int:
        if conversation_id in self.details:
            self.deleted.append(conversation_id)
            return 1
        return 0

    def group_exists(self, group_id: str) -> bool:
        return group_id in self.known_groups

    def set_conversation_group(self, conversation_id: str, group_id: str | None) -> int:
        if conversation_id not in self.details:
            return 0
        self.moves.append((conversation_id, group_id))
        return 1


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def app(store: FakeStore) -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_conversation_store] = lambda: store
    return application


async def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.request(method, path, **kwargs)


async def test_list_conversations(app: FastAPI, store: FakeStore) -> None:
    store.summaries = [summary("a", "First", 4, group_id="g1"), summary("b", "Second", 2)]
    response = await request(app, "GET", "/api/conversations")
    assert response.status_code == 200
    data = response.json()
    assert [c["conversation_id"] for c in data] == ["a", "b"]
    assert data[0]["message_count"] == 4
    # The sidebar partitions on group_id, so the list must carry it.
    assert data[0]["group_id"] == "g1"
    assert data[1]["group_id"] is None


async def test_create_conversation_201(app: FastAPI, store: FakeStore) -> None:
    response = await request(app, "POST", "/api/conversations", json={})
    assert response.status_code == 201
    assert response.json()["conversation_id"] == "new-id"
    assert store.created_titles == [None]


async def test_create_conversation_with_title(app: FastAPI, store: FakeStore) -> None:
    response = await request(app, "POST", "/api/conversations", json={"title": "My chat"})
    assert response.status_code == 201
    assert store.created_titles == ["My chat"]


async def test_get_transcript_with_sources(app: FastAPI, store: FakeStore) -> None:
    store.details["c1"] = ConversationDetail(
        conversation_id="c1",
        title="T",
        created_at=NOW,
        updated_at=NOW,
        messages=[
            MessageRecord(message_id="m1", role="user", content="q?", created_at=NOW, sources=[]),
            MessageRecord(
                message_id="m2",
                role="assistant",
                content="a. [SOURCE]: x.txt",
                created_at=NOW,
                retrieval_method="hybrid",
                latency_ms={"total": 1200},
                reasoning="hmm",
                sources=[
                    StoredSource(rank=1, chunk_id="doc::0", trace={"score": 0.9, "trace": None})
                ],
            ),
        ],
    )
    response = await request(app, "GET", "/api/conversations/c1")
    assert response.status_code == 200
    data = response.json()
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assistant = data["messages"][1]
    assert assistant["sources"][0]["rank"] == 1
    assert assistant["latency_ms"] == {"total": 1200}
    assert assistant["reasoning"] == "hmm"


async def test_get_unknown_conversation_404(app: FastAPI) -> None:
    response = await request(app, "GET", "/api/conversations/ghost")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


async def test_move_conversation_into_group_204(app: FastAPI, store: FakeStore) -> None:
    store.details["c1"] = ConversationDetail(
        conversation_id="c1", title="T", created_at=NOW, updated_at=NOW, messages=[]
    )
    store.known_groups.add("g1")
    response = await request(app, "PATCH", "/api/conversations/c1", json={"group_id": "g1"})
    assert response.status_code == 204
    assert store.moves == [("c1", "g1")]


async def test_ungroup_conversation_204(app: FastAPI, store: FakeStore) -> None:
    store.details["c1"] = ConversationDetail(
        conversation_id="c1", title="T", created_at=NOW, updated_at=NOW, messages=[]
    )
    response = await request(app, "PATCH", "/api/conversations/c1", json={"group_id": None})
    assert response.status_code == 204
    assert store.moves == [("c1", None)]


async def test_move_to_unknown_group_404(app: FastAPI, store: FakeStore) -> None:
    store.details["c1"] = ConversationDetail(
        conversation_id="c1", title="T", created_at=NOW, updated_at=NOW, messages=[]
    )
    response = await request(app, "PATCH", "/api/conversations/c1", json={"group_id": "ghost"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "group_not_found"
    assert store.moves == []


async def test_move_unknown_conversation_404(app: FastAPI, store: FakeStore) -> None:
    store.known_groups.add("g1")
    response = await request(app, "PATCH", "/api/conversations/ghost", json={"group_id": "g1"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


async def test_move_requires_the_group_id_field(app: FastAPI) -> None:
    response = await request(app, "PATCH", "/api/conversations/c1", json={})
    assert response.status_code == 422


async def test_move_rejects_unknown_fields(app: FastAPI) -> None:
    response = await request(
        app, "PATCH", "/api/conversations/c1", json={"group_id": None, "title": "x"}
    )
    assert response.status_code == 422


async def test_delete_conversation_204(app: FastAPI, store: FakeStore) -> None:
    store.details["c1"] = ConversationDetail(
        conversation_id="c1", title="T", created_at=NOW, updated_at=NOW, messages=[]
    )
    response = await request(app, "DELETE", "/api/conversations/c1")
    assert response.status_code == 204
    assert store.deleted == ["c1"]


async def test_delete_unknown_conversation_404(app: FastAPI) -> None:
    response = await request(app, "DELETE", "/api/conversations/ghost")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"
