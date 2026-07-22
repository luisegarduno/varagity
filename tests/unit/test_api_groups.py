"""Unit tests for the conversation-group CRUD routes over a fake store."""

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from varagity.api.deps import get_conversation_store
from varagity.api.main import create_app
from varagity.stores.conversation_store import ConversationGroup

NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC)


def group(group_id: str, name: str = "G") -> ConversationGroup:
    return ConversationGroup(group_id=group_id, name=name, created_at=NOW)


class FakeStore:
    def __init__(self) -> None:
        self.groups: list[ConversationGroup] = []
        self.created_names: list[str] = []
        self.deleted: list[str] = []

    def list_groups(self) -> list[ConversationGroup]:
        return self.groups

    def create_group(self, name: str) -> ConversationGroup:
        self.created_names.append(name)
        return group("new-group-id", name)

    def delete_group(self, group_id: str) -> int:
        if any(existing.group_id == group_id for existing in self.groups):
            self.deleted.append(group_id)
            return 1
        return 0


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


async def test_list_groups(app: FastAPI, store: FakeStore) -> None:
    store.groups = [group("g1", "Alpha"), group("g2", "Beta")]
    response = await request(app, "GET", "/api/groups")
    assert response.status_code == 200
    data = response.json()
    assert [(g["group_id"], g["name"]) for g in data] == [("g1", "Alpha"), ("g2", "Beta")]


async def test_list_groups_empty(app: FastAPI) -> None:
    response = await request(app, "GET", "/api/groups")
    assert response.status_code == 200
    assert response.json() == []


async def test_create_group_201(app: FastAPI, store: FakeStore) -> None:
    response = await request(app, "POST", "/api/groups", json={"name": "Research"})
    assert response.status_code == 201
    data = response.json()
    assert data["group_id"] == "new-group-id"
    assert data["name"] == "Research"
    assert store.created_names == ["Research"]


async def test_create_group_requires_a_name(app: FastAPI, store: FakeStore) -> None:
    assert (await request(app, "POST", "/api/groups", json={})).status_code == 422
    assert (await request(app, "POST", "/api/groups", json={"name": ""})).status_code == 422
    assert store.created_names == []


async def test_create_group_rejects_unknown_fields(app: FastAPI) -> None:
    response = await request(app, "POST", "/api/groups", json={"name": "x", "color": "red"})
    assert response.status_code == 422


async def test_delete_group_204(app: FastAPI, store: FakeStore) -> None:
    store.groups = [group("g1")]
    response = await request(app, "DELETE", "/api/groups/g1")
    assert response.status_code == 204
    assert store.deleted == ["g1"]


async def test_delete_unknown_group_404(app: FastAPI) -> None:
    response = await request(app, "DELETE", "/api/groups/ghost")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "group_not_found"
