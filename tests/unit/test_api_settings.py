"""Unit tests for GET/PATCH /api/settings — the live settings surface.

The app runs over httpx's ASGI transport with the two store dependencies
faked; the override layer itself runs for real (process env + cache), so
these tests cover the route↔layer composition: effective values, persist
rows, atomic rejection, override clearing, and the corpus-stale flag.
"""

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from varagity.api import runtime_settings
from varagity.api.deps import get_app_settings_store, get_vector_store
from varagity.api.main import create_app
from varagity.api.runtime_settings import OVERRIDABLE
from varagity.config import get_settings


class FakeAppSettingsStore:
    """In-memory AppSettingsStore double."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def load_overrides(self) -> dict[str, str]:
        return dict(self._state["overrides"])

    def set_override(self, name: str, value: str) -> None:
        self._state["overrides"][name] = value

    def delete_override(self, name: str) -> None:
        self._state["overrides"].pop(name, None)

    def is_corpus_stale(self) -> bool:
        return self._state["stale"]

    def set_corpus_stale(self, stale: bool) -> None:
        self._state["stale"] = stale


class FakeVectorStore:
    """document_count-only double for the corpus-emptiness check."""

    def __init__(self, n_documents: int) -> None:
        self.n_documents = n_documents

    def document_count(self) -> int:
        return self.n_documents


@pytest.fixture(autouse=True)
def isolate_overrides() -> Iterator[None]:
    """Reset the override layer's process-global state around every test."""
    runtime_settings.reset_for_tests()
    yield
    runtime_settings.reset_for_tests()


@pytest.fixture
def store_state() -> dict[str, Any]:
    return {"overrides": {}, "stale": False}


@pytest.fixture
def app(store_state: dict[str, Any]) -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_app_settings_store] = lambda: FakeAppSettingsStore(
        store_state
    )
    application.dependency_overrides[get_vector_store] = lambda: FakeVectorStore(n_documents=3)
    return application


async def get_catalog(app: FastAPI) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.get("/api/settings")


async def patch_settings(app: FastAPI, overrides: dict[str, Any]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.patch("/api/settings", json={"overrides": overrides})


def by_name(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in body["settings"]}


class TestGetSettings:
    async def test_catalog_covers_every_overridable_with_groups_and_flags(
        self, app: FastAPI
    ) -> None:
        response = await get_catalog(app)
        assert response.status_code == 200
        body = response.json()
        settings = by_name(body)
        assert set(settings) == set(OVERRIDABLE)
        assert settings["RETRIEVAL_METHOD"]["group"] == "retrieval"
        assert settings["CHAT_MODEL_TYPE"]["group"] == "generation"
        assert settings["CHUNKING_STRATEGY"]["group"] == "ingestion"
        assert settings["CHUNKING_STRATEGY"]["reingest_affecting"] is True
        assert settings["TOP_K"]["reingest_affecting"] is False
        assert body["corpus_stale"] is False

    async def test_choices_reflect_the_registries(self, app: FastAPI) -> None:
        settings = by_name((await get_catalog(app)).json())
        assert "reranked" in settings["RETRIEVAL_METHOD"]["choices"]
        assert len(settings["CHUNKING_STRATEGY"]["choices"]) == 5
        assert settings["TOP_K"]["choices"] is None

    async def test_values_are_typed_json_scalars(self, app: FastAPI) -> None:
        settings = by_name((await get_catalog(app)).json())
        assert isinstance(settings["TOP_K"]["value"], int)
        assert isinstance(settings["RERANK_ENABLED"]["value"], bool)
        assert isinstance(settings["SEMANTIC_WEIGHT"]["value"], float)
        assert isinstance(settings["RETRIEVAL_METHOD"]["value"], str)


class TestPatchSettings:
    async def test_patch_applies_persists_and_reports(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        base = get_settings().TOP_K
        response = await patch_settings(app, {"TOP_K": base + 15})
        assert response.status_code == 200
        setting = by_name(response.json())["TOP_K"]
        assert setting["value"] == base + 15
        assert setting["overridden"] is True
        assert base + 15 == get_settings().TOP_K  # the next question sees it
        assert store_state["overrides"] == {"TOP_K": str(base + 15)}

    async def test_bool_override_round_trips(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        response = await patch_settings(app, {"RERANK_ENABLED": True})
        assert by_name(response.json())["RERANK_ENABLED"]["value"] is True
        assert store_state["overrides"]["RERANK_ENABLED"] == "true"
        assert get_settings().RERANK_ENABLED is True

    async def test_clearing_an_override_restores_the_env_value(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        base = get_settings().TOP_K
        await patch_settings(app, {"TOP_K": base + 15})
        response = await patch_settings(app, {"TOP_K": None})
        setting = by_name(response.json())["TOP_K"]
        assert setting["value"] == base
        assert setting["overridden"] is False
        assert store_state["overrides"] == {}
        assert base == get_settings().TOP_K

    async def test_unknown_setting_is_a_structured_422(self, app: FastAPI) -> None:
        response = await patch_settings(app, {"POSTGRES_PASSWORD": "nope"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unknown_setting"

    async def test_invalid_value_changes_nothing(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        base = get_settings().TOP_K
        response = await patch_settings(app, {"TOP_K": 0})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "invalid_settings"
        assert "TOP_K" in error["message"]
        assert base == get_settings().TOP_K
        assert store_state["overrides"] == {}

    async def test_weight_pair_validates_as_a_whole(self, app: FastAPI) -> None:
        lone = await patch_settings(app, {"SEMANTIC_WEIGHT": 0.6})
        assert lone.status_code == 422
        assert "sum to 1.0" in lone.json()["error"]["message"]
        pair = await patch_settings(app, {"SEMANTIC_WEIGHT": 0.6, "BM25_WEIGHT": 0.4})
        assert pair.status_code == 200
        settings = by_name(pair.json())
        assert settings["SEMANTIC_WEIGHT"]["value"] == 0.6
        assert settings["BM25_WEIGHT"]["value"] == 0.4

    async def test_chat_model_type_accepts_llm_aliases_only(self, app: FastAPI) -> None:
        ok = await patch_settings(app, {"CHAT_MODEL_TYPE": "reasoning"})
        assert ok.status_code == 200
        bad = await patch_settings(app, {"CHAT_MODEL_TYPE": "embedding"})
        assert bad.status_code == 422
        assert bad.json()["error"]["code"] == "invalid_settings"


class TestCorpusStale:
    async def test_reingest_affecting_change_flags_stale(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        response = await patch_settings(app, {"CHUNKING_STRATEGY": "token_based"})
        assert response.status_code == 200
        assert response.json()["corpus_stale"] is True
        assert store_state["stale"] is True

    async def test_query_time_change_does_not_flag_stale(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        response = await patch_settings(app, {"TOP_K": get_settings().TOP_K + 2})
        assert response.json()["corpus_stale"] is False
        assert store_state["stale"] is False

    async def test_no_op_reingest_value_does_not_flag_stale(
        self, app: FastAPI, store_state: dict[str, Any]
    ) -> None:
        """Overriding to the already-effective value changes nothing."""
        current = get_settings().CHUNKING_STRATEGY
        response = await patch_settings(app, {"CHUNKING_STRATEGY": current})
        assert response.json()["corpus_stale"] is False
        assert store_state["stale"] is False

    async def test_empty_corpus_never_goes_stale(self, store_state: dict[str, Any]) -> None:
        application = create_app()
        application.dependency_overrides[get_app_settings_store] = lambda: FakeAppSettingsStore(
            store_state
        )
        application.dependency_overrides[get_vector_store] = lambda: FakeVectorStore(n_documents=0)
        response = await patch_settings(application, {"CHUNKING_STRATEGY": "token_based"})
        assert response.status_code == 200
        assert response.json()["corpus_stale"] is False
        assert store_state["stale"] is False
