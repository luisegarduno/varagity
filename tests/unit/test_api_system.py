"""Unit tests for the system routes and the structured error envelope."""

from typing import Any

import httpx
import pytest
from fastapi import FastAPI

import varagity.api.deps as deps
from varagity.api.main import create_app
from varagity.api.schemas import ServiceHealth


@pytest.fixture
def app() -> FastAPI:
    return create_app()


async def get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.get(path)


class TestConfig:
    async def test_lists_every_registry_including_reranked(self, app: FastAPI) -> None:
        response = await get(app, "/api/config")
        assert response.status_code == 200
        data = response.json()
        assert data["retrievers"] == ["bm25", "hybrid", "reranked", "semantic"]
        assert data["chunkers"] == [
            "docling_hybrid",
            "markdown_aware",
            "recursive_character",
            "semantic",
            "token_based",
        ]
        assert data["ocr_engines"] == ["easyocr", "tesseract"]
        assert data["model_types"] == ["default", "embedding", "rerank", "reasoning", "tool"]

    async def test_ranges_cover_the_query_time_knobs(self, app: FastAPI) -> None:
        ranges = (await get(app, "/api/config")).json()["ranges"]
        assert ranges["top_k"]["min"] == 1
        assert ranges["llm_temperature"] == {"min": 0.0, "max": 2.0}
        assert ranges["verbose"] == {"min": 0, "max": 2}
        assert ranges["semantic_weight"] == {"min": 0.0, "max": 1.0}
        assert ranges["bm25_weight"] == {"min": 0.0, "max": 1.0}

    async def test_llm_model_types_and_upload_constraints(
        self, app: FastAPI, settings_env: Any
    ) -> None:
        """The composer's model-type vocabulary + the dropzone's limits."""
        settings_env(UPLOAD_MAX_MB=7, ALLOWED_EXTENSIONS=".md,txt")
        data = (await get(app, "/api/config")).json()
        assert data["llm_model_types"] == ["default", "reasoning", "tool"]
        assert data["upload_max_mb"] == 7
        assert data["allowed_extensions"] == [".md", ".txt"]  # normalized + sorted

    async def test_preview_flag_mirrors_the_kill_switch(
        self, app: FastAPI, settings_env: Any
    ) -> None:
        """The read-only PREVIEW_ENABLED surface (ADR-010; the knob is env-only)."""
        assert (await get(app, "/api/config")).json()["preview_enabled"] is True
        settings_env(PREVIEW_ENABLED="false")
        assert (await get(app, "/api/config")).json()["preview_enabled"] is False


class TestHealth:
    async def test_reports_every_service_with_probe_outcomes(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_check(names: tuple[str, ...]) -> dict[str, ServiceHealth]:
            return {
                name: ServiceHealth(ok=(name != "elasticsearch"), detail=None) for name in names
            }

        monkeypatch.setattr("varagity.api.routes.system.check_services", fake_check)
        response = await get(app, "/api/health")
        assert response.status_code == 200
        data = response.json()
        assert set(data["services"]) == {
            "llamacpp",
            "infinity",
            "postgres",
            "elasticsearch",
            "prefect",
        }
        assert data["ok"] is False  # one probe failed
        assert data["services"]["elasticsearch"]["ok"] is False

    async def test_ok_when_every_probe_passes(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def all_up(names: tuple[str, ...]) -> dict[str, ServiceHealth]:
            return {name: ServiceHealth(ok=True) for name in names}

        monkeypatch.setattr("varagity.api.routes.system.check_services", all_up)
        assert (await get(app, "/api/health")).json()["ok"] is True

    async def test_probes_run_against_unreachable_hosts_report_down(
        self, app: FastAPI, settings_env: Any
    ) -> None:
        """Real probe code path: nothing listens on these ports."""
        settings_env(
            BASE_MODEL_API_URL="http://127.0.0.1:1/v1",
            EMBEDDING_API_URL="http://127.0.0.1:1/v1",
            ELASTICSEARCH_URL="http://127.0.0.1:1",
            PREFECT_API_URL="http://127.0.0.1:1/api",
            POSTGRES_HOST="127.0.0.1",
            POSTGRES_PORT=1,
        )
        data = (await get(app, "/api/health")).json()
        assert data["ok"] is False
        assert all(not service["ok"] for service in data["services"].values())
        assert all(service["detail"] for service in data["services"].values())


class TestChatPreflight:
    async def test_es_down_gives_structured_503_before_the_stream(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def es_down(names: tuple[str, ...]) -> dict[str, ServiceHealth]:
            return {
                name: ServiceHealth(
                    ok=(name != "elasticsearch"),
                    detail="ConnectError: refused" if name == "elasticsearch" else None,
                )
                for name in names
            }

        monkeypatch.setattr(deps, "check_services", es_down)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            response = await client.post("/api/chat", json={"query": "q"})
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        error = response.json()["error"]
        assert error["code"] == "es_unreachable"
        assert "unreachable" in error["message"]


class TestErrorEnvelope:
    async def test_unknown_path_is_enveloped(self, app: FastAPI) -> None:
        response = await get(app, "/api/nope")
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found", "message": "Not Found"}}

    async def test_unhandled_exception_is_enveloped_and_cors_readable(self, app: FastAPI) -> None:
        """Unhandled exceptions must stay browser-readable cross-origin.

        Without CORS headers on the 500, a cross-origin fetch can only
        report "TypeError: Failed to fetch" and the real error is
        invisible (the upload-permission bug's shape).
        """

        @app.get("/api/boom")
        def boom() -> None:
            raise RuntimeError("the disk fell off")

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            response = await client.get("/api/boom", headers={"origin": "http://localhost:3000"})
        assert response.status_code == 500
        error = response.json()["error"]
        assert error["code"] == "internal_error"
        assert "the disk fell off" in error["message"]
        # The part the browser needs: the 500 passed through the CORS
        # middleware's send path, so it is readable cross-origin.
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    async def test_validation_failure_is_enveloped(self, app: FastAPI) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
            response = await client.post("/api/chat", json={})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert "query" in error["message"]

    async def test_openapi_schema_is_served(self, app: FastAPI) -> None:
        response = await get(app, "/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["openapi"].startswith("3.")
        assert "/api/chat" in schema["paths"]
        assert "/api/conversations/{conversation_id}" in schema["paths"]

    async def test_openapi_schema_carries_the_sse_event_payloads(self, app: FastAPI) -> None:
        """The stream payloads are in components/schemas (generated TS types)."""
        schema = (await get(app, "/openapi.json")).json()
        components = schema["components"]["schemas"]
        for name in (
            "RetrievalEvent",
            "DeltaEvent",
            "DoneEvent",
            "ErrorEvent",
            "RetrievedChunk",
            "RetrievalTrace",
            "UsageInfo",
        ):
            assert name in components, f"SSE payload model {name} missing from OpenAPI"
        trace = components["RetrievalTrace"]["properties"]
        assert {"semantic_rank", "bm25_rank", "fused_rank", "rerank_delta", "final_rank"} <= set(
            trace
        )
