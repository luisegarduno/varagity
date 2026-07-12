"""FastAPI app factory: CORS, lifespan, structured errors, routers.

``create_app`` is served by a single uvicorn worker (plan decision #11):
``uvicorn varagity.api.main:create_app --factory``. The lifespan runs the
idempotent migration runner (so an existing ``pgdata`` volume gains the v2
tables without ``down -v``) and warms the registries; every error response
carries the ``{error: {code, message}}`` envelope the GUI banners on.

Importing this module imports :mod:`varagity.pipeline` (via the chat
route), which exports ``PREFECT_API_URL`` before ``prefect`` loads — the
same ordering invariant the CLI honors.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Any

import psycopg
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic.json_schema import models_json_schema
from starlette.exceptions import HTTPException as StarletteHTTPException

from varagity.api.routes import chat, conversations, system
from varagity.api.schemas import (
    DeltaEvent,
    DoneEvent,
    ErrorBody,
    ErrorEvent,
    ErrorResponse,
    RetrievalEvent,
)
from varagity.config import get_settings
from varagity.logging_setup import setup_logging
from varagity.stores.migrate import run_migrations
from varagity.stores.vector_store import default_conninfo

logger = logging.getLogger(__name__)

# The SSE chat protocol's event payloads (spec_v2 §4.3). No route returns
# them directly (they ride inside the event stream), so FastAPI would omit
# them from the OpenAPI schema — and the web app's generated TypeScript
# types are the *whole* wire contract, SSE payloads included (schemas.py).
# create_app() merges their JSON schemas into components/schemas.
_SSE_EVENT_MODELS = (RetrievalEvent, DeltaEvent, DoneEvent, ErrorEvent)

# Stable codes for statuses raised with a plain-string detail; handlers fall
# back to http_<status> beyond these.
_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    500: "internal_error",
    503: "service_unavailable",
}


def _run_startup_migrations() -> None:
    """Apply pending migrations, tolerating an unreachable database.

    In compose the ``api`` service starts only after postgres is healthy,
    so unreachability here means a host-mode run without the stack — the
    API still serves ``/api/health`` and ``/api/config``, and the chat
    preflight reports postgres down with a structured 503. Any *SQL*
    failure still propagates: serving with a half-applied schema would be
    worse than not starting.
    """
    try:
        with psycopg.connect(default_conninfo(), autocommit=True) as conn:
            applied = run_migrations(conn)
    except psycopg.OperationalError as error:
        logger.error(
            "skipping migrations — postgres unreachable at startup (%s); "
            "conversation persistence will 503 until it returns",
            error,
        )
        return
    if applied:
        logger.info("applied %d migration(s): %s", len(applied), ", ".join(applied))


def _warm_registries() -> None:
    """Import the registry packages so their implementations self-register.

    The route modules already import them transitively; doing it here makes
    startup fail loudly if a registration import breaks, instead of at the
    first request.
    """
    import varagity.chunking  # noqa: F401
    import varagity.ingest.parsers  # noqa: F401
    import varagity.retrieval  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup work before the app serves: registries, then migrations.

    Args:
        app: The FastAPI instance being started.

    Yields:
        Nothing — control for the app's serving lifetime.
    """
    _warm_registries()
    await run_in_threadpool(_run_startup_migrations)
    yield


def _error_payload(code: str, message: str) -> dict[str, object]:
    """Build the structured error envelope as a plain dict.

    Args:
        code: Stable, machine-readable code.
        message: Human-readable detail.

    Returns:
        The ``{"error": {"code", "message"}}`` body.
    """
    return ErrorResponse(error=ErrorBody(code=code, message=message)).model_dump()


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Render any HTTPException in the structured envelope.

    Routes raise ``HTTPException(detail={"code": …, "message": …})`` for
    domain errors; framework-raised exceptions (404 on an unknown path,
    405, …) carry a plain-string detail and get a status-derived code.

    Args:
        request: The offending request (unused; handler signature).
        exc: The raised exception (a Starlette ``HTTPException``).

    Returns:
        The enveloped JSON response.
    """
    assert isinstance(exc, StarletteHTTPException)  # registered for exactly this type
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        payload = _error_payload(str(detail["code"]), str(detail["message"]))
    else:
        code = _STATUS_CODES.get(exc.status_code, f"http_{exc.status_code}")
        payload = _error_payload(code, str(detail))
    return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Render request-validation failures in the structured envelope.

    Args:
        request: The offending request (unused; handler signature).
        exc: The raised ``RequestValidationError``.

    Returns:
        The enveloped 422 response; the field-level errors ride along under
        ``error.message`` in FastAPI's standard rendering.
    """
    assert isinstance(exc, RequestValidationError)  # registered for exactly this type
    return JSONResponse(
        status_code=422,
        content=_error_payload("validation_error", str(exc.errors())),
    )


def _install_sse_event_schemas(app: FastAPI) -> None:
    """Publish the SSE event payload models in the app's OpenAPI schema.

    The chat protocol's payloads (:data:`_SSE_EVENT_MODELS`, plus whatever
    they reference — ``RetrievedChunk``, ``RetrievalTrace``, ``UsageInfo``)
    cross the wire inside ``text/event-stream`` frames, which FastAPI's
    route inspection never sees. Merging their JSON schemas into
    ``components/schemas`` keeps the generated TypeScript types
    (``openapi-typescript``) covering the *entire* contract, so the web
    app's SSE handling can't drift by hand-editing either.

    Args:
        app: The application whose ``openapi()`` gets the merged schema.
    """

    def openapi_with_sse_events() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        _, definitions = models_json_schema(
            [(model, "serialization") for model in _SSE_EVENT_MODELS],
            ref_template="#/components/schemas/{model}",
        )
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, model_schema in definitions.get("$defs", {}).items():
            components.setdefault(name, model_schema)
        app.openapi_schema = schema
        return schema

    app.openapi = openapi_with_sse_events  # type: ignore[method-assign]


def create_app() -> FastAPI:
    """Build the configured FastAPI application.

    Returns:
        The app, ready for uvicorn (``--factory``).
    """
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    try:
        version = metadata.version("varagity")
    except metadata.PackageNotFoundError:  # running from a raw checkout
        version = "0.0.0"
    app = FastAPI(
        title="Varagity API",
        description="Contextual-Retrieval RAG over local GPUs (spec_v2 §4).",
        version=version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.include_router(system.router)
    app.include_router(conversations.router)
    app.include_router(chat.router)
    _install_sse_event_schemas(app)
    return app
