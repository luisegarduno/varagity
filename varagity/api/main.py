"""FastAPI app factory: CORS, lifespan, structured errors, routers.

``create_app`` is served by a single uvicorn worker (plan decision #11):
``uvicorn varagity.api.main:create_app --factory``. The lifespan warms the
registries, runs the idempotent migration runner (so an existing ``pgdata``
volume gains the v2 tables without ``down -v``), and replays persisted
runtime setting overrides (spec_v2 §4.7); every error response carries the
``{error: {code, message}}`` envelope the GUI banners on.

Importing this module imports :mod:`varagity.pipeline` (via the chat
route), which exports ``PREFECT_API_URL`` before ``prefect`` loads — the
same ordering invariant the CLI honors.
"""

import logging
from collections.abc import AsyncIterator, Callable
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

from varagity.api import runtime_settings
from varagity.api.routes import (
    chat,
    conversations,
    documents,
    ingest,
    metrics,
    system,
)
from varagity.api.routes import settings as settings_routes
from varagity.api.schemas import (
    DeltaEvent,
    DoneEvent,
    ErrorBody,
    ErrorEvent,
    ErrorResponse,
    IngestLogEvent,
    IngestProgressEvent,
    IngestStatusEvent,
    RetrievalEvent,
)
from varagity.config import get_settings
from varagity.logging_setup import setup_logging
from varagity.stores.app_settings_store import AppSettingsStore
from varagity.stores.migrate import run_migrations
from varagity.stores.vector_store import default_conninfo

logger = logging.getLogger(__name__)

# The SSE protocols' event payloads (spec_v2 §4.3 chat + §4.2 ingest
# status). No route returns them directly (they ride inside the event
# streams), so FastAPI would omit them from the OpenAPI schema — and the
# web app's generated TypeScript types are the *whole* wire contract, SSE
# payloads included (schemas.py). create_app() merges their JSON schemas
# into components/schemas.
_SSE_EVENT_MODELS = (
    RetrievalEvent,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    IngestStatusEvent,
    IngestProgressEvent,
    IngestLogEvent,
)

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


def _load_runtime_overrides() -> None:
    """Replay persisted setting overrides, tolerating an unreachable database.

    Mirrors :func:`_run_startup_migrations`' posture: unreachability means a
    host-mode run without the stack — the API serves on env defaults and
    the settings routes 503 until postgres returns. Invalid persisted rows
    are dropped-with-a-log inside the loader (the API must boot).
    """
    try:
        with AppSettingsStore() as store:
            overrides = store.load_overrides()
    except psycopg.OperationalError as error:
        logger.error(
            "skipping persisted setting overrides — postgres unreachable at startup (%s)", error
        )
        return
    runtime_settings.load_persisted_overrides(lambda: overrides)


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
    # After migrations (the app_settings table must exist) and before the
    # first request: overrides survive an api restart (spec_v2 §4.7).
    await run_in_threadpool(_load_runtime_overrides)
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


class StructuredServerErrors:
    """Render unhandled route exceptions as the structured 500 envelope.

    Starlette's stack is ``ServerErrorMiddleware → user middleware (CORS)
    → ExceptionMiddleware → routes``: an exception no handler catches
    propagates *past* the CORS middleware and becomes a bare text 500
    **without** ``Access-Control-Allow-Origin`` — which a browser can only
    report as ``TypeError: Failed to fetch``, hiding the real error (the
    exact failure mode of the first unwritable-``./docs`` upload). This
    pure-ASGI middleware is registered *inside* CORS, so its enveloped 500
    flows through the CORS send path and stays readable cross-origin —
    "errors are structured" (spec_v2 §4.1) holds for every failure.

    A response whose headers already flushed (an SSE stream mid-flight)
    can't change status; those re-raise, and the stream protocols carry
    their own in-band ``error`` events instead.
    """

    def __init__(self, app: Callable[..., Any]) -> None:
        """Wrap the downstream ASGI app.

        Args:
            app: The next ASGI callable in the stack.
        """
        self.app = app

    async def __call__(
        self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]
    ) -> None:
        """Serve one connection, enveloping any unhandled exception.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as error:
            if response_started:
                raise
            logger.exception("unhandled error serving %s", scope.get("path", "?"))
            response = JSONResponse(
                status_code=500,
                content=_error_payload("internal_error", f"{type(error).__name__}: {error}"),
            )
            await response(scope, receive, send)  # type: ignore[arg-type]


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
    # Order matters: add_middleware prepends (last added = outermost), so
    # registering the error enveloper *before* CORS seats it inside — its
    # 500s pass through the CORS send path and stay browser-readable.
    app.add_middleware(StructuredServerErrors)
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
    app.include_router(settings_routes.router)
    app.include_router(documents.router)
    app.include_router(ingest.router)
    if settings.METRICS_ENABLED:
        app.include_router(metrics.router)
    _install_sse_event_schemas(app)
    return app
