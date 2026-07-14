"""FastAPI dependencies: providers, health probes, and the chat preflight.

Providers are deliberately thin functions so tests swap them via
``app.dependency_overrides`` — the same seam the flows expose with their
injectable ``retriever``/``llm`` parameters. The health probes back both
``GET /api/health`` (report) and the chat preflight (clean ``503`` with a
machine-readable code *before* the SSE stream opens — once the 200 headers
flush, status can't change; spec_v2 plan decision #7).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterator

import httpx
import psycopg
from fastapi import HTTPException

from varagity.api.schemas import ServiceHealth
from varagity.config import get_settings
from varagity.models.llm import LLMClient
from varagity.models.registry import get_model
from varagity.observability import metrics
from varagity.retrieval import get_retriever
from varagity.retrieval.base import Retriever
from varagity.stores.app_settings_store import AppSettingsStore
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.conversation_store import ConversationStore
from varagity.stores.vector_store import ContextualVectorDB, default_conninfo

logger = logging.getLogger(__name__)

# Local/LAN services: a probe that needs longer than this is down for
# interactive purposes.
PROBE_TIMEOUT_SECONDS = 3.0

# Services the chat pipeline cannot answer without. Prefect is deliberately
# absent: with no server reachable, flows fall back to an ephemeral
# in-process API (see varagity.pipeline) — degraded tracking, working chat.
CHAT_REQUIRED_SERVICES = ("postgres", "elasticsearch", "llamacpp", "infinity")


def get_llm() -> LLMClient:
    """Provide the chat LLM client (override seam for tests).

    Returns:
        The registry client for ``settings.CHAT_MODEL_TYPE`` (an LLM alias —
        the config validator guarantees it's never ``embedding``/``rerank``,
        so this is always an :class:`LLMClient`).
    """
    client = get_model(get_settings().CHAT_MODEL_TYPE)
    assert isinstance(client, LLMClient)  # CHAT_MODEL_TYPE is validated to the LLM aliases
    return client


def get_retriever_resolver() -> Callable[[str], Retriever]:
    """Provide the retrieval-method resolver (override seam for tests).

    Returns:
        :func:`varagity.retrieval.get_retriever`.
    """
    return get_retriever


def get_conversation_store() -> Iterator[ConversationStore]:
    """Provide a per-request conversation store, closed after the response.

    Yields:
        A connected store.

    Raises:
        HTTPException: ``503 postgres_unreachable`` when the database is
            down (the structured-error shape the GUI banners on).
    """
    try:
        store = ConversationStore()
    except psycopg.OperationalError as error:
        raise _unreachable("postgres", str(error)) from error
    try:
        yield store
    finally:
        store.close()


def get_app_settings_store() -> Iterator[AppSettingsStore]:
    """Provide a per-request app-settings store, closed after the response.

    Yields:
        A connected store.

    Raises:
        HTTPException: ``503 postgres_unreachable`` when the database is
            down.
    """
    try:
        store = AppSettingsStore()
    except psycopg.OperationalError as error:
        raise _unreachable("postgres", str(error)) from error
    try:
        yield store
    finally:
        store.close()


def get_vector_store() -> Iterator[ContextualVectorDB]:
    """Provide a per-request vector store, closed after the response.

    Backs the corpus routes (document list/count/delete — spec_v2 §4.2).

    Yields:
        A connected store.

    Raises:
        HTTPException: ``503 postgres_unreachable`` when the database is
            down.
    """
    try:
        store = ContextualVectorDB()
    except psycopg.OperationalError as error:
        raise _unreachable("postgres", str(error)) from error
    try:
        yield store
    finally:
        store.close()


def get_bm25_store() -> Iterator[ElasticsearchBM25]:
    """Provide a per-request BM25 store, closed after the response.

    The Elasticsearch client performs no I/O at construction, so
    unreachability surfaces on the operation itself — routes map those
    failures to the structured ``503 es_unreachable``.

    Yields:
        A configured store.
    """
    store = ElasticsearchBM25()
    try:
        yield store
    finally:
        store.close()


def get_conversation_store_factory() -> Callable[[], ConversationStore]:
    """Provide a store *factory* for routes that connect off the event loop.

    The chat route persists from a worker thread after the stream ends —
    a per-request dependency handle would pin the connection open for the
    stream's whole lifetime instead of the write's.

    Returns:
        A zero-argument :class:`ConversationStore` constructor.
    """
    return ConversationStore


async def _probe_http(
    client: httpx.AsyncClient, url: str, *, headers: dict[str, str] | None = None
) -> ServiceHealth:
    """Probe one HTTP dependency.

    Args:
        client: The shared async client (carries the timeout).
        url: Probe URL — a cheap authenticated-or-open GET.
        headers: Optional auth headers.

    Returns:
        The probe outcome.
    """
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    except httpx.HTTPError as error:
        return ServiceHealth(ok=False, detail=f"{type(error).__name__}: {error}")
    return ServiceHealth(ok=True)


def _probe_postgres() -> ServiceHealth:
    """Probe PostgreSQL with a short-timeout connect + ``SELECT 1``.

    Returns:
        The probe outcome.
    """
    try:
        with psycopg.connect(
            default_conninfo(), connect_timeout=int(PROBE_TIMEOUT_SECONDS)
        ) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as error:
        return ServiceHealth(ok=False, detail=f"{type(error).__name__}: {error}")
    return ServiceHealth(ok=True)


async def check_services(names: tuple[str, ...]) -> dict[str, ServiceHealth]:
    """Probe the named backing services concurrently.

    Args:
        names: Service names among ``llamacpp``, ``infinity``, ``postgres``,
            ``elasticsearch``, ``prefect``.

    Returns:
        Probe outcome per requested name.

    Raises:
        KeyError: If a name is not a known service.
    """
    settings = get_settings()
    # /v1/models is the OpenAI surface's cheapest GET; llama.cpp's and
    # infinity's root /health endpoints sit outside the configured /v1
    # base URLs, so probing through them avoids URL surgery.
    urls: dict[str, tuple[str, dict[str, str] | None]] = {
        "llamacpp": (f"{settings.BASE_MODEL_API_URL.rstrip('/')}/models", None),
        "infinity": (
            f"{settings.EMBEDDING_API_URL.rstrip('/')}/models",
            {"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"},
        ),
        "elasticsearch": (f"{settings.ELASTICSEARCH_URL.rstrip('/')}/_cluster/health", None),
        "prefect": (f"{settings.PREFECT_API_URL.rstrip('/')}/health", None),
    }

    async def run_one(name: str, client: httpx.AsyncClient) -> ServiceHealth:
        if name == "postgres":
            return await asyncio.to_thread(_probe_postgres)
        url, headers = urls[name]
        return await _probe_http(client, url, headers=headers)

    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
        results = await asyncio.gather(*(run_one(name, client) for name in names))
    statuses = dict(zip(names, results, strict=True))
    # Every probe refreshes the reachability gauge (spec_v2 §6.2); in
    # compose the api container's own healthcheck calls /api/health every
    # 15 s, so the gauge stays current without a dedicated poller.
    for name, health in statuses.items():
        metrics.set_dependency_up(name, health.ok)
    return statuses


def _unreachable(service: str, detail: str) -> HTTPException:
    """Build the structured 503 for one down service.

    Args:
        service: The service name (becomes the ``<service>_unreachable``
            code, e.g. ``es_unreachable`` — spec_v2 §4.3's example).
        detail: Probe failure detail.

    Returns:
        The exception carrying the ``{code, message}`` detail dict.
    """
    code_name = "es" if service == "elasticsearch" else service
    return HTTPException(
        status_code=503,
        detail={
            "code": f"{code_name}_unreachable",
            "message": f"{service} unreachable — is the stack up? ({detail})",
        },
    )


async def require_chat_services() -> None:
    """Chat preflight: 503 with a machine-readable code if a dependency is down.

    Runs before ``POST /api/chat`` opens its stream, so an outage surfaces
    as a clean structured error instead of a half-stream.

    Raises:
        HTTPException: ``503`` naming the first unreachable service.
    """
    statuses = await check_services(CHAT_REQUIRED_SERVICES)
    for name in CHAT_REQUIRED_SERVICES:
        status = statuses[name]
        if not status.ok:
            raise _unreachable(name, status.detail or "no detail")


def get_services_preflight() -> Callable[[], Awaitable[None]]:
    """Provide the chat reachability preflight (override seam for tests).

    Returns:
        :func:`require_chat_services`.
    """
    return require_chat_services


async def require_ingest_services() -> None:
    """Ingest preflight: 503 with a machine-readable code if a dependency is down.

    Ingestion needs both stores and the embedder; the chat LLM joins only
    when ``CONTEXTUALIZE`` is on (the identity path never calls it). Runs
    before ``POST /api/ingest`` spawns the background run, so an outage is
    a clean structured error instead of an instantly failed run.

    Raises:
        HTTPException: ``503`` naming the first unreachable service.
    """
    required: tuple[str, ...] = ("postgres", "elasticsearch", "infinity")
    if get_settings().CONTEXTUALIZE:
        required += ("llamacpp",)
    statuses = await check_services(required)
    for name in required:
        status = statuses[name]
        if not status.ok:
            raise _unreachable(name, status.detail or "no detail")


def get_ingest_preflight() -> Callable[[], Awaitable[None]]:
    """Provide the ingest reachability preflight (override seam for tests).

    Returns:
        :func:`require_ingest_services`.
    """
    return require_ingest_services
