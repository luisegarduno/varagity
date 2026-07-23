"""System routes: liveness/dependency health and static capabilities.

``GET /api/health`` reports per-service reachability; ``GET /api/config``
exposes the registries and valid ranges the GUI builds its controls from
(spec_v2 §4.2) — a newly registered retriever/chunker/parser appears here,
and therefore in the UI, with zero API edits.
"""

from fastapi import APIRouter

from varagity.api.deps import check_services
from varagity.api.schemas import ConfigResponse, HealthResponse, NumericRange
from varagity.chunking import CHUNKER_REGISTRY
from varagity.config import get_settings
from varagity.ingest.parsers.pdf import OCR_ENGINE_FACTORIES
from varagity.models.registry import LLM_MODEL_TYPES, MODEL_TYPES
from varagity.retrieval.base import RETRIEVER_REGISTRY

router = APIRouter(tags=["system"])

ALL_SERVICES = ("llamacpp", "infinity", "postgres", "elasticsearch", "prefect")

# Valid ranges of the numeric query-time knobs (spec_v2 §4.2 "valid
# ranges"), matching the config.py validators.
_RANGES: dict[str, NumericRange] = {
    "top_k": NumericRange(min=1),
    "rerank_top_n": NumericRange(min=1),
    "rerank_candidates": NumericRange(min=1),
    "hyde_max_tokens": NumericRange(min=1),
    "hyde_max_chars": NumericRange(min=1),
    "semantic_weight": NumericRange(min=0.0, max=1.0),
    "bm25_weight": NumericRange(min=0.0, max=1.0),
    "llm_temperature": NumericRange(min=0.0, max=2.0),
    "max_tokens": NumericRange(min=1),
    "condense_history_turns": NumericRange(min=0),
    "condense_max_tokens": NumericRange(min=1),
    "condense_max_chars": NumericRange(min=1),
    "chunk_size": NumericRange(min=1),
    "chunk_overlap": NumericRange(min=0),
    "verbose": NumericRange(min=0, max=2),
}


@router.get("/api/health")
async def health() -> HealthResponse:
    """Report API liveness and per-dependency reachability.

    Always ``200`` while the API process is alive — the compose healthcheck
    gates on the process, and per-dependency state lives in the body so a
    flapping backing service doesn't flap the ``api`` container.

    Returns:
        The aggregate flag and each service's probe outcome.
    """
    services = await check_services(ALL_SERVICES)
    return HealthResponse(ok=all(s.ok for s in services.values()), services=services)


@router.get("/api/config")
def config() -> ConfigResponse:
    """Expose the registered capabilities and valid ranges.

    Mostly static registry contents; the two upload constraints
    (``upload_max_mb``, ``allowed_extensions``) are read from the effective
    settings per request so the dropzone always validates against what the
    server enforces (``ALLOWED_EXTENSIONS`` is runtime-overridable).

    Returns:
        Registry contents (retrievers, chunkers, OCR engines, model types),
        the numeric knobs' valid ranges, and the upload constraints.
    """
    settings = get_settings()
    return ConfigResponse(
        retrievers=sorted(RETRIEVER_REGISTRY),
        chunkers=sorted(CHUNKER_REGISTRY),
        ocr_engines=sorted(OCR_ENGINE_FACTORIES),
        model_types=list(MODEL_TYPES),
        llm_model_types=list(LLM_MODEL_TYPES),
        ranges=_RANGES,
        upload_max_mb=settings.UPLOAD_MAX_MB,
        allowed_extensions=sorted(settings.allowed_extension_set),
        preview_enabled=settings.PREVIEW_ENABLED,
    )
