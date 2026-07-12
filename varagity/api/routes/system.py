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
from varagity.ingest.parsers.pdf import OCR_ENGINE_FACTORIES
from varagity.models.registry import MODEL_TYPES
from varagity.retrieval.base import RETRIEVER_REGISTRY

router = APIRouter(tags=["system"])

ALL_SERVICES = ("llamacpp", "infinity", "postgres", "elasticsearch", "prefect")

# Valid ranges of the numeric query-time knobs (spec_v2 §4.2 "valid
# ranges"), matching the config.py validators.
_RANGES: dict[str, NumericRange] = {
    "top_k": NumericRange(min=1),
    "rerank_top_n": NumericRange(min=1),
    "rerank_candidates": NumericRange(min=1),
    "llm_temperature": NumericRange(min=0.0, max=2.0),
    "max_tokens": NumericRange(min=1),
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

    Returns:
        Registry contents (retrievers, chunkers, OCR engines, model types)
        plus the numeric knobs' valid ranges.
    """
    return ConfigResponse(
        retrievers=sorted(RETRIEVER_REGISTRY),
        chunkers=sorted(CHUNKER_REGISTRY),
        ocr_engines=sorted(OCR_ENGINE_FACTORIES),
        model_types=list(MODEL_TYPES),
        ranges=_RANGES,
    )
