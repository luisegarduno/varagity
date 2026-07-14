"""``GET /metrics`` — the Prometheus scrape target (spec_v2 §6).

Serves the process-wide default registry (the spec_v2 §6.2 catalog in
:mod:`varagity.observability.metrics`) in the Prometheus text exposition
format. A plain route rather than a mounted ``make_asgi_app()``: mounting
307-redirects the bare ``/metrics`` path (Starlette mount semantics), and
``generate_latest`` over the default registry is exactly what that ASGI
app serves anyway — one process, one registry, single uvicorn worker
(plan decision #11, no multiprocess registry).

``settings.METRICS_ENABLED`` gates inclusion of this router at app build
time (:func:`varagity.api.main.create_app`), so disabling metrics turns
the endpoint into a structured 404.
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

router = APIRouter(tags=["observability"])


@router.get("/metrics")
def metrics() -> Response:
    """Expose the metric catalog for a Prometheus scrape.

    Returns:
        The default registry rendered in the Prometheus text format
        (``text/plain; version=0.0.4``).
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
