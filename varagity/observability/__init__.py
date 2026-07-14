"""In-app observability: the Prometheus metric catalog (spec_v2 §6).

One module, :mod:`varagity.observability.metrics`, holds the spec_v2 §6.2
catalog plus the recording helpers the pipeline probe points call. The
API's ``GET /metrics`` route exposes the same process-wide registry.
"""
