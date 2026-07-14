"""Route modules of the HTTP API (spec_v2 §4.1).

One module per surface: ``chat`` (SSE streaming), ``conversations``
(CRUD), ``system`` (health + capabilities), ``metrics`` (the Prometheus
scrape target, spec_v2 §6). The corpus (``documents``, ``ingest``) and
``settings`` routes land with their GUI in Phase 8.
"""
