"""The HTTP API (spec_v2 §4.1): FastAPI over the same flows the CLI runs.

The API lives inside the package so it imports the pipeline directly — no
network hop, and the "app is a client of every backing service" invariant
holds with a second front door. ``async`` at the edge, sync flows
underneath: endpoints run the synchronous pipeline in a threadpool while
the event loop streams SSE frames out.

Import :func:`create_app` lazily from :mod:`varagity.api.main` (uvicorn's
``--factory`` target) — importing it pulls FastAPI and the whole pipeline.
"""
