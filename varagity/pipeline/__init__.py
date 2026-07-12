"""Prefect orchestration layer: thin ``@flow``/``@task`` adapters (spec §9–10).

Business logic lives in the plain modules (``ingest.loader``, ``retrieval``,
``generation``); this package wraps their stage functions as tracked,
retryable Prefect task runs and exposes the two flows the CLI invokes
directly — no worker or deployment needed (spec §21 #8).

Importing this package exports ``settings.PREFECT_API_URL`` into the process
environment **before** ``prefect`` is imported: Prefect captures its
environment at import time, so a later export would be silently ignored and
runs would never reach the compose ``prefect`` service. In-container the
variable is already set (``env_file``) and the export is a no-op; on the
host it carries the ``.env`` value into Prefect. With no server reachable,
Prefect 3 falls back to an ephemeral in-process API, so flows still run
without the stack (untracked once the process exits).
"""

import os

from varagity.config import get_settings

# Must precede the prefect import chain below (see the module docstring).
os.environ.setdefault("PREFECT_API_URL", get_settings().PREFECT_API_URL)

from varagity.pipeline.eval_flow import eval_flow, ocr_benchmark_flow  # noqa: E402
from varagity.pipeline.ingest_flow import ingest_flow  # noqa: E402
from varagity.pipeline.query_flow import query_flow, query_stream_flow  # noqa: E402

__all__ = ["eval_flow", "ingest_flow", "ocr_benchmark_flow", "query_flow", "query_stream_flow"]
