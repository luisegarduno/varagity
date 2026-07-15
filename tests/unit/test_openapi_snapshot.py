"""Drift guard: ``golden-docs/openapi.json`` must match the live app's schema.

The snapshot is what the docs site renders (``golden-docs/api.md``) and what
``pnpm gen:types`` mirrors into ``web/lib/types.ts`` — a stale copy means the
documented contract lies. ``METRICS_ENABLED`` is pinned because it gates the
``/metrics`` route's presence in the schema (the export script pins the same).
"""

import json
from pathlib import Path
from typing import Any

from varagity.api.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "golden-docs" / "openapi.json"


class TestOpenAPISnapshot:
    def test_snapshot_matches_the_live_schema(self, settings_env: Any) -> None:
        settings_env(METRICS_ENABLED="true")
        live = create_app().openapi()
        snapshot = json.loads(SNAPSHOT_PATH.read_text())
        # Parsed-object comparison: key order (the export sorts, FastAPI
        # doesn't) can never fail this — only real contract drift can.
        assert live == snapshot, (
            "golden-docs/openapi.json no longer matches the app's schema — "
            "regenerate it with `uv run python scripts/export_openapi.py`"
        )
