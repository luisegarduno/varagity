"""Export the API's OpenAPI schema to ``golden-docs/openapi.json``.

The checked-in snapshot feeds two consumers: the rendered HTTP-contract
page (``golden-docs/api.md`` embeds it via the ``neoteroi.mkdocsoad``
plugin) and the drift guard (``tests/unit/test_openapi_snapshot.py``),
which fails whenever the live app's schema no longer matches this file.
Regenerate with::

    uv run python scripts/export_openapi.py

``create_app()`` never runs the FastAPI lifespan, so no backing services
are needed. ``METRICS_ENABLED`` is pinned to ``true`` before any settings
read: the ``/metrics`` route joins the schema only when enabled, and the
snapshot must not depend on the exporting machine's ``.env``.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "golden-docs" / "openapi.json"

# The project is a uv "virtual" project (no build system): ``varagity`` is
# importable from the repo root, not installed into the venv — running this
# file as ``python scripts/export_openapi.py`` puts ``scripts/`` (not the
# root) on ``sys.path``.
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    """Build the app and write its OpenAPI schema to the snapshot path.

    The ``varagity`` imports are deferred until after ``METRICS_ENABLED``
    is pinned: importing :mod:`varagity.api.main` already reads settings
    (the ``PREFECT_API_URL`` export in :mod:`varagity.pipeline`), and the
    cached ``Settings`` must see the pinned value.
    """
    os.environ["METRICS_ENABLED"] = "true"

    from varagity.config import get_settings

    get_settings.cache_clear()

    from varagity.api.main import create_app

    schema = create_app().openapi()
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(schema['paths'])} paths)")


if __name__ == "__main__":
    main()
