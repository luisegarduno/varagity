"""``AppSettingsStore`` — persisted runtime setting overrides (spec_v2 §4.7).

A thin psycopg wrapper over the ``app_settings`` table (migration ``002``):
one row per overridden :class:`~varagity.config.Settings` field, keyed by
the field name with the value in its **env-string** form — exactly what
pydantic-settings parses from the environment, so the override layer
(:mod:`varagity.api.runtime_settings`) replays rows as env vars verbatim.

Keys beginning with ``_`` are reserved for app metadata and never surface
as overrides; the one in use is :data:`CORPUS_STALE_KEY` — the "an
ingest-affecting setting changed since the last reingest" flag behind the
GUI's persistent "Re-ingest to apply" affordance.
"""

import logging

import psycopg

from varagity.stores.base import ClosingContextMixin
from varagity.stores.vector_store import default_conninfo

logger = logging.getLogger(__name__)

CORPUS_STALE_KEY = "_corpus_stale"
"""Reserved metadata key flagging the corpus stale (needs ``--reingest``)."""

_UPSERT_SQL = """
INSERT INTO app_settings (key, value, updated_at)
VALUES (%(key)s, %(value)s, now())
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
"""


class AppSettingsStore(ClosingContextMixin):
    """Persisted key/value overrides over PostgreSQL.

    Owns one autocommit connection (every operation is a single statement).
    Use as a context manager or call :meth:`close` when done.
    """

    def __init__(self, conninfo: str | None = None) -> None:
        """Connect to the settings database.

        Args:
            conninfo: libpq connection string; defaults to the
                ``POSTGRES_*`` settings.

        Raises:
            psycopg.OperationalError: If the database is unreachable.
        """
        self._conn = psycopg.connect(conninfo or default_conninfo(), autocommit=True)

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        if not self._conn.closed:
            self._conn.close()

    def load_overrides(self) -> dict[str, str]:
        """Read every persisted setting override.

        Returns:
            Setting name → env-string value, excluding reserved (``_``-
            prefixed) metadata rows.
        """
        rows = self._conn.execute(
            "SELECT key, value FROM app_settings WHERE key NOT LIKE '\\_%'"
        ).fetchall()
        return {key: value for key, value in rows}

    def set_override(self, name: str, value: str) -> None:
        """Insert or update one override row.

        Args:
            name: The Settings field name (e.g. ``"RETRIEVAL_METHOD"``).
            value: The value in env-string form (e.g. ``"reranked"``,
                ``"true"``, ``"0.7"``).
        """
        self._conn.execute(_UPSERT_SQL, {"key": name, "value": value})

    def delete_override(self, name: str) -> None:
        """Remove one override row (reverting the setting to its env value).

        Args:
            name: The Settings field name (deleting an absent row is a
                no-op).
        """
        self._conn.execute("DELETE FROM app_settings WHERE key = %s", (name,))

    def is_corpus_stale(self) -> bool:
        """Read the corpus-stale metadata flag.

        Returns:
            ``True`` when a reingest-affecting override changed after the
            last reingest (set by ``PATCH /api/settings``, cleared by a
            completed ``POST /api/ingest`` with ``reingest=true``).
        """
        row = self._conn.execute(
            "SELECT value FROM app_settings WHERE key = %s", (CORPUS_STALE_KEY,)
        ).fetchone()
        return row is not None and row[0] == "true"

    def set_corpus_stale(self, stale: bool) -> None:
        """Write the corpus-stale metadata flag.

        Args:
            stale: The new flag value (stored as ``"true"``/``"false"``).
        """
        self._conn.execute(
            _UPSERT_SQL, {"key": CORPUS_STALE_KEY, "value": "true" if stale else "false"}
        )
