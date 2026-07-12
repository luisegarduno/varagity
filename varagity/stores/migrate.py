"""Idempotent ordered-SQL migration runner (spec_v2 §9.3).

``schema.sql`` runs only on the postgres container's **first boot**
(``docker-entrypoint-initdb.d``), so schema added after a volume exists
never reaches it. This runner reconciles: ordered ``migrations/NNN_*.sql``
files — each written ``IF NOT EXISTS``-safe — are applied in a transaction
apiece and tracked by filename in a ``schema_migrations`` table. Running it
repeatedly is a no-op; running it against a fresh volume and a v1 volume
converges both to the same schema. It is invoked from the API's startup
lifespan (``varagity.api.main``).

Alembic is the noted heavier alternative if migrations ever get non-trivial
(plan decision #8; recorded in the runbook and ADR-005).
"""

import logging
import re
from pathlib import Path

import psycopg

logger = logging.getLogger(__name__)

MIGRATIONS_PATH = Path(__file__).parent / "migrations"
"""Directory holding the ordered ``NNN_*.sql`` migration files."""

# Ordered, descriptive, no surprises: 001_conversations.sql. Anything else
# in the directory is a mistake worth failing loudly over — a silently
# skipped migration is exactly the class of drift this runner exists to end.
_MIGRATION_NAME_RE = re.compile(r"^\d{3}_[a-z0-9_]+\.sql$")

_CREATE_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def run_migrations(conn: psycopg.Connection, migrations_path: Path | None = None) -> list[str]:
    """Apply every unapplied migration file, in filename order.

    Each unapplied file runs in its own transaction together with its
    ``schema_migrations`` bookkeeping row, so a failing migration rolls
    back atomically and a re-run re-attempts it.

    Args:
        conn: An open (autocommit) connection to the target database.
        migrations_path: Directory of ``NNN_*.sql`` files; defaults to
            :data:`MIGRATIONS_PATH`.

    Returns:
        The names of the migrations applied by this call, in order
        (empty when the database was already current).

    Raises:
        ValueError: If the directory contains a ``.sql`` file that does not
            match the ``NNN_name.sql`` convention (it would silently escape
            ordering otherwise).
        psycopg.Error: If a migration's SQL fails (the transaction for that
            file is rolled back).
    """
    path = MIGRATIONS_PATH if migrations_path is None else migrations_path
    sql_files = sorted(path.glob("*.sql"))
    misnamed = [f.name for f in sql_files if not _MIGRATION_NAME_RE.match(f.name)]
    if misnamed:
        raise ValueError(f"migration files must be named NNN_name.sql; got {misnamed} in {path}")

    conn.execute(_CREATE_TRACKING_TABLE_SQL)
    applied = {name for (name,) in conn.execute("SELECT name FROM schema_migrations")}

    ran: list[str] = []
    for sql_file in sql_files:
        if sql_file.name in applied:
            continue
        with conn.transaction():
            conn.execute(sql_file.read_text())
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (sql_file.name,))
        logger.info("applied migration %s", sql_file.name)
        ran.append(sql_file.name)
    if not ran:
        logger.info("schema is current (%d migration(s) already applied)", len(applied))
    return ran
