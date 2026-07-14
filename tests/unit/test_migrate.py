"""Unit tests for the migration runner's ordering/tracking logic.

A scripted fake connection stands in for psycopg (the real-database
behavior — actual DDL, fresh-vs-existing volume convergence — is covered
by the integration suite).
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from varagity.stores.migrate import run_migrations


class FakeConnection:
    """Records executes; scripts the schema_migrations SELECT result."""

    def __init__(self, applied: list[str] | None = None) -> None:
        self.applied = applied or []
        self.executed: list[tuple[str, Any]] = []
        self.transactions = 0

    def execute(self, sql: str, params: Any = None) -> Any:
        self.executed.append((sql.strip(), params))
        if "SELECT name FROM schema_migrations" in sql:
            return iter([(name,) for name in self.applied])
        return None

    @contextmanager
    def transaction(self) -> Any:
        self.transactions += 1
        yield


def write_migrations(path: Path, names: list[str]) -> None:
    for name in names:
        (path / name).write_text(f"-- {name}\nCREATE TABLE IF NOT EXISTS t_{name[:3]} (id INT);")


def applied_inserts(conn: FakeConnection) -> list[str]:
    return [
        params[0]
        for sql, params in conn.executed
        if sql.startswith("INSERT INTO schema_migrations")
    ]


def test_applies_all_pending_in_filename_order(tmp_path: Path) -> None:
    write_migrations(tmp_path, ["002_second.sql", "001_first.sql", "003_third.sql"])
    conn = FakeConnection()
    ran = run_migrations(conn, tmp_path)  # type: ignore[arg-type]
    assert ran == ["001_first.sql", "002_second.sql", "003_third.sql"]
    assert applied_inserts(conn) == ran
    assert conn.transactions == 3


def test_skips_already_applied(tmp_path: Path) -> None:
    write_migrations(tmp_path, ["001_first.sql", "002_second.sql"])
    conn = FakeConnection(applied=["001_first.sql"])
    ran = run_migrations(conn, tmp_path)  # type: ignore[arg-type]
    assert ran == ["002_second.sql"]


def test_noop_when_current(tmp_path: Path) -> None:
    write_migrations(tmp_path, ["001_first.sql"])
    conn = FakeConnection(applied=["001_first.sql"])
    assert run_migrations(conn, tmp_path) == []  # type: ignore[arg-type]
    assert conn.transactions == 0


def test_creates_tracking_table_first(tmp_path: Path) -> None:
    write_migrations(tmp_path, ["001_first.sql"])
    conn = FakeConnection()
    run_migrations(conn, tmp_path)  # type: ignore[arg-type]
    assert "CREATE TABLE IF NOT EXISTS schema_migrations" in conn.executed[0][0]


def test_migration_sql_executed_verbatim(tmp_path: Path) -> None:
    write_migrations(tmp_path, ["001_first.sql"])
    conn = FakeConnection()
    run_migrations(conn, tmp_path)  # type: ignore[arg-type]
    executed_sql = [sql for sql, _ in conn.executed]
    assert any("CREATE TABLE IF NOT EXISTS t_001" in sql for sql in executed_sql)


def test_misnamed_sql_file_fails_loudly(tmp_path: Path) -> None:
    write_migrations(tmp_path, ["001_first.sql"])
    (tmp_path / "conversations.sql").write_text("SELECT 1;")
    conn = FakeConnection()
    with pytest.raises(ValueError, match="conversations.sql"):
        run_migrations(conn, tmp_path)  # type: ignore[arg-type]
    assert conn.executed == []  # nothing ran — fail before touching the db


def test_default_directory_is_the_package_migrations() -> None:
    from varagity.stores.migrate import MIGRATIONS_PATH

    names = sorted(p.name for p in MIGRATIONS_PATH.glob("*.sql"))
    assert names == ["001_conversations.sql", "002_app_settings.sql"]
    conn = FakeConnection(applied=names)
    assert run_migrations(conn) == []  # type: ignore[arg-type]
