"""Unit tests for the persisted-overrides store (spec_v2 §4.7).

Real Postgres round-trips run in the integration suite; a scripted fake
connection verifies each method's SQL, the reserved-key exclusion, and the
corpus-stale flag's string encoding.
"""

from typing import Any

import psycopg
import pytest

from varagity.stores.app_settings_store import CORPUS_STALE_KEY, AppSettingsStore


class FakeCursor:
    def __init__(self, *, row: Any = None, rows: list[Any] | None = None):
        self._row = row
        self._rows = rows or []

    def fetchone(self) -> Any:
        return self._row

    def fetchall(self) -> list[Any]:
        return self._rows


class FakeConnection:
    def __init__(self, results: list[FakeCursor] | None = None) -> None:
        self.results = list(results or [])
        self.queries: list[tuple[str, Any]] = []
        self.closed = False

    def execute(self, sql: str, params: Any = None) -> FakeCursor:
        self.queries.append((sql, params))
        return self.results.pop(0) if self.results else FakeCursor()

    def close(self) -> None:
        self.closed = True


def store_with(conn: FakeConnection) -> AppSettingsStore:
    store = AppSettingsStore.__new__(AppSettingsStore)
    store._conn = conn  # type: ignore[assignment]
    return store


class TestLifecycle:
    def test_init_connects_and_close_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = FakeConnection()
        connects: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            psycopg,
            "connect",
            lambda conninfo, autocommit: connects.append((conninfo, autocommit)) or conn,
        )
        store = AppSettingsStore("host=example dbname=x")
        assert connects == [("host=example dbname=x", True)]
        store.close()
        assert conn.closed
        store.close()  # already closed — not re-closed


class TestOverrides:
    def test_load_overrides_excludes_reserved_keys_in_sql(self) -> None:
        conn = FakeConnection([FakeCursor(rows=[("RETRIEVAL_METHOD", "reranked")])])
        assert store_with(conn).load_overrides() == {"RETRIEVAL_METHOD": "reranked"}
        assert "NOT LIKE '\\_%'" in conn.queries[0][0]

    def test_set_override_upserts(self) -> None:
        conn = FakeConnection()
        store_with(conn).set_override("TOP_K", "12")
        sql, params = conn.queries[0]
        assert "ON CONFLICT (key) DO UPDATE" in sql
        assert params == {"key": "TOP_K", "value": "12"}

    def test_delete_override(self) -> None:
        conn = FakeConnection()
        store_with(conn).delete_override("TOP_K")
        assert conn.queries[0][1] == ("TOP_K",)


class TestCorpusStaleFlag:
    def test_absent_row_reads_not_stale(self) -> None:
        assert store_with(FakeConnection([FakeCursor()])).is_corpus_stale() is False

    @pytest.mark.parametrize(("stored", "expected"), [("true", True), ("false", False)])
    def test_reads_the_string_encoding(self, stored: str, expected: bool) -> None:
        conn = FakeConnection([FakeCursor(row=(stored,))])
        assert store_with(conn).is_corpus_stale() is expected

    @pytest.mark.parametrize(("stale", "encoded"), [(True, "true"), (False, "false")])
    def test_writes_the_string_encoding(self, stale: bool, encoded: str) -> None:
        conn = FakeConnection()
        store_with(conn).set_corpus_stale(stale)
        assert conn.queries[0][1] == {"key": CORPUS_STALE_KEY, "value": encoded}
