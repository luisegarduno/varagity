"""Unit tests for the store-derived corpus collector (spec_v3 §6.1a).

The store is faked: what matters here is the collector's contract — that it
renders the gauges, caches within the TTL so a scrape is not a query storm,
and above all that a store outage degrades ``/metrics`` instead of taking
it down.
"""

from typing import Any

import psycopg
import pytest
from prometheus_client import CollectorRegistry

from varagity.observability import corpus as corpus_module
from varagity.observability.corpus import CorpusCollector, register_corpus_collector


class FakeStore:
    """A ContextualVectorDB stand-in with scripted counts."""

    def __init__(self, owner: "FakeStoreFactory") -> None:
        self._owner = owner

    def __enter__(self) -> "FakeStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._owner.closed += 1

    def document_count(self) -> int:
        return self._owner.documents

    def chunk_count(self) -> int:
        return self._owner.chunks

    def document_count_by_type(self) -> dict[str, int]:
        return dict(self._owner.by_type)

    def chunk_count_by_strategy(self) -> dict[str, int]:
        return dict(self._owner.by_strategy)


class FakeStoreFactory:
    """Builds :class:`FakeStore`s, counting calls and scripting failure."""

    def __init__(self) -> None:
        self.documents = 2
        self.chunks = 7
        self.by_type: dict[str, int] = {"md": 1, "pdf": 1}
        self.by_strategy: dict[str, int] = {"semantic": 7}
        self.calls = 0
        self.closed = 0
        self.error: Exception | None = None

    def __call__(self) -> FakeStore:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return FakeStore(self)


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def factory() -> FakeStoreFactory:
    return FakeStoreFactory()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def collector(factory: FakeStoreFactory, clock: FakeClock) -> CorpusCollector:
    return CorpusCollector(store_factory=factory, ttl_seconds=10.0, clock=clock)


def samples(collector: CorpusCollector) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Flatten one collect() into {(name, sorted labels): value}."""
    out = {}
    for family in collector.collect():
        for sample in family.samples:
            out[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return out


class TestRendering:
    def test_gauges_carry_the_store_counts(self, collector: CorpusCollector) -> None:
        values = samples(collector)
        assert values[("varagity_corpus_documents", ())] == 2
        assert values[("varagity_corpus_chunks", ())] == 7
        assert values[("varagity_corpus_documents_by_type", (("file_type", "md"),))] == 1
        assert values[("varagity_corpus_documents_by_type", (("file_type", "pdf"),))] == 1
        assert (
            values[("varagity_corpus_chunks_by_strategy", (("chunking_strategy", "semantic"),))]
            == 7
        )

    def test_empty_corpus_reports_zero_not_absence(
        self, collector: CorpusCollector, factory: FakeStoreFactory
    ) -> None:
        """An empty corpus is a fact worth 0, not a missing series."""
        factory.documents = 0
        factory.chunks = 0
        factory.by_type = {}
        factory.by_strategy = {}
        values = samples(collector)
        assert values[("varagity_corpus_documents", ())] == 0
        assert values[("varagity_corpus_chunks", ())] == 0

    def test_describe_does_not_touch_the_store(
        self, collector: CorpusCollector, factory: FakeStoreFactory
    ) -> None:
        """Registration must not turn app startup into a database query."""
        families = list(collector.describe())
        assert factory.calls == 0
        assert {f.name for f in families} == {
            "varagity_corpus_documents",
            "varagity_corpus_chunks",
            "varagity_corpus_documents_by_type",
            "varagity_corpus_chunks_by_strategy",
        }


class TestCaching:
    def test_repeat_scrape_within_ttl_reuses_the_snapshot(
        self, collector: CorpusCollector, factory: FakeStoreFactory, clock: FakeClock
    ) -> None:
        samples(collector)
        clock.now += 9.0
        samples(collector)
        assert factory.calls == 1

    def test_scrape_after_ttl_requeries(
        self, collector: CorpusCollector, factory: FakeStoreFactory, clock: FakeClock
    ) -> None:
        samples(collector)
        clock.now += 11.0
        factory.documents = 5
        values = samples(collector)
        assert factory.calls == 2
        assert values[("varagity_corpus_documents", ())] == 5

    def test_connection_is_closed_after_each_refresh(
        self, collector: CorpusCollector, factory: FakeStoreFactory, clock: FakeClock
    ) -> None:
        samples(collector)
        clock.now += 11.0
        samples(collector)
        assert factory.closed == 2


class TestStoreOutage:
    def test_outage_serves_the_last_snapshot(
        self,
        collector: CorpusCollector,
        factory: FakeStoreFactory,
        clock: FakeClock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        samples(collector)
        clock.now += 11.0
        factory.error = psycopg.OperationalError("connection refused")

        values = samples(collector)

        assert values[("varagity_corpus_documents", ())] == 2  # stale, not absent
        assert "postgres unreachable" in caplog.text

    def test_outage_before_any_success_emits_no_samples(
        self, collector: CorpusCollector, factory: FakeStoreFactory
    ) -> None:
        """A never-reachable store must not fabricate a 0-document corpus."""
        factory.error = psycopg.OperationalError("connection refused")

        families = list(collector.collect())

        assert [f.name for f in families]  # the families still exist
        assert all(f.samples == [] for f in families)

    def test_outage_never_raises_through_collect(
        self, collector: CorpusCollector, factory: FakeStoreFactory
    ) -> None:
        """A scrape that raises would 500 /metrics — including the healthy metrics."""
        factory.error = psycopg.OperationalError("connection refused")
        list(collector.collect())  # must not raise

    def test_a_restores_table_lock_serves_the_last_snapshot(
        self,
        collector: CorpusCollector,
        factory: FakeStoreFactory,
        clock: FakeClock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A restore holds the tables until it commits: the scrape gives up and serves stale."""
        samples(collector)
        clock.now += 11.0
        factory.documents = 99  # what the store would say once the restore commits
        factory.error = psycopg.errors.LockNotAvailable("canceling statement due to lock timeout")

        with caplog.at_level("INFO", logger="varagity.observability.corpus"):
            values = samples(collector)

        assert values[("varagity_corpus_documents", ())] == 2  # the last snapshot
        assert "tables are locked" in caplog.text
        assert "unreachable" not in caplog.text  # not an outage

    def test_recovery_refreshes(
        self, collector: CorpusCollector, factory: FakeStoreFactory, clock: FakeClock
    ) -> None:
        factory.error = psycopg.OperationalError("down")
        list(collector.collect())
        factory.error = None
        clock.now += 11.0

        values = samples(collector)

        assert values[("varagity_corpus_documents", ())] == 2


class TestGaugeStore:
    def test_the_default_store_carries_a_lock_timeout(
        self, monkeypatch: pytest.MonkeyPatch, settings_env: Any
    ) -> None:
        settings_env(POSTGRES_HOST="db.test", POSTGRES_DB="varagity")
        seen: list[str] = []
        monkeypatch.setattr(
            corpus_module, "ContextualVectorDB", lambda conninfo: seen.append(conninfo)
        )
        corpus_module.gauge_store()
        (conninfo,) = seen
        params = psycopg.conninfo.conninfo_to_dict(conninfo)
        assert params["options"] == f"-c lock_timeout={corpus_module.GAUGE_LOCK_TIMEOUT_MS}"
        assert params["host"] == "db.test"
        assert CorpusCollector()._store_factory is corpus_module.gauge_store


class TestRegistration:
    def test_registers_into_the_given_registry(self, factory: FakeStoreFactory) -> None:
        registry = CollectorRegistry()
        register_corpus_collector(registry, CorpusCollector(store_factory=factory))
        assert registry.get_sample_value("varagity_corpus_documents") == 2

    def test_re_registration_replaces_rather_than_raises(self, factory: FakeStoreFactory) -> None:
        """create_app() runs many times per test session against one registry."""
        registry = CollectorRegistry()
        register_corpus_collector(registry, CorpusCollector(store_factory=factory))
        register_corpus_collector(registry, CorpusCollector(store_factory=factory))
        assert registry.get_sample_value("varagity_corpus_documents") == 2
