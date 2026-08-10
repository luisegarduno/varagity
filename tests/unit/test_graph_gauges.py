"""Unit tests for the workdir-derived graph collector (spec_graphrag §6).

The workdir is a real temporary directory holding the two files the
collector reads — an engine doc-status store and the summary sidecar — so
the file shapes are pinned rather than mocked away. What matters is the
contract: the gauges render, a scrape is not a read storm, an **unbuilt**
graph reports nothing rather than zeros, and an unreadable workdir degrades
``/metrics`` instead of taking it down.
"""

import json
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from varagity.graph.manifest import WorkdirManifest, save_summary
from varagity.observability.graph import (
    DOC_STATUS_FILENAME,
    GraphCollector,
    read_document_statuses,
    register_graph_collector,
)
from varagity.observability.metrics import catalog


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class WorkdirFactory:
    """Hands out a workdir path, counting the reads a scrape triggers."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.calls = 0

    def __call__(self) -> Path:
        self.calls += 1
        return self.workdir


def write_doc_statuses(workdir: Path, statuses: dict[str, str]) -> None:
    """Write an engine doc-status store holding ``doc_key → status``."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / DOC_STATUS_FILENAME).write_text(
        json.dumps(
            {
                key: {"status": status, "content_length": 10, "file_path": key}
                for key, status in statuses.items()
            }
        ),
        encoding="utf-8",
    )


def write_summary(workdir: Path, *, entities: int | None, relations: int | None, docs: int) -> None:
    """Write the summary sidecar through its own writer (shape stays honest)."""
    manifest = WorkdirManifest(docs={})
    save_summary(workdir, manifest, entities=entities, relations=relations)
    # message_guid_count() derives from the manifest, which this test does
    # not need to fabricate in full; patch the count in directly.
    path = workdir / "varagity_graph_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["message_guids"] = docs
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    built = tmp_path / "lightrag"
    write_doc_statuses(
        built,
        {"a::2024-01-01": "processed", "b::2024-01-02": "processed", "c::2024-01-03": "failed"},
    )
    write_summary(built, entities=42, relations=17, docs=310)
    return built


@pytest.fixture
def factory(workdir: Path) -> WorkdirFactory:
    return WorkdirFactory(workdir)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def collector(factory: WorkdirFactory, clock: FakeClock) -> GraphCollector:
    return GraphCollector(workdir_factory=factory, ttl_seconds=10.0, clock=clock)


def samples(collector: GraphCollector) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Flatten one collect() into {(name, sorted labels): value}."""
    out = {}
    for family in collector.collect():
        for sample in family.samples:
            out[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return out


class TestDocumentStatuses:
    def test_counts_by_status(self, workdir: Path) -> None:
        assert read_document_statuses(workdir) == {"processed": 2, "failed": 1}

    def test_absent_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        """A workdir nothing was ever enqueued into has no statuses."""
        assert read_document_statuses(tmp_path) == {}

    def test_statusless_record_counts_as_unknown(self, tmp_path: Path) -> None:
        (tmp_path / DOC_STATUS_FILENAME).write_text(json.dumps({"a": {}}), encoding="utf-8")
        assert read_document_statuses(tmp_path) == {"unknown": 1}

    def test_malformed_file_raises(self, tmp_path: Path) -> None:
        """The caller turns this into stale-serving, not into a zero graph."""
        (tmp_path / DOC_STATUS_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError):
            read_document_statuses(tmp_path)


class TestRendering:
    def test_gauges_carry_the_workdir_counts(self, collector: GraphCollector) -> None:
        values = samples(collector)
        assert values[("varagity_graph_documents", (("status", "processed"),))] == 2
        assert values[("varagity_graph_documents", (("status", "failed"),))] == 1
        assert values[("varagity_graph_entities", ())] == 42
        assert values[("varagity_graph_relations", ())] == 17
        assert values[("varagity_graph_messages", ())] == 310

    def test_unbuilt_graph_reports_no_samples(self, tmp_path: Path, clock: FakeClock) -> None:
        """An unbuilt graph is not a graph with zero entities."""
        collector = GraphCollector(
            workdir_factory=WorkdirFactory(tmp_path / "never-built"), clock=clock
        )
        families = list(collector.collect())
        assert [family.name for family in families] == [
            "varagity_graph_documents",
            "varagity_graph_entities",
            "varagity_graph_relations",
            "varagity_graph_messages",
        ]
        assert all(family.samples == [] for family in families)

    def test_sizes_the_engine_would_not_say_stay_absent(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """A graph whose size is unknown emits documents but no node counts."""
        built = tmp_path / "lightrag"
        write_doc_statuses(built, {"a::2024-01-01": "pending"})
        write_summary(built, entities=None, relations=None, docs=0)
        collector = GraphCollector(workdir_factory=WorkdirFactory(built), clock=clock)
        values = samples(collector)
        assert values[("varagity_graph_documents", (("status", "pending"),))] == 1
        assert ("varagity_graph_entities", ()) not in values
        assert ("varagity_graph_relations", ()) not in values
        assert values[("varagity_graph_messages", ())] == 0

    def test_workdir_without_a_summary_still_reports_statuses(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        built = tmp_path / "lightrag"
        write_doc_statuses(built, {"a::2024-01-01": "processing"})
        collector = GraphCollector(workdir_factory=WorkdirFactory(built), clock=clock)
        values = samples(collector)
        assert values[("varagity_graph_documents", (("status", "processing"),))] == 1
        assert ("varagity_graph_entities", ()) not in values

    def test_describe_does_not_touch_the_workdir(
        self, collector: GraphCollector, factory: WorkdirFactory
    ) -> None:
        """Registration must not turn app startup into a disk read."""
        families = list(collector.describe())
        assert factory.calls == 0
        assert {family.name for family in families} == set(
            name for name in catalog() if name.startswith("varagity_graph_")
        )


class TestCaching:
    def test_repeat_scrape_within_ttl_reuses_the_snapshot(
        self, collector: GraphCollector, factory: WorkdirFactory, clock: FakeClock
    ) -> None:
        samples(collector)
        clock.now += 9.0
        samples(collector)
        assert factory.calls == 1

    def test_scrape_after_ttl_rereads(
        self,
        collector: GraphCollector,
        factory: WorkdirFactory,
        clock: FakeClock,
        workdir: Path,
    ) -> None:
        samples(collector)
        clock.now += 11.0
        write_doc_statuses(workdir, {"a::2024-01-01": "processed", "b::2024-01-02": "processed"})
        write_summary(workdir, entities=50, relations=20, docs=400)
        values = samples(collector)
        assert factory.calls == 2
        assert values[("varagity_graph_entities", ())] == 50
        assert ("varagity_graph_documents", (("status", "failed"),)) not in values


class TestWorkdirOutage:
    def test_unreadable_workdir_serves_the_last_snapshot(
        self,
        collector: GraphCollector,
        clock: FakeClock,
        workdir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        samples(collector)
        clock.now += 11.0
        (workdir / DOC_STATUS_FILENAME).write_text("{truncated", encoding="utf-8")

        values = samples(collector)

        assert values[("varagity_graph_entities", ())] == 42  # stale, not absent
        assert values[("varagity_graph_documents", (("status", "processed"),))] == 2
        assert "would not read" in caplog.text

    def test_outage_before_any_success_emits_no_samples(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """A never-readable workdir must not fabricate an empty graph."""
        built = tmp_path / "lightrag"
        built.mkdir()
        (built / DOC_STATUS_FILENAME).write_text("{truncated", encoding="utf-8")
        collector = GraphCollector(workdir_factory=WorkdirFactory(built), clock=clock)

        families = list(collector.collect())

        assert [family.name for family in families]  # the families still exist
        assert all(family.samples == [] for family in families)

    def test_outage_never_raises_through_collect(self, tmp_path: Path) -> None:
        """A scrape that raises would 500 /metrics — including the healthy metrics."""
        built = tmp_path / "lightrag"
        built.mkdir()
        (built / DOC_STATUS_FILENAME).write_text("nonsense", encoding="utf-8")
        collector = GraphCollector(workdir_factory=WorkdirFactory(built))
        list(collector.collect())  # must not raise

    def test_recovery_refreshes(
        self, collector: GraphCollector, clock: FakeClock, workdir: Path
    ) -> None:
        (workdir / DOC_STATUS_FILENAME).write_text("{truncated", encoding="utf-8")
        list(collector.collect())
        write_doc_statuses(workdir, {"a::2024-01-01": "processed"})
        clock.now += 11.0

        values = samples(collector)

        assert values[("varagity_graph_documents", (("status", "processed"),))] == 1


class TestRegistration:
    def test_registers_into_the_given_registry(self, factory: WorkdirFactory) -> None:
        registry = CollectorRegistry()
        register_graph_collector(registry, GraphCollector(workdir_factory=factory))
        assert registry.get_sample_value("varagity_graph_entities") == 42

    def test_re_registration_replaces_rather_than_raises(self, factory: WorkdirFactory) -> None:
        """create_app() runs many times per test session against one registry."""
        registry = CollectorRegistry()
        register_graph_collector(registry, GraphCollector(workdir_factory=factory))
        register_graph_collector(registry, GraphCollector(workdir_factory=factory))
        assert registry.get_sample_value("varagity_graph_entities") == 42


class TestCatalog:
    def test_graph_gauges_are_catalogued(self) -> None:
        """The gauges are visible to the dashboard guard.

        It checks every panel expression against ``catalog()``, so a gauge
        the catalog does not know is a panel nothing can police (§6.4).
        """
        declared = catalog()
        assert declared["varagity_graph_documents"] == ("status",)
        assert declared["varagity_graph_entities"] == ()
        assert declared["varagity_graph_relations"] == ()
        assert declared["varagity_graph_messages"] == ()

    def test_the_corpus_gauges_survive_the_merge(self) -> None:
        """Both gauge families are merged in, not one over the other."""
        declared = catalog()
        assert "varagity_corpus_documents" in declared
        assert declared["varagity_corpus_documents_by_type"] == ("file_type",)
