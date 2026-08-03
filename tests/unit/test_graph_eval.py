"""Unit tests for the `eval graph` bake-off harness (spec_graphrag §12).

No engine library, no GPU service, no store: a throwaway ``_probe`` engine is
registered for the duration of a test (the ``test_chat_engines.py`` precedent)
and hands back canned answers, so the scoring math, the settings pinning, the
corpus/workdir lifecycle, and the results shape are all exercised end to end
through :func:`run_graph_eval` itself.

The corpus half is real: the harness builds the shipped fixture ``chat.db``
and parses it with the product parser, which is what makes the manifest-pin
regression (raw phone handles instead of "Bob Nakamura") catchable here rather
than three hours into a bake-off.
"""

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from varagity.config import get_settings
from varagity.eval.datasets import GRAPH_GOLDEN_KINDS, GraphGoldenEntry
from varagity.eval.graph_eval import (
    ENGINE_DISTRIBUTIONS,
    GRAPH_EVAL_ROOT,
    INCREMENTAL_DELTA_MESSAGES,
    MANIFEST_SUFFIX,
    corpus_stem,
    engine_versions,
    manifest_settings_pins,
    match_fact_groups,
    measure_graph_engine,
    open_engine_session,
    pinned_graph_settings,
    prepare_corpus,
    provenance_recall,
    run_graph_eval,
    select_engines,
    validate_golden_against_manifest,
)
from varagity.eval.graph_fixtures import FixtureManifest, build_fixture_chat_db
from varagity.graph.base import GRAPH_ENGINE_REGISTRY
from varagity.graph.records import (
    BuildReport,
    GraphAnswer,
    GraphEntity,
    GraphEvidence,
    GraphStats,
)
from varagity.graph.sources.base import MessageBatch, batch_for_path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Scripted guids the shipped fixture always contains (pinned by
# tests/unit/test_graph_fixtures.py).
GUID_A = "fx-hw-001"
GUID_B = "fx-bob-001"
GUID_C = "fx-crew-001"

GOLDEN_LINES: list[dict[str, Any]] = [
    {
        "query": "What does Bob think about computers?",
        "kind": "aggregation",
        "expected_facts": [["mechanical keyboard"], ["trackpad"]],
        "required_guids": [GUID_A, GUID_B],
    },
    {
        "query": "Did Jane ever say she likes keyboards?",
        "kind": "verification",
        "expected_facts": [["yes"], ["march"]],
        "required_guids": [GUID_B],
    },
    {
        "query": "Who told me it was Bob's birthday?",
        "kind": "relation",
        "expected_facts": [["carol"]],
        "required_guids": [GUID_C],
    },
]


def write_golden(path: Path, entries: Sequence[dict[str, Any]] = GOLDEN_LINES) -> Path:
    """Write a golden JSONL file for a test run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(entry)}\n" for entry in entries),
        encoding="utf-8",
    )
    return path


def golden_entries(
    entries: Sequence[dict[str, Any]] = GOLDEN_LINES,
) -> list[GraphGoldenEntry]:
    return [GraphGoldenEntry.model_validate(entry) for entry in entries]


def answer(
    text: str,
    *,
    guids: Sequence[str] = (),
    mode: str = "hybrid",
    latency_s: float = 1.25,
) -> GraphAnswer:
    return GraphAnswer(
        answer=text,
        evidence=GraphEvidence(
            entities=[GraphEntity(name="Bob Nakamura")],
            message_guids=list(guids),
        ),
        mode=mode,
        latency_s=latency_s,
    )


@dataclass
class ProbeSession:
    """A :class:`~varagity.graph.base.GraphSession` double with canned answers."""

    engine: "ProbeEngine"
    workdir: Path
    builds: list[list[MessageBatch]] = field(default_factory=list)
    closed: bool = False

    def build(self, batches: Sequence[MessageBatch], *, verbose: int = 0) -> BuildReport:
        self.builds.append(list(batches))
        self.engine.builds.append([len(batch.messages) for batch in batches])
        return BuildReport(
            messages_seen=sum(len(batch.messages) for batch in batches),
            wall_clock_s=0.5 * len(self.builds),
            failures=list(self.engine.build_failures),
        )

    def query(self, question: str, *, mode: str | None = None, verbose: int = 0) -> GraphAnswer:
        self.engine.asked.append((question, mode))
        if question in self.engine.raises:
            raise RuntimeError(self.engine.raises[question])
        canned = self.engine.script.get(question, answer("nothing useful"))
        return canned if mode is None else canned.model_copy(update={"mode": mode})

    def stats(self) -> GraphStats:
        return GraphStats(entities=len(self.builds) * 10, relations=len(self.builds) * 20)


class ProbeEngine:
    """A registered graph engine whose sessions answer from a script."""

    def __init__(self) -> None:
        self.script: dict[str, GraphAnswer] = {}
        self.raises: dict[str, str] = {}
        self.build_failures: list[str] = []
        self.sessions: list[ProbeSession] = []
        self.builds: list[list[int]] = []
        self.asked: list[tuple[str, str | None]] = []

    @contextmanager
    def session(self, workdir: Path) -> Iterator[ProbeSession]:
        session = ProbeSession(self, workdir)
        self.sessions.append(session)
        try:
            yield session
        finally:
            session.closed = True


@pytest.fixture
def probe() -> Iterator[ProbeEngine]:
    """Register a throwaway ``_probe`` engine and remove it afterwards."""
    engine = ProbeEngine()
    engine.script = {
        GOLDEN_LINES[0]["query"]: answer(
            "Bob loves his mechanical keyboard and hates the trackpad.",
            guids=[GUID_A, GUID_B],
        ),
        # Half the fact groups, and only half the required provenance.
        GOLDEN_LINES[1]["query"]: answer("Yes, she did.", guids=[GUID_B, "fx-unrelated"]),
        # A correct answer from an engine that cannot cite messages at all.
        GOLDEN_LINES[2]["query"]: answer("Carol told you.", guids=[]),
    }
    GRAPH_ENGINE_REGISTRY["_probe"] = engine
    try:
        yield engine
    finally:
        GRAPH_ENGINE_REGISTRY.pop("_probe", None)


@pytest.fixture
def at_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from the repo root: the fixture script paths are relative to it."""
    monkeypatch.chdir(_REPO_ROOT)


class TestFactScoring:
    def test_a_group_is_satisfied_by_any_variant_case_insensitively(self) -> None:
        assert match_fact_groups("He prefers ARM chips", [["apple silicon", "arm"]]) == [True]

    def test_every_group_is_scored_independently(self) -> None:
        matched = match_fact_groups(
            "mechanical keyboard, definitely",
            [["mechanical keyboard"], ["trackpad"], ["arm"]],
        )
        assert matched == [True, False, False]

    def test_an_empty_answer_satisfies_nothing(self) -> None:
        assert match_fact_groups("", [["anything"]]) == [False]


class TestProvenanceRecall:
    def test_intersection_over_required(self) -> None:
        assert provenance_recall(["a", "b", "z"], ["a", "b"]) == 1.0
        assert provenance_recall(["a", "z"], ["a", "b"]) == 0.5

    def test_no_surfaced_guids_is_none_not_zero(self) -> None:
        """★ Criterion §8.2#4: "cannot cite" is reported, never scored as 0."""
        assert provenance_recall([], ["a"]) is None

    def test_surfaced_but_wrong_guids_score_zero(self) -> None:
        assert provenance_recall(["x"], ["a"]) == 0.0

    def test_empty_requirements_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one required guid"):
            provenance_recall(["a"], [])


class TestMeasureGraphEngine:
    def _session(self, probe: ProbeEngine) -> ProbeSession:
        session = ProbeSession(probe, Path("unused"))
        probe.sessions.append(session)
        return session

    def test_scores_facts_provenance_and_slices(self, probe: ProbeEngine) -> None:
        results = measure_graph_engine(self._session(probe), golden_entries())
        summary = results["summary"]
        assert summary["n_queries"] == 3
        # 1.0 (both groups) + 0.5 (one of two) + 1.0 (the only group) over 3.
        assert summary["fact_recall"] == pytest.approx((1.0 + 0.5 + 1.0) / 3, abs=1e-4)
        # Provenance averages only the two entries that reported any.
        assert summary["provenance_recall"] == pytest.approx((1.0 + 1.0) / 2)
        assert summary["n_provenance_reported"] == 2
        assert summary["mean_latency_s"] == 1.25
        assert summary["errors"] == 0

    def test_by_kind_only_holds_the_kinds_present(self, probe: ProbeEngine) -> None:
        results = measure_graph_engine(self._session(probe), golden_entries())
        assert list(results["by_kind"]) == list(GRAPH_GOLDEN_KINDS)
        assert results["by_kind"]["aggregation"]["fact_recall"] == 1.0
        assert results["by_kind"]["verification"]["fact_recall"] == 0.5
        # The relation question was answered correctly but cited nothing.
        assert results["by_kind"]["relation"]["provenance_recall"] is None
        assert results["by_kind"]["relation"]["n_provenance_reported"] == 0

    def test_per_query_detail_keeps_the_answer_and_the_misses(self, probe: ProbeEngine) -> None:
        records = measure_graph_engine(self._session(probe), golden_entries())["queries"]
        assert [record["kind"] for record in records] == ["aggregation", "verification", "relation"]
        assert records[0]["answer"].startswith("Bob loves")
        assert records[0]["missed_facts"] == []
        assert records[1]["missed_facts"] == [["march"]]
        # Only required guids count as matched; the engine's extra id is not one.
        assert records[1]["matched_guids"] == [GUID_B]
        assert records[1]["evidence"]["message_guids"] == 2
        assert records[0]["mode"] == "hybrid"
        assert records[0]["latency_s"] == 1.25

    def test_the_question_is_put_verbatim_and_the_kind_is_never_shown(
        self, probe: ProbeEngine
    ) -> None:
        """Plan decision #10: ``kind`` is golden metadata, not a hint."""
        measure_graph_engine(self._session(probe), golden_entries())
        assert [asked for asked, _ in probe.asked] == [entry["query"] for entry in GOLDEN_LINES]

    def test_a_mode_override_is_passed_through_and_recorded(self, probe: ProbeEngine) -> None:
        results = measure_graph_engine(self._session(probe), golden_entries(), mode="global")
        assert {mode for _, mode in probe.asked} == {"global"}
        assert {record["mode"] for record in results["queries"]} == {"global"}

    def test_a_failing_query_is_recorded_not_raised(self, probe: ProbeEngine) -> None:
        """Engine failures are §8.2#2 data — losing an engine's run is not."""
        probe.raises = {GOLDEN_LINES[0]["query"]: "the graph exploded"}
        results = measure_graph_engine(self._session(probe), golden_entries())
        failed = results["queries"][0]
        assert failed["error"] == "RuntimeError: the graph exploded"
        assert failed["fact_recall"] == 0.0
        assert failed["provenance_recall"] is None
        assert failed["latency_s"] is None
        assert failed["missed_facts"] == [["mechanical keyboard"], ["trackpad"]]
        assert results["summary"]["errors"] == 1
        assert results["summary"]["n_queries"] == 3  # still scored, as a miss

    def test_no_entries_raises(self, probe: ProbeEngine) -> None:
        with pytest.raises(ValueError, match="at least one golden entry"):
            measure_graph_engine(self._session(probe), [])


class TestSelectEngines:
    def test_default_is_every_registered_engine_sorted(self, probe: ProbeEngine) -> None:
        assert select_engines(None) == sorted(GRAPH_ENGINE_REGISTRY)
        assert "_probe" in select_engines(None)

    def test_the_filter_selects_and_dedupes(self, probe: ProbeEngine) -> None:
        assert select_engines(["_probe", "_probe"]) == ["_probe"]

    def test_an_unknown_engine_names_what_is_registered(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            select_engines(["made_up"])
        message = str(excinfo.value)
        assert "made_up" in message
        assert "lightrag" in message  # the listing names what IS available

    def test_an_empty_filter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no graph engines selected"):
            select_engines([])


class TestEngineVersions:
    def test_the_distribution_map_covers_every_registered_adapter(self) -> None:
        """The drift guard for a map that is deliberately re-declared."""
        assert set(ENGINE_DISTRIBUTIONS) == set(GRAPH_ENGINE_REGISTRY)

    def test_a_mapped_engine_stamps_its_version_when_installed(self) -> None:
        """The shipped engine is a main dependency, so this stamps a real version."""
        stamped = engine_versions(sorted(ENGINE_DISTRIBUTIONS))
        assert set(stamped) == set(ENGINE_DISTRIBUTIONS)
        assert all(value is None or value for value in stamped.values())

    def test_an_uninstalled_distribution_stamps_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(ENGINE_DISTRIBUTIONS, "_probe", "varagity-not-a-distribution")
        assert engine_versions(["_probe"]) == {"_probe": None}

    def test_an_unmapped_engine_stamps_none(self, probe: ProbeEngine) -> None:
        assert engine_versions(["_probe"]) == {"_probe": None}


class TestManifestPins:
    def test_pins_come_from_what_the_generator_wrote(self, tmp_path: Path) -> None:
        manifest = build_fixture_chat_db(
            tmp_path / "fixture.db",
            script_path=_REPO_ROOT / "tests/fixtures/graph/scripted_messages.json",
            blob_path=_REPO_ROOT / "tests/fixtures/graph/attributed_body_sample.bin",
        )
        pins = manifest_settings_pins(manifest)
        assert pins["GRAPH_OWNER_ALIASES"] == manifest.owner_label
        assert pins["GRAPH_HANDLE_NAMES_FILE"] == ""
        for handle, name in manifest.handle_names.items():
            assert f"{handle}={name}" in pins["GRAPH_HANDLE_NAMES"]

    def test_pinning_is_what_makes_the_parse_name_people(self, tmp_path: Path) -> None:
        """★ The Phase-2 as-built note: without the pins, every golden fails."""
        db_path = tmp_path / "fixture.db"
        manifest = build_fixture_chat_db(
            db_path,
            script_path=_REPO_ROOT / "tests/fixtures/graph/scripted_messages.json",
            blob_path=_REPO_ROOT / "tests/fixtures/graph/attributed_body_sample.bin",
        )
        with pinned_graph_settings(
            GRAPH_OWNER_ALIASES="", GRAPH_HANDLE_NAMES="", GRAPH_HANDLE_NAMES_FILE=""
        ):
            unpinned = batch_for_path(db_path, tmp_path)
        with pinned_graph_settings(**manifest_settings_pins(manifest)):
            pinned = batch_for_path(db_path, tmp_path)
        assert "Bob Nakamura" not in {message.sender_name for message in unpinned.messages}
        assert "Bob Nakamura" in {message.sender_name for message in pinned.messages}

    def test_merges_handle_maps_across_corpora(self) -> None:
        base = _manifest(handle_names={"+1": "Bob"})
        delta = _manifest(handle_names={"+1": "Bob", "+2": "Ada"})
        pins = manifest_settings_pins(base, delta)
        assert pins["GRAPH_HANDLE_NAMES"] == "+1=Bob,+2=Ada"

    def test_disagreeing_owner_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="owner label"):
            manifest_settings_pins(_manifest(owner_label="Sam"), _manifest(owner_label="Alex"))

    def test_no_manifest_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one manifest"):
            manifest_settings_pins()

    def test_the_pins_are_restored_on_exit(self, settings_env: Any) -> None:
        settings_env(GRAPH_OWNER_ALIASES="Original")
        with pinned_graph_settings(GRAPH_OWNER_ALIASES="Pinned"):
            assert get_settings().graph_owner_label == "Pinned"
        assert get_settings().graph_owner_label == "Original"


def _manifest(**overrides: Any) -> FixtureManifest:
    """A minimal manifest for the pure-logic tests."""
    base: dict[str, Any] = {
        "profile": "smoke",
        "seed": 13,
        "message_count": 2,
        "scripted_count": 2,
        "filler_count": 0,
        "tapback_count": 0,
        "thread_counts": {"t": 2},
        "guids": [GUID_A, GUID_B],
        "first_timestamp": datetime(2016, 3, 4, tzinfo=UTC),
        "last_timestamp": datetime(2024, 3, 4, tzinfo=UTC),
        "owner_label": "Sam",
        "handle_names": {"+1": "Bob"},
    }
    return FixtureManifest.model_validate({**base, **overrides})


class TestValidateGoldenAgainstManifest:
    def test_anchored_goldens_pass(self) -> None:
        validate_golden_against_manifest(golden_entries(GOLDEN_LINES[:1]), _manifest())

    def test_a_missing_anchor_names_it(self) -> None:
        entries = golden_entries([{**GOLDEN_LINES[0], "required_guids": ["fx-ghost"]}])
        with pytest.raises(ValueError, match="fx-ghost"):
            validate_golden_against_manifest(entries, _manifest())


class TestCorpusStem:
    def test_an_uncapped_run_keeps_the_bare_profile_name(self) -> None:
        assert corpus_stem("full") == "full"
        assert corpus_stem("smoke", None) == "smoke"

    def test_a_capped_run_gets_its_own_name(self) -> None:
        """★ A capped run must never write over the uncapped corpus."""
        assert corpus_stem("full", 1000) == "full-mt1000"
        assert corpus_stem("smoke", 50) == "smoke-mt50"


class TestPrepareCorpus:
    def test_builds_the_database_and_writes_its_manifest(
        self, tmp_path: Path, at_repo_root: None
    ) -> None:
        db_path, manifest = prepare_corpus(tmp_path, profile="smoke")
        assert db_path == tmp_path / "smoke.db"
        assert db_path.is_file()
        written = (tmp_path / f"smoke{MANIFEST_SUFFIX}").read_text(encoding="utf-8")
        assert json.loads(written)["message_count"] == manifest.message_count
        assert manifest.filler_count == 0

    def test_reuse_reads_the_manifest_instead_of_rebuilding(
        self, tmp_path: Path, at_repo_root: None
    ) -> None:
        _, manifest = prepare_corpus(tmp_path, profile="smoke")
        manifest_path = tmp_path / f"smoke{MANIFEST_SUFFIX}"
        marked = json.loads(manifest_path.read_text(encoding="utf-8"))
        marked["scripted_count"] = 99  # a value a rebuild would overwrite
        manifest_path.write_text(json.dumps(marked), encoding="utf-8")
        _, reused = prepare_corpus(tmp_path, profile="smoke", reuse=True)
        assert reused.scripted_count == 99
        assert reused.message_count == manifest.message_count

    def test_reuse_without_a_manifest_rebuilds(self, tmp_path: Path, at_repo_root: None) -> None:
        _, built = prepare_corpus(tmp_path, profile="smoke", reuse=True)
        assert built.scripted_count > 0
        assert (tmp_path / f"smoke{MANIFEST_SUFFIX}").is_file()

    def test_the_message_target_and_name_are_honored(
        self, tmp_path: Path, at_repo_root: None
    ) -> None:
        _, smoke = prepare_corpus(tmp_path, profile="smoke")
        path, delta = prepare_corpus(
            tmp_path,
            profile="full",
            message_target=smoke.message_count + 4,
            name="smoke-delta",
        )
        assert path.name == "smoke-delta.db"
        assert delta.message_count == smoke.message_count + 4
        assert delta.filler_count == 4


class TestOpenEngineSession:
    def test_a_missing_engine_library_names_the_sync_that_fixes_it(self) -> None:
        """An adapter's lazy engine import fails inside ``session()``."""

        class MissingEngine:
            @contextmanager
            def session(self, workdir: Path) -> Iterator[Any]:
                raise ModuleNotFoundError("No module named 'lightrag'", name="lightrag")
                yield  # pragma: no cover  (unreachable — keeps this a generator)

        GRAPH_ENGINE_REGISTRY["_missing"] = MissingEngine()
        try:
            with (
                pytest.raises(RuntimeError) as excinfo,
                open_engine_session("_missing", Path("unused")),
            ):
                pass  # pragma: no cover
        finally:
            GRAPH_ENGINE_REGISTRY.pop("_missing", None)
        message = str(excinfo.value)
        assert "_missing" in message
        assert "lightrag" in message
        assert "uv sync" in message

    def test_a_healthy_session_is_yielded_and_closed(self, probe: ProbeEngine) -> None:
        with open_engine_session("_probe", Path("workdir")) as session:
            assert session.workdir == Path("workdir")
        assert probe.sessions[0].closed is True


class TestRunGraphEval:
    def _run(self, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
        return run_graph_eval(
            engines=["_probe"],
            eval_root=tmp_path / "graph",
            golden_path=write_golden(tmp_path / "golden.jsonl"),
            results_dir=tmp_path / "results",
            verbose=0,
            **kwargs,
        )

    def test_smoke_run_builds_scores_and_persists(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        results = self._run(tmp_path)
        assert results["kind"] == "graph_eval"
        assert results["profile"] == "smoke"
        assert results["n_queries"] == 3
        assert results["corpus"]["scripted"] == results["corpus"]["messages"]
        assert results["corpus"]["threads"] == 10
        assert results["engine_versions"] == {"_probe": None}
        assert results["chat_model"] and results["embedding_model"]
        # The pins that made the parse name people are recorded with the run.
        assert results["pinned_settings"]["GRAPH_OWNER_ALIASES"]
        assert "Bob Nakamura" in results["pinned_settings"]["GRAPH_HANDLE_NAMES"]

        engine = results["engines"]["_probe"]
        assert engine["build"]["messages_seen"] == results["corpus"]["messages"]
        assert engine["build"]["stats"] == {"entities": 10, "relations": 20, "communities": None}
        assert engine["summary"]["fact_recall"] == pytest.approx((1.0 + 0.5 + 1.0) / 3, abs=1e-4)
        assert list(engine["by_kind"]) == list(GRAPH_GOLDEN_KINDS)
        assert len(engine["queries"]) == 3

        written = Path(results["results_path"])
        assert written.parent == tmp_path / "results"
        assert json.loads(written.read_text(encoding="utf-8"))["kind"] == "graph_eval"

    def test_a_relative_eval_root_still_hands_engines_absolute_workdirs(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        """Live-gate regression (2026-07-26): an engine rejected a relative path."""
        relative_root = Path(os.path.relpath(tmp_path / "graph", _REPO_ROOT))
        assert not relative_root.is_absolute()
        run_graph_eval(
            engines=["_probe"],
            eval_root=relative_root,
            golden_path=write_golden(tmp_path / "golden.jsonl"),
            results_dir=tmp_path / "results",
            verbose=0,
        )
        workdir = probe.sessions[0].workdir
        assert workdir.is_absolute()
        assert workdir == (tmp_path / "graph" / "smoke" / "_probe").resolve()

    def test_the_incremental_check_rebuilds_an_overlapping_batch(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        """★ Criterion §8.2#3: the marginal cost of a re-export."""
        results = self._run(tmp_path)
        corpus_messages = results["corpus"]["messages"]
        incremental = results["engines"]["_probe"]["incremental"]
        assert incremental["new_messages"] == INCREMENTAL_DELTA_MESSAGES
        assert incremental["messages_seen"] == corpus_messages + INCREMENTAL_DELTA_MESSAGES
        assert incremental["stats_before"] == {
            "entities": 10,
            "relations": 20,
            "communities": None,
        }
        assert incremental["stats"]["entities"] == 20  # grew across the second build
        assert results["incremental_new_messages"] == INCREMENTAL_DELTA_MESSAGES
        # Both builds went through the engine: the corpus, then the superset.
        assert probe.builds == [[corpus_messages], [corpus_messages + INCREMENTAL_DELTA_MESSAGES]]

    def test_the_delta_batch_is_a_superset_of_the_corpus(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        """The delta must overlap, or it measures a fresh index, not a re-index."""
        self._run(tmp_path)
        indexed = {message.guid for message in probe.sessions[0].builds[0][0].messages}
        delta = {message.guid for message in probe.sessions[0].builds[1][0].messages}
        assert indexed < delta
        assert len(delta - indexed) == INCREMENTAL_DELTA_MESSAGES
        assert all(guid.startswith("fill-") for guid in delta - indexed)

    def test_the_full_profile_skips_the_incremental_check(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        results = self._run(tmp_path, profile="full", seed=7)
        assert results["corpus"]["messages"] > 10_000
        assert results["corpus"]["filler"] > 0
        assert results["engines"]["_probe"]["incremental"] is None
        assert results["incremental_new_messages"] is None
        assert results["seed"] == 7

    def test_a_capped_run_gets_its_own_corpus_and_working_directory(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        """★ The Phase-5 cap: an engine too slow for 10,001 messages.

        Everything the capped run touches is named for the cap, so it cannot
        disturb an uncapped run indexing the same profile beside it.
        """
        results = self._run(tmp_path, profile="full", message_target=250)
        corpus_dir = tmp_path / "graph" / "corpus"

        assert results["profile"] == "full"
        assert results["message_target"] == 250
        assert results["corpus_stem"] == "full-mt250"
        assert results["corpus"]["messages"] == 250
        assert results["corpus"]["filler"] == 250 - results["corpus"]["scripted"]

        assert Path(results["corpus_path"]) == corpus_dir / "full-mt250.db"
        assert (corpus_dir / f"full-mt250{MANIFEST_SUFFIX}").is_file()
        # The uncapped names are the ones a parallel full run owns.
        assert not (corpus_dir / "full.db").exists()
        assert not (corpus_dir / f"full{MANIFEST_SUFFIX}").exists()

        workdir = Path(results["engines"]["_probe"]["workdir"])
        assert workdir == (tmp_path / "graph" / "full-mt250" / "_probe").resolve()
        assert not (tmp_path / "graph" / "full").exists()

        # The goldens still resolved: scripted messages ride in every corpus.
        assert results["engines"]["_probe"]["build"]["messages_seen"] == 250
        assert results["engines"]["_probe"]["summary"]["n_queries"] == 3

    def test_a_capped_run_leaves_an_uncapped_corpus_alone(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        """★ The live constraint: a full-profile run is mid-flight beside this."""
        corpus_dir = tmp_path / "graph" / "corpus"
        uncapped_path, uncapped = prepare_corpus(corpus_dir, profile="full")
        assert uncapped.message_count > 10_000

        # --skip-build must not mistake the uncapped corpus for this one.
        results = self._run(tmp_path, profile="full", message_target=250, skip_build=True)
        assert results["corpus"]["messages"] == 250

        after = FixtureManifest.model_validate_json(
            (corpus_dir / f"full{MANIFEST_SUFFIX}").read_text(encoding="utf-8")
        )
        assert after.message_count == uncapped.message_count
        assert uncapped_path.is_file()

    def test_skip_build_reuses_the_capped_run_s_own_corpus_and_workdir(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        """Re-scoring a capped run must find the graph the cap built."""
        first = self._run(tmp_path, profile="full", message_target=250)
        sentinel = Path(first["engines"]["_probe"]["workdir"]) / "engine-state.bin"
        sentinel.write_bytes(b"expensive")
        builds_before = len(probe.builds)

        second = self._run(tmp_path, profile="full", message_target=250, skip_build=True)
        assert sentinel.is_file()
        assert second["corpus_path"] == first["corpus_path"]
        assert second["engines"]["_probe"]["workdir"] == first["engines"]["_probe"]["workdir"]
        assert len(probe.builds) == builds_before  # nothing was re-indexed
        assert second["engines"]["_probe"]["summary"]["n_queries"] == 3

    def test_a_cap_below_the_script_still_carries_every_golden_anchor(
        self,
        tmp_path: Path,
        at_repo_root: None,
        probe: ProbeEngine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The scripted messages are the floor — a cap under it is reported."""
        with caplog.at_level("WARNING", logger="varagity.eval.graph_eval"):
            results = self._run(tmp_path, profile="full", message_target=50)
        assert results["corpus"]["messages"] == results["corpus"]["scripted"] > 50
        assert results["corpus"]["filler"] == 0
        assert results["engines"]["_probe"]["summary"]["n_queries"] == 3
        assert "message_target 50 is below" in caplog.text

    def test_a_non_positive_message_target_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive count"):
            run_graph_eval(profile="full", message_target=0, engines=["_probe"], eval_root=tmp_path)

    def test_skip_build_reuses_the_corpus_and_the_workdir(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        first = self._run(tmp_path)
        workdir = Path(first["engines"]["_probe"]["workdir"])
        sentinel = workdir / "engine-state.bin"
        sentinel.write_bytes(b"expensive")
        builds_before = len(probe.builds)

        second = self._run(tmp_path, skip_build=True)
        assert sentinel.is_file()  # the workdir was not wiped
        assert second["skip_build"] is True
        assert second["engines"]["_probe"]["build"] is None
        assert second["engines"]["_probe"]["incremental"] is None
        assert len(probe.builds) == builds_before  # nothing was re-indexed
        assert second["engines"]["_probe"]["summary"]["n_queries"] == 3

    def test_a_rebuild_wipes_the_previous_working_directory(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        first = self._run(tmp_path)
        stale = Path(first["engines"]["_probe"]["workdir"]) / "stale.bin"
        stale.write_bytes(b"old graph")
        self._run(tmp_path)
        assert not stale.exists()

    def test_an_unknown_engine_fails_before_anything_is_built(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        with pytest.raises(ValueError, match="unknown graph engine"):
            run_graph_eval(
                engines=["_probe", "nope"],
                eval_root=tmp_path / "graph",
                golden_path=write_golden(tmp_path / "golden.jsonl"),
                results_dir=tmp_path / "results",
                verbose=0,
            )
        assert probe.sessions == []
        assert not (tmp_path / "graph").exists()

    def test_an_unknown_profile_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown corpus profile"):
            run_graph_eval(profile="enormous", engines=["_probe"], eval_root=tmp_path)

    def test_a_drifted_golden_anchor_fails_before_any_engine_runs(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        golden = write_golden(
            tmp_path / "golden.jsonl",
            [{**GOLDEN_LINES[0], "required_guids": ["fx-ghost-001"]}],
        )
        with pytest.raises(ValueError, match="fx-ghost-001"):
            run_graph_eval(
                engines=["_probe"],
                eval_root=tmp_path / "graph",
                golden_path=golden,
                results_dir=tmp_path / "results",
                verbose=0,
            )
        assert probe.sessions == []

    def test_build_failures_and_query_errors_are_reported_not_raised(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        probe.build_failures = ["doc 3 timed out"]
        probe.raises = {GOLDEN_LINES[2]["query"]: "no such node"}
        results = self._run(tmp_path)
        engine = results["engines"]["_probe"]
        assert engine["build"]["failures"] == ["doc 3 timed out"]
        assert engine["summary"]["errors"] == 1

    def test_the_mode_override_reaches_the_engine(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine
    ) -> None:
        results = self._run(tmp_path, mode="global")
        assert results["mode"] == "global"
        assert {mode for _, mode in probe.asked} == {"global"}

    def test_settings_are_restored_after_the_run(
        self, tmp_path: Path, at_repo_root: None, probe: ProbeEngine, settings_env: Any
    ) -> None:
        settings_env(GRAPH_OWNER_ALIASES="Host Owner")
        self._run(tmp_path)
        assert get_settings().graph_owner_label == "Host Owner"


class TestHarnessDefaults:
    def test_the_eval_root_is_the_gitignored_data_directory(self) -> None:
        assert GRAPH_EVAL_ROOT.as_posix() == "data/eval/graph"

    def test_the_delta_is_small_enough_to_measure_the_margin(self) -> None:
        assert 0 < INCREMENTAL_DELTA_MESSAGES <= 50
