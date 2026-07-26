"""Unit tests for the synthetic message corpus and its golden QA (spec_graphrag §12).

Every assertion runs through the **product** path — build the database with
:func:`build_fixture_chat_db`, then read it back with the Phase-1 parser via
``batch_for_path`` — so a generator that writes something the parser can't
recover fails here rather than in the bake-off. The ``full`` profile is
exercised at a reduced message target (the ``message_target`` knob); the real
10⁴ build runs in Phase 5.

The module constants are repo-root-relative (the eval convention: every CLI
command runs from the root), so the paths are re-anchored on ``__file__``
here to keep the suite independent of the working directory.
"""

import json
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from varagity.eval.datasets import GRAPH_GOLDEN_KINDS, GraphGoldenEntry, load_graph_golden
from varagity.eval.graph_fixtures import (
    ATTRIBUTED_BODY_PATH,
    ATTRIBUTED_BODY_TEXT,
    FILLER_BLOCKLIST,
    GRAPH_GOLDEN_PATH,
    PROFILE_TARGETS,
    SCRIPT_PATH,
    TAPBACK_KINDS,
    FixtureManifest,
    apple_date,
    assert_filler_clean,
    build_fixture_chat_db,
)
from varagity.graph import MessageBatch, batch_for_path
from varagity.graph.sources.imessage import _TAPBACK_KINDS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / SCRIPT_PATH
_BLOB = _REPO_ROOT / ATTRIBUTED_BODY_PATH
_GOLDEN = _REPO_ROOT / GRAPH_GOLDEN_PATH

_SCRIPTED: dict[str, Any] = json.loads(_SCRIPT.read_text(encoding="utf-8"))
_BLOCKED = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in FILLER_BLOCKLIST) + r")s?\b",
    re.IGNORECASE,
)
# A minimal valid golden line, for the loader's error paths.
_GOLDEN_LINE = json.dumps(
    {
        "query": "who fixed it",
        "kind": "relation",
        "expected_facts": [["someone"]],
        "required_guids": ["fx-hw-001"],
    }
)


def _build(dest: Path, **kwargs: Any) -> FixtureManifest:
    """Build a fixture database from the shipped script and blob."""
    return build_fixture_chat_db(dest, script_path=_SCRIPT, blob_path=_BLOB, **kwargs)


def _build_and_parse(
    tmp_path: Path,
    settings_env: Callable[..., None],
    **kwargs: Any,
) -> tuple[FixtureManifest, MessageBatch]:
    """Build a fixture db, pin the manifest's names, and parse it back."""
    dest = tmp_path / "corpus" / "fixture.db"
    manifest = _build(dest, **kwargs)
    settings_env(
        GRAPH_OWNER_ALIASES=manifest.owner_label,
        GRAPH_HANDLE_NAMES=",".join(f"{h}={n}" for h, n in manifest.handle_names.items()),
    )
    return manifest, batch_for_path(dest, dest.parent)


class TestScriptedRoundTrip:
    def test_every_scripted_message_survives_the_round_trip(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """Guid, sender, thread, timestamp, and text all come back verbatim."""
        manifest, batch = _build_and_parse(tmp_path, settings_env)
        parsed = {message.guid: message for message in batch.messages}
        names = {person["key"]: person["name"] for person in _SCRIPTED["participants"]}
        threads = {thread["key"]: thread for thread in _SCRIPTED["threads"]}

        assert len(parsed) == len(_SCRIPTED["messages"]) == manifest.scripted_count
        for entry in _SCRIPTED["messages"]:
            message = parsed[entry["guid"]]
            assert message.text == entry["text"]
            assert message.timestamp == datetime.fromisoformat(entry["at"])
            assert message.thread_id == threads[entry["thread"]]["guid"]
            if entry["from"] == "me":
                assert message.is_from_me is True
                assert message.sender_name == manifest.owner_label
                assert message.sender_handle == ""
            else:
                assert message.is_from_me is False
                assert message.sender_name == names[entry["from"]]

    def test_manifest_matches_what_the_parser_finds(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        manifest, batch = _build_and_parse(tmp_path, settings_env)
        assert manifest.profile == "smoke"
        assert manifest.filler_count == 0
        assert manifest.message_count == len(batch.messages)
        assert sorted(manifest.guids) == sorted(m.guid for m in batch.messages)
        assert manifest.thread_counts == Counter(m.thread_id for m in batch.messages)
        assert manifest.first_timestamp == min(m.timestamp for m in batch.messages)
        assert manifest.last_timestamp == max(m.timestamp for m in batch.messages)
        assert manifest.handle_names and manifest.owner_label

    def test_tapbacks_fold_onto_their_targets_and_are_not_messages(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        manifest, batch = _build_and_parse(tmp_path, settings_env)
        parsed = {message.guid: message for message in batch.messages}
        names = {person["key"]: person["name"] for person in _SCRIPTED["participants"]}
        scripted = [entry for entry in _SCRIPTED["messages"] if entry.get("tapbacks")]

        assert scripted, "the script must exercise tapbacks"
        assert manifest.tapback_count == sum(len(e["tapbacks"]) for e in scripted)
        assert not [guid for guid in parsed if "-tb" in guid]  # reaction rows aren't messages
        for entry in scripted:
            folded = {(t.kind, t.sender_name) for t in parsed[entry["guid"]].tapbacks}
            assert folded == {
                (
                    tapback["kind"],
                    manifest.owner_label if tapback["from"] == "me" else names[tapback["from"]],
                )
                for tapback in entry["tapbacks"]
            }

    def test_both_epoch_eras_decode_to_the_scripted_times(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """The script spans the 2017 boundary, so both date formats are written."""
        _, batch = _build_and_parse(tmp_path, settings_env)
        parsed = {message.guid: message for message in batch.messages}
        assert parsed["fx-hw-001"].timestamp == datetime(2014, 3, 4, 18, 22, tzinfo=UTC)
        assert parsed["fx-hw-042"].timestamp == datetime(2024, 4, 2, 11, 20, tzinfo=UTC)
        years = {message.timestamp.year for message in batch.messages}
        assert min(years) < 2017 < max(years)
        assert max(years) - min(years) >= 10  # a decade-spanning corpus (§12 Q2)

    def test_attributed_body_rows_decode_inside_the_fixture(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """NULL-``text`` rows carry the captured blob, so decode runs in eval too."""
        _, batch = _build_and_parse(tmp_path, settings_env)
        parsed = {message.guid: message for message in batch.messages}
        attributed = [e["guid"] for e in _SCRIPTED["messages"] if e.get("attributed")]
        assert attributed, "the script must exercise the attributedBody path"
        for guid in attributed:
            assert parsed[guid].text == ATTRIBUTED_BODY_TEXT

    def test_thread_names_cover_both_naming_paths(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """A named group keeps its display name; an unnamed one lists participants."""
        _, batch = _build_and_parse(tmp_path, settings_env)
        parsed = {message.guid: message for message in batch.messages}
        assert parsed["fx-hw-001"].thread_name == "Hardware Talk"
        assert (
            parsed["fx-crew-001"].thread_name
            == "Bob Nakamura, Carol Mendez, Erin Vasquez, Jane Okafor"
        )
        assert parsed["fx-bob-001"].thread_name == "Bob Nakamura"  # 1:1, no display name

    def test_the_corpus_is_conversationally_shaped(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """Ten threads, both directions, and enough volume to be discriminative."""
        manifest, batch = _build_and_parse(tmp_path, settings_env)
        assert len(manifest.thread_counts) == 10
        assert manifest.scripted_count >= 200
        assert sum(1 for m in batch.messages if m.is_from_me) >= 40
        assert len({m.sender_name for m in batch.messages}) == 6  # 5 participants + the owner


class TestDeterminism:
    def test_same_seed_and_profile_give_identical_parsed_content(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """Determinism is over parsed content — SQLite page layout is not the contract."""
        first_manifest, first = _build_and_parse(
            tmp_path / "a", settings_env, profile="full", message_target=400
        )
        second_manifest, second = _build_and_parse(
            tmp_path / "b", settings_env, profile="full", message_target=400
        )
        assert first_manifest.model_dump() == second_manifest.model_dump()
        assert [m.model_dump() for m in first.messages] == [m.model_dump() for m in second.messages]

    def test_a_different_seed_changes_the_filler_but_not_the_script(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        _, first = _build_and_parse(
            tmp_path / "a", settings_env, profile="full", message_target=400, seed=13
        )
        _, second = _build_and_parse(
            tmp_path / "b", settings_env, profile="full", message_target=400, seed=99
        )
        scripted = {entry["guid"]: entry["text"] for entry in _SCRIPTED["messages"]}

        def texts(batch: MessageBatch, prefix: str) -> list[str]:
            return [m.text for m in batch.messages if m.guid.startswith(prefix)]

        assert texts(first, "fill-") != texts(second, "fill-")
        for batch in (first, second):
            assert {m.guid: m.text for m in batch.messages if m.guid.startswith("fx-")} == scripted


class TestFillerDiscipline:
    _TARGET = 600

    def test_filler_never_mentions_a_golden_term(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """The property that keeps the full profile discriminative, not merely large."""
        _, batch = _build_and_parse(
            tmp_path, settings_env, profile="full", message_target=self._TARGET
        )
        filler = [m for m in batch.messages if m.guid.startswith("fill-")]
        assert filler
        for message in filler:
            assert _BLOCKED.search(message.text) is None, message.text
            assert _BLOCKED.search(message.sender_name) is None
            assert _BLOCKED.search(message.thread_name) is None

    def test_filler_guids_never_collide_with_scripted_ones(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        manifest, batch = _build_and_parse(
            tmp_path, settings_env, profile="full", message_target=self._TARGET
        )
        guids = [m.guid for m in batch.messages]
        assert len(set(guids)) == len(guids)
        scripted = {entry["guid"] for entry in _SCRIPTED["messages"]}
        assert scripted <= set(guids)
        assert {g for g in guids if g.startswith("fill-")}.isdisjoint(scripted)
        assert manifest.filler_count == len(guids) - manifest.scripted_count

    def test_the_full_profile_reaches_its_message_target(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        manifest, batch = _build_and_parse(
            tmp_path, settings_env, profile="full", message_target=self._TARGET
        )
        assert manifest.message_count == len(batch.messages) == self._TARGET
        assert PROFILE_TARGETS["full"] > 10_000  # the shipped profile clears §12 Q2
        assert PROFILE_TARGETS["smoke"] == 0  # scripted only

    def test_filler_is_thread_shaped_and_spans_both_epoch_eras(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """Day-bursts across many threads, so transcripts have real day boundaries."""
        manifest, batch = _build_and_parse(
            tmp_path, settings_env, profile="full", message_target=self._TARGET
        )
        filler = [m for m in batch.messages if m.guid.startswith("fill-")]
        assert len(manifest.thread_counts) > 10  # scripted threads plus filler threads
        assert len({m.thread_id for m in filler}) >= 10
        years = {m.timestamp.year for m in filler}
        assert min(years) < 2017 <= max(years)  # both date formats written
        assert any(m.is_from_me for m in filler)
        assert not all(m.is_from_me for m in filler)

    def test_some_filler_threads_are_unnamed_groups(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        """The participant-name fallback runs at scale, not only on the crew thread."""
        _, batch = _build_and_parse(
            tmp_path, settings_env, profile="full", message_target=self._TARGET
        )
        filler_threads = {
            m.thread_id: m.thread_name for m in batch.messages if m.guid.startswith("fill-")
        }
        assert any("," in name for name in filler_threads.values())  # participant fallback
        assert any("," not in name for name in filler_threads.values())  # display name


class TestGeneratorGuards:
    def test_unknown_profile_is_rejected_with_the_options(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown fixture profile"):
            _build(tmp_path / "x.db", profile="enormous")

    def test_rebuilding_replaces_the_previous_database(
        self, tmp_path: Path, settings_env: Callable[..., None]
    ) -> None:
        dest = tmp_path / "corpus" / "fixture.db"
        _build(dest, profile="full", message_target=400)
        manifest = _build(dest, profile="smoke")
        settings_env(GRAPH_OWNER_ALIASES=manifest.owner_label)
        assert len(batch_for_path(dest, dest.parent).messages) == manifest.scripted_count

    def test_apple_date_writes_each_era_in_its_own_units(self) -> None:
        seconds = apple_date(datetime(2015, 6, 1, 12, 0, tzinfo=UTC))
        nanos = apple_date(datetime(2023, 3, 4, 18, 22, tzinfo=UTC))
        assert seconds < 10**12 < nanos  # the parser's per-row magnitude test
        assert apple_date(datetime(2001, 1, 1, tzinfo=UTC)) == 0

    def test_assert_filler_clean_matches_whole_words_only(self) -> None:
        assert_filler_clean("the alarm went off at the warm farm", context="message")
        assert_filler_clean("charming harmony", context="message")
        with pytest.raises(ValueError, match="golden term"):
            assert_filler_clean("ask Bob about it", context="message")
        with pytest.raises(ValueError, match="golden term"):
            assert_filler_clean("bring the keyboards", context="message")  # plural form

    def test_tapback_kinds_stay_pinned_to_the_parser(self) -> None:
        """The vocabulary is re-declared, not imported — this is the drift guard."""
        assert TAPBACK_KINDS == _TAPBACK_KINDS

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            ({}, "duplicate scripted guid"),
            ({"guid": "x", "thread": "nope"}, "unknown thread"),
            ({"guid": "x", "from": "nobody"}, "unknown sender"),
            ({"guid": "x", "attributed": True}, "flagged attributed"),
            ({"guid": "x", "tapbacks": [{"kind": "shrugged", "from": "me"}]}, "tapback kind"),
            ({"guid": "x", "tapbacks": [{"kind": "loved", "from": "who"}]}, "unknown sender"),
        ],
    )
    def test_an_inconsistent_script_fails_loudly(
        self, tmp_path: Path, extra: dict[str, Any], message: str
    ) -> None:
        script = json.loads(_SCRIPT.read_text(encoding="utf-8"))
        script["messages"].append({**script["messages"][0], **extra})
        script_path = tmp_path / "broken.json"
        script_path.write_text(json.dumps(script), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            build_fixture_chat_db(tmp_path / "x.db", script_path=script_path, blob_path=_BLOB)


class TestGraphGoldenQA:
    def test_the_shipped_paths_are_the_repo_root_relative_ones(self) -> None:
        assert SCRIPT_PATH.as_posix() == "tests/fixtures/graph/scripted_messages.json"
        assert GRAPH_GOLDEN_PATH.as_posix() == "data/eval/graph_golden_qa.jsonl"
        assert _SCRIPT.is_file() and _GOLDEN.is_file()

    def test_the_shipped_golden_set_loads(self) -> None:
        entries = load_graph_golden(_GOLDEN)
        assert 15 <= len(entries) <= 20
        assert all(entry.notes for entry in entries)  # every entry documents its distractors

    def test_every_kind_has_enough_entries_to_slice_on(self) -> None:
        counts = Counter(entry.kind for entry in load_graph_golden(_GOLDEN))
        assert set(counts) == set(GRAPH_GOLDEN_KINDS)
        assert min(counts.values()) >= 4

    def test_kind_literal_stays_pinned_to_the_vocabulary(self) -> None:
        """The tuple↔schema regression guard (the chunker/registry precedent)."""
        annotation = GraphGoldenEntry.model_fields["kind"].annotation
        assert set(getattr(annotation, "__args__", ())) == set(GRAPH_GOLDEN_KINDS)

    def test_every_required_guid_exists_in_the_smoke_corpus(self, tmp_path: Path) -> None:
        """Goldens are authored against scripted messages only (plan decision #11)."""
        manifest = _build(tmp_path / "smoke.db", profile="smoke")
        available = set(manifest.guids)
        for entry in load_graph_golden(_GOLDEN):
            for guid in entry.required_guids:
                assert guid.startswith("fx-"), guid  # never a filler guid
                assert guid in available, f"{entry.query!r} cites unknown message {guid!r}"

    def test_every_entry_is_anchored_in_the_scripted_conversations(self) -> None:
        """At least one fact variant per entry must actually appear in the corpus.

        Catches a typo'd or invented anchor. Groups may legitimately be
        answer-shaped (dates, negations), so the check is per entry, not per
        group.
        """
        corpus = " ".join(entry["text"] for entry in _SCRIPTED["messages"]).lower()
        for entry in load_graph_golden(_GOLDEN):
            grounded = [
                group
                for group in entry.expected_facts
                if any(variant.lower() in corpus for variant in group)
            ]
            assert grounded, f"{entry.query!r} has no fact anchored in the corpus"

    def test_a_malformed_line_names_the_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(f"{_GOLDEN_LINE}\nnot json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="golden.jsonl:2: not valid JSON"):
            load_graph_golden(path)

    def test_an_invalid_entry_names_the_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(_GOLDEN_LINE.replace("relation", "vibes"), encoding="utf-8")
        with pytest.raises(ValueError, match="golden.jsonl:1: invalid graph golden entry"):
            load_graph_golden(path)

    def test_an_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text("\n  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="golden dataset is empty"):
            load_graph_golden(path)

    def test_a_repeated_query_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(f"{_GOLDEN_LINE}\n{_GOLDEN_LINE}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="already appears on line 1"):
            load_graph_golden(path)

    def test_an_empty_fact_group_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is empty"):
            GraphGoldenEntry(query="q", kind="relation", expected_facts=[[]], required_guids=["g"])

    def test_a_blank_fact_variant_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="blank"):
            GraphGoldenEntry(
                query="q", kind="relation", expected_facts=[["  "]], required_guids=["g"]
            )

    def test_provenance_anchors_are_required(self) -> None:
        """Provenance recall divides by the anchor count, so it can't be empty."""
        with pytest.raises(ValueError, match="required_guids"):
            GraphGoldenEntry(query="q", kind="relation", expected_facts=[["a"]], required_guids=[])
