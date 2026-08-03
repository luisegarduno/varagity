"""Unit tests for the graph workdir's sidecar files.

The manifest is pure arithmetic over rendered transcripts plus two atomic
file writes, so all of it is testable without an engine, a workdir, or a
model. Two properties carry the feature:

* the **diff** must classify a corpus exactly — a document mis-filed as
  unchanged keeps a stale transcript in the graph forever (LightRAG's
  enqueue dedup will never take it back), and one mis-filed as removed
  deletes a real conversation;
* every degrade path must land on "no manifest", never on a half-trusted
  one.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from varagity.graph.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    SUMMARY_FILENAME,
    ManifestDoc,
    WorkdirManifest,
    content_sha256,
    load_manifest,
    load_summary,
    save_manifest,
    save_summary,
)
from varagity.graph.render import TranscriptDoc

THREAD = "iMessage;-;+15125550101"


def doc(key: str = f"{THREAD}::2016-03-04", *, text: str = "hello", guids: list[str] | None = None):
    return TranscriptDoc(
        doc_key=key,
        thread_id=key.partition("::")[0],
        thread_name="Hardware Talk",
        text=text,
        message_guids=["g1", "g2"] if guids is None else guids,
    )


class TestContentHash:
    def test_the_same_text_always_hashes_the_same(self) -> None:
        """Rendering is pure, so an unchanged conversation must hash stably."""
        assert content_sha256("hello") == content_sha256("hello")
        assert len(content_sha256("hello")) == 64

    def test_a_single_added_message_changes_the_hash(self) -> None:
        assert content_sha256("hello") != content_sha256("hello\nand more")


class TestDiff:
    def test_an_empty_manifest_makes_everything_new(self) -> None:
        diff = WorkdirManifest().diff([doc("a::1"), doc("b::1")])
        assert diff.new == ["a::1", "b::1"]
        assert (diff.changed, diff.unchanged, diff.removed) == ([], [], [])

    def test_identical_content_is_unchanged_and_costs_no_extraction(self) -> None:
        rendered = [doc()]
        manifest = WorkdirManifest().merged(rendered, prune=True)
        diff = manifest.diff(rendered)
        assert diff.unchanged == [doc().doc_key]
        assert diff.pending == []

    def test_a_grown_transcript_is_changed_not_new(self) -> None:
        """★ The trap: same key, different content — a delete must precede it."""
        manifest = WorkdirManifest().merged([doc(text="hello")], prune=True)
        diff = manifest.diff([doc(text="hello\nand it boots")])
        assert diff.changed == [doc().doc_key]
        assert diff.new == []
        assert diff.pending == [doc().doc_key]

    def test_a_document_the_render_omits_is_removed(self) -> None:
        manifest = WorkdirManifest().merged([doc("a::1"), doc("b::1")], prune=True)
        assert manifest.diff([doc("a::1")]).removed == ["b::1"]

    def test_removed_keys_are_sorted_having_no_render_order_to_inherit(self) -> None:
        manifest = WorkdirManifest().merged([doc("z::1"), doc("a::1"), doc("m::1")], prune=True)
        assert manifest.diff([]).removed == ["a::1", "m::1", "z::1"]

    def test_pending_keeps_new_before_changed(self) -> None:
        manifest = WorkdirManifest().merged([doc("a::1", text="old")], prune=True)
        diff = manifest.diff([doc("a::1", text="new"), doc("b::1")])
        assert diff.pending == ["b::1", "a::1"]

    def test_the_four_buckets_partition_the_union(self) -> None:
        manifest = WorkdirManifest().merged(
            [doc("keep::1"), doc("edit::1", text="old"), doc("gone::1")], prune=True
        )
        diff = manifest.diff([doc("keep::1"), doc("edit::1", text="new"), doc("add::1")])
        assert (diff.new, diff.changed, diff.unchanged, diff.removed) == (
            ["add::1"],
            ["edit::1"],
            ["keep::1"],
            ["gone::1"],
        )


class TestMerge:
    def test_a_full_render_replaces_the_manifest_wholesale(self) -> None:
        manifest = WorkdirManifest().merged([doc("a::1"), doc("b::1")], prune=True)
        assert set(manifest.merged([doc("a::1")], prune=True).docs) == {"a::1"}

    def test_a_bounded_render_keeps_what_it_did_not_mention(self) -> None:
        """★ Decision #9: a partial render may not speak for the whole archive."""
        manifest = WorkdirManifest().merged([doc("a::1"), doc("b::1")], prune=True)
        assert set(manifest.merged([doc("a::1")], prune=False).docs) == {"a::1", "b::1"}

    def test_merging_records_guids_thread_name_and_span(self) -> None:
        entry = WorkdirManifest().merged([doc(f"{THREAD}::2016-03-04..2016-03-06")], prune=True)
        (record,) = entry.docs.values()
        assert record.message_guids == ["g1", "g2"]
        assert record.thread_name == "Hardware Talk"
        assert record.span == "2016-03-04..2016-03-06"

    def test_merging_never_mutates_the_manifest_it_was_called_on(self) -> None:
        original = WorkdirManifest()
        original.merged([doc()], prune=True)
        assert original.docs == {}

    def test_a_retained_key_keeps_its_old_record_not_the_new_render(self) -> None:
        """★ A document the engine would not delete is still stale in the graph.

        Writing the new hash would make the very next build call it
        unchanged — and the stale transcript would become permanent, which is
        the failure the manifest exists to prevent.
        """
        manifest = WorkdirManifest().merged([doc("a::1", text="old")], prune=True)
        merged = manifest.merged([doc("a::1", text="new")], prune=True, retain=["a::1"])
        assert merged.docs["a::1"] == manifest.docs["a::1"]
        assert merged.diff([doc("a::1", text="new")]).changed == ["a::1"]

    def test_a_retained_key_survives_a_pruning_render_that_omits_it(self) -> None:
        """A vanished source whose delete failed is still in the graph."""
        manifest = WorkdirManifest().merged([doc("a::1"), doc("gone::1")], prune=True)
        merged = manifest.merged([doc("a::1")], prune=True, retain=["gone::1"])
        assert set(merged.docs) == {"a::1", "gone::1"}
        assert merged.diff([doc("a::1")]).removed == ["gone::1"]  # retried next build

    def test_retaining_a_key_the_manifest_never_had_adds_nothing(self) -> None:
        assert WorkdirManifest().merged([], prune=True, retain=["ghost::1"]).docs == {}

    def test_without_drops_only_the_named_keys(self) -> None:
        manifest = WorkdirManifest().merged([doc("a::1"), doc("b::1")], prune=True)
        assert set(manifest.without(["a::1", "never-indexed"]).docs) == {"b::1"}

    def test_a_merged_manifest_carries_the_current_schema_version(self) -> None:
        assert WorkdirManifest(version=99).merged([], prune=True).version == MANIFEST_VERSION
        assert WorkdirManifest(version=99).without([]).version == MANIFEST_VERSION


class TestProjections:
    def test_the_guid_index_is_the_provenance_map(self) -> None:
        manifest = WorkdirManifest().merged([doc("a::1", guids=["g1"]), doc("b::1")], prune=True)
        assert manifest.guid_index() == {"a::1": ["g1"], "b::1": ["g1", "g2"]}

    def test_the_guid_index_is_a_copy_callers_cannot_corrupt(self) -> None:
        manifest = WorkdirManifest().merged([doc()], prune=True)
        manifest.guid_index()[doc().doc_key].append("intruder")
        assert manifest.guid_index()[doc().doc_key] == ["g1", "g2"]

    def test_messages_are_counted_distinctly_across_documents(self) -> None:
        manifest = WorkdirManifest().merged(
            [doc("a::1", guids=["g1", "g2"]), doc("b::1", guids=["g2", "g3"])], prune=True
        )
        assert manifest.message_guid_count() == 3

    def test_an_empty_manifest_counts_nothing(self) -> None:
        assert WorkdirManifest().message_guid_count() == 0


class TestPersistence:
    def test_a_saved_manifest_round_trips(self, tmp_path: Path) -> None:
        manifest = WorkdirManifest().merged([doc()], prune=True)
        save_manifest(tmp_path, manifest)
        assert load_manifest(tmp_path) == manifest

    def test_the_manifest_lands_beside_the_engines_own_files(self, tmp_path: Path) -> None:
        save_manifest(tmp_path, WorkdirManifest())
        assert (tmp_path / MANIFEST_FILENAME).is_file()

    def test_saving_creates_the_workdir(self, tmp_path: Path) -> None:
        save_manifest(tmp_path / "fresh", WorkdirManifest())
        assert (tmp_path / "fresh" / MANIFEST_FILENAME).is_file()

    def test_a_save_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        save_manifest(tmp_path, WorkdirManifest().merged([doc()], prune=True))
        assert [path.name for path in tmp_path.iterdir()] == [MANIFEST_FILENAME]

    def test_a_second_save_replaces_the_first(self, tmp_path: Path) -> None:
        save_manifest(tmp_path, WorkdirManifest().merged([doc("a::1")], prune=True))
        save_manifest(tmp_path, WorkdirManifest().merged([doc("b::1")], prune=True))
        assert set(load_manifest(tmp_path).docs) == {"b::1"}

    def test_a_missing_manifest_is_simply_empty(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path) == WorkdirManifest()

    @pytest.mark.parametrize("payload", ["{not json", "[]", '{"docs": 7}'])
    def test_a_malformed_manifest_degrades_to_empty(
        self, tmp_path: Path, payload: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / MANIFEST_FILENAME).write_text(payload, encoding="utf-8")
        with caplog.at_level("WARNING", logger="varagity.graph.manifest"):
            assert load_manifest(tmp_path) == WorkdirManifest()
        assert "unreadable" in caplog.text

    def test_a_foreign_schema_version_degrades_to_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """★ Trusting fields that may mean something else would corrupt the graph."""
        save_manifest(
            tmp_path,
            WorkdirManifest(
                version=MANIFEST_VERSION + 1, docs={"a::1": ManifestDoc(content_sha256="x")}
            ),
        )
        with caplog.at_level("WARNING", logger="varagity.graph.manifest"):
            assert load_manifest(tmp_path).docs == {}
        assert "version" in caplog.text

    def test_an_unwritable_path_is_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A build must not die because a sidecar could not be written."""
        blocked = tmp_path / "wd"
        blocked.mkdir()
        (blocked / MANIFEST_FILENAME).mkdir()  # a directory where the file goes
        with caplog.at_level("WARNING", logger="varagity.graph.manifest"):
            save_manifest(blocked, WorkdirManifest())
        assert "could not write" in caplog.text


class TestSummary:
    def test_the_summary_records_the_graph_and_the_manifest(self, tmp_path: Path) -> None:
        manifest = WorkdirManifest().merged([doc("a::1"), doc("b::1", guids=["g3"])], prune=True)
        written = save_summary(tmp_path, manifest, entities=12, relations=7)
        loaded = load_summary(tmp_path)
        assert loaded == written
        assert (written.entities, written.relations) == (12, 7)
        assert (written.docs, written.message_guids) == (2, 3)

    def test_the_refresh_stamp_is_aware_and_current(self, tmp_path: Path) -> None:
        written = save_summary(tmp_path, WorkdirManifest(), entities=0, relations=0)
        assert written.refreshed_at.tzinfo is not None
        assert datetime.now(UTC) - written.refreshed_at < timedelta(minutes=1)

    def test_unknown_counts_stay_unknown_rather_than_zero(self, tmp_path: Path) -> None:
        """The honesty rule the records carry: None means "the engine won't say"."""
        written = save_summary(tmp_path, WorkdirManifest(), entities=None, relations=None)
        assert (written.entities, written.relations) == (None, None)

    def test_a_missing_summary_reads_as_none(self, tmp_path: Path) -> None:
        assert load_summary(tmp_path) is None

    def test_a_malformed_summary_is_ignored_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / SUMMARY_FILENAME).write_text("{nope", encoding="utf-8")
        with caplog.at_level("WARNING", logger="varagity.graph.manifest"):
            assert load_summary(tmp_path) is None
        assert "unreadable" in caplog.text

    def test_the_sidecar_is_small_readable_json(self, tmp_path: Path) -> None:
        """Prometheus reads this every scrape window — it must never be a graph walk."""
        save_summary(tmp_path, WorkdirManifest(), entities=1, relations=2)
        payload = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
        assert set(payload) == {"entities", "relations", "message_guids", "docs", "refreshed_at"}
