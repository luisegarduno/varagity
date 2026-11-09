"""Unit tests for the graph bundle — the engine's workdir, verbatim (ADR-019).

The workdir is laid out with the file names LightRAG 1.5.4 actually writes,
plus the repo's two sidecars written by their real writers; the service is
the real :class:`~varagity.graph.service.GraphService` over a fake engine
(``test_graph_service.py``'s), so the write lock under test is the one every
build takes. Bundles are real zips in ``tmp_path``.
"""

import hashlib
import logging
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_graph_service import FAKE_FORMAT, FakeEngine
from varagity.backup import graph as graph_backup
from varagity.backup.bundle import (
    MANIFEST_NAME,
    BundleDamaged,
    BundleIncompatible,
    BundleInvalid,
    NothingToExport,
    inspect_bundle,
    read_manifest,
    verify_bundle,
)
from varagity.backup.graph import (
    GraphExportReport,
    GraphRestore,
    export_graph,
    graph_counts,
    is_temp_file,
    restore_graph,
    workdir_files,
)
from varagity.graph.base import GRAPH_ENGINE_REGISTRY
from varagity.graph.manifest import ManifestDoc, WorkdirManifest, save_manifest, save_summary
from varagity.graph.records import EngineFormat, GraphStats
from varagity.graph.service import GraphBuildInProgress, GraphService, GraphUnavailable

DIM = 4

# LightRAG's twelve namespace files (no workspace: flat in the workdir).
ENGINE_FILES = {
    "kv_store_full_docs.json": b'{"t1::2026-01-01": {"content": "Thread: Bob"}}',
    "kv_store_text_chunks.json": b'{"t1::2026-01-01-chunk-000": {"tokens": 12}}',
    "kv_store_llm_response_cache.json": b'{"default:extract:9f": {"return": "(entity)"}}',
    "kv_store_full_entities.json": b'{"t1::2026-01-01": {"entity_names": ["BOB"]}}',
    "kv_store_full_relations.json": b'{"t1::2026-01-01": {"relation_pairs": []}}',
    "kv_store_entity_chunks.json": b'{"BOB": {"chunk_ids": ["c0"], "count": 1}}',
    "kv_store_relation_chunks.json": b"{}",
    "kv_store_doc_status.json": b'{"t1::2026-01-01": {"status": "processed"}}',
    "vdb_entities.json": b'{"embedding_dim": 1024, "data": [], "matrix": "AAAA"}',
    "vdb_relationships.json": b'{"embedding_dim": 1024, "data": [], "matrix": ""}',
    "vdb_chunks.json": b'{"embedding_dim": 1024, "data": [], "matrix": "AAAA"}',
    "graph_chunk_entity_relation.graphml": b"<graphml><graph><node id='BOB'/></graph></graphml>",
}
SIDECARS = ("varagity_graph_summary.json", "varagity_manifest.json")


@pytest.fixture
def stack(settings_env: Callable[..., None], tmp_path: Path) -> Path:
    """This stack's embedding space and corpus roots; returns the exports folder."""
    settings_env(
        EMBEDDING_MODEL="test-model",
        EMBEDDING_DIM=DIM,
        GRAPH_DOCS_PATH=str(tmp_path / "graph-docs"),
        GRAPH_STORAGE_PATH=str(tmp_path / "graph-data"),
    )
    return tmp_path / "graph-docs" / "varagity-exports"


def build_workdir(workdir: Path, *, entities: int | None = 5, relations: int | None = 3) -> Path:
    """Lay out a built graph's workdir: the engine's files and both sidecars."""
    workdir.mkdir(parents=True, exist_ok=True)
    for name, data in ENGINE_FILES.items():
        (workdir / name).write_bytes(data)
    manifest = WorkdirManifest(
        docs={
            "t1::2026-01-01": ManifestDoc(content_sha256="a" * 64, message_guids=["m1", "m2"]),
            "t1::2026-01-02..2026-01-04": ManifestDoc(
                content_sha256="b" * 64, message_guids=["m3"]
            ),
        }
    )
    save_manifest(workdir, manifest)
    save_summary(workdir, manifest, entities=entities, relations=relations)
    return workdir


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setitem(GRAPH_ENGINE_REGISTRY, "_backup_probe", fake)
    return fake


def graph_service(workdir: Path) -> GraphService:
    return GraphService("_backup_probe", workdir=workdir)


@pytest.fixture
def built(tmp_path: Path, engine: FakeEngine) -> GraphService:
    """A real graph service over a built workdir."""
    return graph_service(build_workdir(tmp_path / "graph-data" / "lightrag"))


def export(service: GraphService, dest: Path, **kwargs: Any) -> GraphExportReport:
    kwargs.setdefault("read_stale", lambda: False)
    return export_graph(service, dest, **kwargs)


def members_of(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist() if name != MANIFEST_NAME}


class TestExport:
    def test_zips_exactly_the_workdirs_files(self, built: GraphService, stack: Path) -> None:
        """★ Verbatim: every engine file and both sidecars, byte for byte, under lightrag/."""
        workdir = built.workdir
        (workdir / "kv_store_doc_status.json.tmp.4242.139.1731000000000").write_bytes(b"half")
        (workdir / "varagity_manifest.json.4242.tmp").write_bytes(b"half")
        result = export(built, stack)

        members = members_of(result.bundle.path)
        expected = sorted([*ENGINE_FILES, *SIDECARS])
        assert sorted(members) == [f"lightrag/{name}" for name in expected]
        for name in expected:
            assert members[f"lightrag/{name}"] == (workdir / name).read_bytes(), name
        assert result.files == len(expected)

    def test_members_are_deflated_and_hashed_in_name_order(
        self, built: GraphService, stack: Path
    ) -> None:
        result = export(built, stack)
        manifest = read_manifest(result.bundle.path)
        names = [member.name for member in manifest.members]
        assert names == sorted(names)
        with zipfile.ZipFile(result.bundle.path) as archive:
            assert archive.namelist()[-1] == MANIFEST_NAME
            for member in manifest.members:
                data = archive.read(member.name)
                assert member.size_bytes == len(data)
                assert member.sha256 == hashlib.sha256(data).hexdigest()
                assert archive.getinfo(member.name).compress_type == zipfile.ZIP_DEFLATED

    def test_the_manifest_records_the_engine_format_and_the_summary_counts(
        self, built: GraphService, stack: Path
    ) -> None:
        result = export(built, stack, read_stale=lambda: True)
        manifest = read_manifest(result.bundle.path)
        assert manifest.corpus == "graph"
        assert manifest.engine == FAKE_FORMAT
        assert manifest.counts == {"thread_days": 2, "messages": 3, "entities": 5, "relations": 3}
        assert manifest.stale is True
        assert (manifest.embedding_model, manifest.embedding_dim) == ("test-model", DIM)
        assert manifest.ingest_settings is None  # a RAG notion
        assert result.counts == manifest.counts

    def test_the_bundle_is_named_for_the_graph_and_passes_its_own_verdict(
        self, built: GraphService, stack: Path
    ) -> None:
        """The export never writes a member the restore's verdict would refuse."""
        result = export(built, stack)
        assert result.bundle.name.startswith("varagity-graph-")
        assert result.bundle.path.parent == stack
        verdict = inspect_bundle(result.bundle.path, "graph")
        assert verdict.restorable, verdict.reason
        assert verify_bundle(result.bundle.path, "graph") == result.bundle.manifest

    def test_the_folder_is_the_engines_name(
        self, tmp_path: Path, stack: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = EngineFormat(name="probe_engine", package="probe-graph", version="2.0.1")
        monkeypatch.setitem(GRAPH_ENGINE_REGISTRY, "_backup_probe", FakeEngine(storage=other))
        service = graph_service(build_workdir(tmp_path / "graph-data" / "probe_engine"))
        members = members_of(export(service, stack).bundle.path)
        assert all(name.startswith("probe_engine/") for name in members)

    def test_an_engine_name_that_cannot_be_a_folder_is_refused(
        self, tmp_path: Path, stack: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = EngineFormat(name="../escape", package="probe-graph", version="1.0.0")
        monkeypatch.setitem(GRAPH_ENGINE_REGISTRY, "_backup_probe", FakeEngine(storage=bad))
        service = graph_service(build_workdir(tmp_path / "graph-data" / "lightrag"))
        with pytest.raises(ValueError, match="can't name a bundle folder"):
            export(service, stack)
        assert not stack.exists() or list(stack.iterdir()) == []
        assert service.building is False

    def test_a_graph_with_no_summary_yet_exports_without_counts(
        self, built: GraphService, stack: Path
    ) -> None:
        """A first build killed partway leaves files but no summary — still a backup."""
        (built.workdir / "varagity_graph_summary.json").unlink()
        manifest = read_manifest(export(built, stack).bundle.path)
        assert manifest.counts == {}

    def test_sizes_the_graph_would_not_report_are_left_out(
        self, tmp_path: Path, engine: FakeEngine, stack: Path
    ) -> None:
        workdir = build_workdir(tmp_path / "graph-data" / "lightrag", entities=None, relations=None)
        manifest = read_manifest(export(graph_service(workdir), stack).bundle.path)
        assert manifest.counts == {"thread_days": 2, "messages": 3}

    def test_progress_walks_snapshot_pack_finalize_counting_bytes(
        self, built: GraphService, stack: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(graph_backup, "_COPY_BYTES", 16)  # many small blocks
        frames: list[tuple[str, int | None, int | None]] = []
        export(built, stack, on_progress=lambda *frame: frames.append(frame))
        total = sum(path.stat().st_size for path in workdir_files(built.workdir))
        assert frames[0] == ("snapshot", None, None)
        assert frames[-1] == ("finalize", None, None)
        packing = [frame for frame in frames if frame[0] == "pack"]
        assert packing[0] == ("pack", 0, total)
        assert packing[-1] == ("pack", total, total)
        assert len(packing) <= graph_backup._PROGRESS_FRAMES + 2  # throttled, not per block
        done = [current for _, current, _ in packing]
        assert done == sorted(done)

    def test_a_file_that_grew_since_listing_never_overflows_the_bar(self) -> None:
        frames: list[tuple[str, int | None, int | None]] = []
        progress = graph_backup._ByteProgress(lambda *frame: frames.append(frame), 100)
        progress.advance(80)
        progress.advance(40)  # 120 of a listed 100
        progress.finish()
        assert frames[-1] == ("pack", 120, 120)
        assert all(current <= total for _, current, total in frames)

    def test_the_export_logs_file_names_not_container_paths(
        self, built: GraphService, stack: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="varagity.backup.graph"):
            result = export(built, stack)
        lines = [record.getMessage() for record in caplog.records]
        assert any(
            "exporting 14 graph file(s)" in line and "lightrag-hku 1.5.4" in line for line in lines
        )
        assert any(f"to {result.bundle.name}" in line for line in lines)
        assert all(str(built.workdir) not in line for line in lines)


class TestTheWriteLock:
    def test_refuses_while_a_build_holds_the_write_lock(
        self, built: GraphService, stack: Path
    ) -> None:
        """★ No copy of a workdir a build is changing — and no bundle left behind."""
        with built.writing(), pytest.raises(GraphBuildInProgress):
            export(built, stack)
        assert not stack.exists() or list(stack.iterdir()) == []

    def test_holds_the_lock_while_it_copies_and_reads_keep_answering(
        self, built: GraphService, stack: Path
    ) -> None:
        """★ No build can start mid-copy; graph chat answers throughout."""
        seen: list[Any] = []

        def during(stage: str, current: int | None, total: int | None) -> None:
            if stage == "pack" and current == 0:
                seen.append(built.building)
                with pytest.raises(GraphBuildInProgress):
                    built.build([])
                seen.append(built.stats())  # reads take no lock

        export(built, stack, on_progress=during)
        assert seen == [True, GraphStats(entities=4, relations=3, communities=None)]
        assert built.building is False  # released once published
        built.close()

    def test_the_stale_flag_is_read_after_the_copy_under_the_lock(
        self, built: GraphService, stack: Path
    ) -> None:
        """★ A source deleted mid-export flags the graph stale, so the bundle must say so.

        The delete route sets the flag without taking any lock. Read before
        the copy, the flag would record a graph still holding the deleted
        file's messages as fresh, and a restore would copy that back.
        """
        stages: list[str] = []
        reads: list[tuple[bool, list[str]]] = []

        def read_stale() -> bool:
            reads.append((built.building, list(stages)))
            return True  # the delete landed while the files were being copied

        result = export(
            built,
            stack,
            read_stale=read_stale,
            on_progress=lambda stage, current, total: stages.append(stage),
        )
        assert len(reads) == 1
        held, before = reads[0]
        assert held is True  # still under the write lock
        assert "pack" in before and before[-1] == "finalize"  # every file was copied first
        assert read_manifest(result.bundle.path).stale is True

    def test_a_failed_copy_leaves_no_bundle_and_frees_the_lock(
        self, built: GraphService, stack: Path
    ) -> None:
        locked = built.workdir / "vdb_chunks.json"
        locked.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                export(built, stack)
        finally:
            locked.chmod(0o644)
        assert [path.name for path in stack.iterdir()] == []  # the partial is gone too
        assert built.building is False


class TestNothingToExport:
    @pytest.mark.parametrize("layout", ["missing", "empty", "only-temp-files"])
    def test_an_unbuilt_graph_has_nothing_to_export(
        self, tmp_path: Path, engine: FakeEngine, stack: Path, layout: str
    ) -> None:
        workdir = tmp_path / "graph-data" / "lightrag"
        if layout != "missing":
            workdir.mkdir(parents=True)
        if layout == "only-temp-files":
            (workdir / "kv_store_doc_status.json.tmp.1.2.3").write_bytes(b"half")
        service = graph_service(workdir)
        assert workdir_files(workdir) == []
        with pytest.raises(NothingToExport, match="nothing has been built"):
            export(service, stack)
        assert service.building is False
        assert engine.opens == 0  # an export never opens a session


class TestWorkdirListing:
    def test_entries_a_bundle_cannot_carry_are_skipped_with_a_warning(
        self, built: GraphService, stack: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        workdir = built.workdir
        (workdir / "nested").mkdir()
        (workdir / "nested" / "x.json").write_text("{}")
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        (workdir / "linked.json").symlink_to(outside)
        (workdir / ".hidden.json").write_text("{}")
        (workdir / "with space.json").write_text("{}")
        os.mkfifo(workdir / "pipe.json")  # reading it would block the export forever

        with caplog.at_level(logging.WARNING, logger="varagity.backup.graph"):
            result = export(built, stack)
        members = members_of(result.bundle.path)
        assert sorted(members) == sorted(f"lightrag/{name}" for name in [*ENGINE_FILES, *SIDECARS])
        warned = " ".join(record.getMessage() for record in caplog.records)
        for name, why in (
            ("nested", "a folder"),
            ("linked.json", "a symlink"),
            (".hidden.json", "a name a bundle can't carry"),
            ("with space.json", "a name a bundle can't carry"),
            ("pipe.json", "not a regular file"),
        ):
            assert f"not exporting {name} from the graph workdir — {why}" in warned

    def test_workdir_files_lists_what_the_export_packs(self, built: GraphService) -> None:
        names = [path.name for path in workdir_files(built.workdir)]
        assert names == sorted([*ENGINE_FILES, *SIDECARS])

    def test_a_missing_workdir_lists_nothing(self, tmp_path: Path) -> None:
        assert workdir_files(tmp_path / "nope") == []

    def test_counts_need_a_summary(self, tmp_path: Path) -> None:
        assert graph_counts(tmp_path) == {}

    @pytest.mark.parametrize(
        ("name", "temp"),
        [
            ("kv_store_doc_status.json.tmp.4242.139.1731000000000", True),  # LightRAG's
            ("varagity_manifest.json.4242.tmp", True),  # the sidecars' own
            ("kv_store_doc_status.json", False),
            ("graph_chunk_entity_relation.graphml", False),
        ],
    )
    def test_temp_files(self, name: str, temp: bool) -> None:
        assert is_temp_file(name) is temp


def test_a_real_workdir_lists_by_name_whatever_the_creation_order(tmp_path: Path) -> None:
    workdir = tmp_path / "lightrag"
    workdir.mkdir()
    for name in ("vdb_chunks.json", "kv_store_full_docs.json", "graph_x.graphml"):
        (workdir / name).write_text("{}")
    assert [path.name for path in workdir_files(workdir)] == [
        "graph_x.graphml",
        "kv_store_full_docs.json",
        "vdb_chunks.json",
    ]


# ── the graph restore (phase 4) ────────────────────────────────────────────


def snapshot(workdir: Path) -> dict[str, bytes]:
    """Every file in a workdir, by name — what a restore must reproduce byte for byte."""
    return {path.name: path.read_bytes() for path in sorted(workdir.iterdir()) if path.is_file()}


def rezip_member(bundle: Path, name: str, data: bytes) -> Path:
    """Rewrite one member of a bundle, keeping its original ``bundle.json`` (now a lie)."""
    with zipfile.ZipFile(bundle) as archive:
        contents = {entry: archive.read(entry) for entry in archive.namelist()}
    contents[name] = data
    target = bundle.with_name(f"edited-{bundle.name}")
    with zipfile.ZipFile(target, "w") as archive:
        for entry, payload in contents.items():
            archive.writestr(entry, payload)
    return target


def restore(service: GraphService, bundle: Path, **kwargs: Any) -> GraphRestore:
    kwargs.setdefault("run_id", "r1")
    return restore_graph(service, bundle, **kwargs)


@pytest.fixture
def exported(built: GraphService, stack: Path) -> tuple[Path, dict[str, bytes]]:
    """A bundle of the built graph, and the workdir's files at the moment it was made."""
    return export(built, stack).bundle.path, snapshot(built.workdir)


class TestRestore:
    def test_brings_the_exported_workdir_back_byte_for_byte(
        self, built: GraphService, exported: tuple[Path, dict[str, bytes]], engine: FakeEngine
    ) -> None:
        """★ No model call, no re-extraction — the files the graph was, verbatim."""
        bundle, before = exported
        workdir = built.workdir
        (workdir / "kv_store_full_docs.json").write_bytes(b'{"changed": "after the backup"}')
        (workdir / "vdb_extra.json").write_bytes(b"{}")  # a namespace written since

        result = restore(built, bundle)

        assert snapshot(workdir) == before
        assert sorted(path.name for path in workdir.parent.iterdir()) == ["lightrag"]
        assert result.files == len(before)
        assert result.counts == {"thread_days": 2, "messages": 3, "entities": 5, "relations": 3}
        assert result.stats == GraphStats(entities=4, relations=3, communities=None)
        assert result.bundle_name == bundle.name
        assert result.size_bytes == bundle.stat().st_size
        assert built.building is False
        assert engine.opens == 1  # opened once, on the restored files
        built.close()

    def test_restores_onto_a_wiped_stack(
        self, tmp_path: Path, engine: FakeEngine, stack: Path, built: GraphService
    ) -> None:
        bundle = export(built, stack).bundle.path
        before = snapshot(built.workdir)
        wiped = graph_service(tmp_path / "wiped" / "lightrag")
        restore(wiped, bundle)
        assert snapshot(wiped.workdir) == before
        wiped.close()

    def test_progress_walks_validate_extract_swap_open_counting_bytes(
        self,
        built: GraphService,
        exported: tuple[Path, dict[str, bytes]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(graph_backup, "_COPY_BYTES", 16)  # many small blocks
        bundle, before = exported
        frames: list[tuple[str, int | None, int | None]] = []
        restore(built, bundle, on_progress=lambda *frame: frames.append(frame))
        total = sum(len(data) for data in before.values())
        stages = [stage for stage, _, _ in frames]
        assert stages[0] == "validate"
        assert stages[-2:] == ["swap", "open"]
        extracting = [frame for frame in frames if frame[0] == "extract"]
        assert extracting[0] == ("extract", 0, total)
        assert extracting[-1] == ("extract", total, total)
        assert len(extracting) <= graph_backup._PROGRESS_FRAMES + 2  # throttled, not per block
        built.close()

    def test_the_stale_flag_rides_the_bundle(self, built: GraphService, stack: Path) -> None:
        """The flow writes the bundle's own flag: the graph is what was exported."""
        bundle = export(built, stack, read_stale=lambda: True).bundle.path
        assert restore(built, bundle).manifest.stale is True
        built.close()

    def test_the_restore_logs_file_names_not_container_paths(
        self,
        built: GraphService,
        exported: tuple[Path, dict[str, bytes]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        bundle, _ = exported
        with caplog.at_level(logging.INFO, logger="varagity.backup.graph"):
            restore(built, bundle)
        # Only the backup package's lines are relayed into the panel's log.
        lines = [
            record.getMessage()
            for record in caplog.records
            if record.name.startswith("varagity.backup")
        ]
        assert any(
            "restoring 14 graph file(s)" in line and "lightrag-hku 1.5.4" in line for line in lines
        )
        assert any(f"from {bundle.name}" in line for line in lines)
        assert all(str(built.workdir) not in line for line in lines)
        built.close()


class TestRestoreRefusals:
    @pytest.mark.parametrize(
        "entry", ["lightrag/../escaped.json", "/etc/escaped.json", "lightrag/nested/x.json"]
    )
    def test_unsafe_members_are_refused_before_anything_is_extracted(
        self,
        built: GraphService,
        exported: tuple[Path, dict[str, bytes]],
        tmp_path: Path,
        entry: str,
    ) -> None:
        """★ Zip-slip never reaches the disk: refused before the staging folder exists."""
        bundle, before = exported
        with zipfile.ZipFile(bundle, "a") as archive:
            archive.writestr(entry, b"{}")
        with pytest.raises(BundleInvalid):
            restore(built, bundle)
        assert snapshot(built.workdir) == before
        assert not built.staging_dir("r1").exists()
        assert not (tmp_path / "graph-data" / "escaped.json").exists()

    def test_a_member_damaged_after_verification_never_goes_live(
        self, built: GraphService, exported: tuple[Path, dict[str, bytes]]
    ) -> None:
        """★ Each member's sha256 is checked again as it is extracted."""
        bundle, before = exported
        name = "lightrag/kv_store_doc_status.json"
        with zipfile.ZipFile(bundle) as archive:
            original = archive.read(name)
        damaged = rezip_member(bundle, name, original.replace(b"processed", b"PROCESSED"))
        assert inspect_bundle(
            damaged, "graph"
        ).restorable  # same sizes: the cheap verdict can't tell
        with pytest.raises(BundleDamaged, match="kv_store_doc_status.json"):
            restore(built, damaged)
        assert snapshot(built.workdir) == before
        assert sorted(path.name for path in built.workdir.parent.iterdir()) == ["lightrag"]

    def test_a_bundle_from_another_engine_version_is_refused_untouched(
        self, tmp_path: Path, stack: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = EngineFormat(name="lightrag", package="lightrag-hku", version="1.4.9")
        monkeypatch.setitem(GRAPH_ENGINE_REGISTRY, "_backup_probe", FakeEngine(storage=old))
        service = graph_service(build_workdir(tmp_path / "graph-data" / "lightrag"))
        bundle = export(service, stack).bundle.path
        before = snapshot(service.workdir)
        with pytest.raises(BundleIncompatible, match="lightrag-hku 1.4.9"):
            restore(service, bundle)
        assert snapshot(service.workdir) == before
        assert not service.staging_dir("r1").exists()

    def test_a_restored_graph_that_will_not_open_leaves_the_graph_as_it_was(
        self,
        built: GraphService,
        exported: tuple[Path, dict[str, bytes]],
        engine: FakeEngine,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        bundle, before = exported
        (built.workdir / "kv_store_full_docs.json").write_bytes(b'{"live": "since the backup"}')
        live = snapshot(built.workdir)

        def refuse_the_backup(workdir: Path) -> None:
            if snapshot(workdir) == before:
                raise RuntimeError("vdb embedding_dim mismatch")

        engine.on_open = refuse_the_backup
        with (
            caplog.at_level(logging.WARNING, logger="varagity.backup.graph"),
            pytest.raises(GraphUnavailable, match="embedding_dim mismatch"),
        ):
            restore(built, bundle)
        assert snapshot(built.workdir) == live
        assert sorted(path.name for path in built.workdir.parent.iterdir()) == ["lightrag"]
        assert "the restored graph would not open, so the restore was undone" in caplog.text
        built.close()

    def test_a_held_write_lock_refuses_the_swap_untouched(
        self, built: GraphService, exported: tuple[Path, dict[str, bytes]]
    ) -> None:
        bundle, before = exported
        with built.writing(), pytest.raises(GraphBuildInProgress):
            restore(built, bundle)
        assert snapshot(built.workdir) == before
        assert not built.staging_dir("r1").exists()  # the extracted files went with it

    def test_a_staging_folder_already_there_is_never_reused_or_removed(
        self, built: GraphService, exported: tuple[Path, dict[str, bytes]]
    ) -> None:
        bundle, before = exported
        squatter = built.staging_dir("r1")
        squatter.mkdir()
        (squatter / "keep.json").write_text("{}")
        with pytest.raises(FileExistsError):
            restore(built, bundle)
        assert (squatter / "keep.json").exists()
        assert snapshot(built.workdir) == before
