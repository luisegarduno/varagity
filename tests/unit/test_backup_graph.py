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
    NothingToExport,
    inspect_bundle,
    read_manifest,
    verify_bundle,
)
from varagity.backup.graph import (
    GraphExportReport,
    export_graph,
    graph_counts,
    is_temp_file,
    workdir_files,
)
from varagity.graph.base import GRAPH_ENGINE_REGISTRY
from varagity.graph.manifest import ManifestDoc, WorkdirManifest, save_manifest, save_summary
from varagity.graph.records import EngineFormat, GraphStats
from varagity.graph.service import GraphBuildInProgress, GraphService

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
