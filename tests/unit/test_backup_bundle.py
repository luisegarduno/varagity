"""Unit tests for the backup bundle container (ADR-019).

Bundles are real zips in ``tmp_path`` — the ``test_graph_manifest.py``
pattern — so these exercise the actual write path: members streamed and
hashed, the manifest sealed last, the ``.partial`` → ``os.replace`` publish,
and the one-member manifest read the listing depends on.
"""

import hashlib
import os
import zipfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from varagity.backup import bundle as bundle_module
from varagity.backup.bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    MANIFEST_NAME,
    BundleInvalid,
    BundleManifest,
    bundle_name,
    corpus_root,
    display_path,
    exports_dir,
    is_bundle_name,
    list_bundles,
    read_manifest,
    sweep_partials,
    write_bundle,
)

WHEN = datetime(2026, 11, 8, 13, 12, 41, tzinfo=UTC)


@pytest.fixture
def exports(tmp_path: Path) -> Path:
    return tmp_path / "docs" / "varagity-exports"


def seal_defaults(**overrides: object) -> dict[str, Any]:
    values: dict[str, Any] = {
        "embedding_model": "test-model",
        "embedding_dim": 4,
        "stale": False,
        "counts": {"documents": 1, "chunks": 2},
        "ingest_settings": {"CHUNK_SIZE": "400"},
    }
    values.update(overrides)
    return values


def write_sample(
    directory: Path,
    *,
    members: dict[str, bytes] | None = None,
    when: datetime = WHEN,
    stored: frozenset[str] = frozenset({"embeddings.f32"}),
) -> Path:
    """Write a small sealed bundle and return its published path."""
    contents = members or {
        "documents.jsonl": b'{"doc_id":"a"}\n',
        "chunks.jsonl": b'{"chunk_id":"a::0"}\n{"chunk_id":"a::1"}\n',
        "embeddings.f32": bytes(range(32)),
    }
    with write_bundle(directory, "rag", created_at=when) as writer:
        for name, data in contents.items():
            with writer.member(name, compress=name not in stored) as sink:
                # Two writes per member: the hash must cover the whole stream.
                half = len(data) // 2
                sink.write(data[:half])
                sink.write(data[half:])
        writer.seal(**seal_defaults())
    return writer.result.path


class TestWriteAndRead:
    def test_manifest_round_trips(self, exports: Path) -> None:
        path = write_sample(exports)
        manifest = read_manifest(path)
        assert manifest.format == BUNDLE_FORMAT
        assert manifest.version == BUNDLE_VERSION
        assert manifest.corpus == "rag"
        assert manifest.created_at == WHEN
        assert manifest.embedding_model == "test-model"
        assert manifest.embedding_dim == 4
        assert manifest.stale is False
        assert manifest.counts == {"documents": 1, "chunks": 2}
        assert manifest.ingest_settings == {"CHUNK_SIZE": "400"}
        assert [member.name for member in manifest.members] == [
            "documents.jsonl",
            "chunks.jsonl",
            "embeddings.f32",
        ]

    def test_each_member_records_its_own_sha256_and_size(self, exports: Path) -> None:
        path = write_sample(exports)
        manifest = read_manifest(path)
        with zipfile.ZipFile(path) as archive:
            for member in manifest.members:
                data = archive.read(member.name)
                assert member.size_bytes == len(data)
                assert member.sha256 == hashlib.sha256(data).hexdigest()

    def test_the_manifest_is_the_last_member(self, exports: Path) -> None:
        path = write_sample(exports)
        with zipfile.ZipFile(path) as archive:
            assert archive.namelist()[-1] == MANIFEST_NAME
            assert archive.namelist().count(MANIFEST_NAME) == 1

    def test_compression_is_per_member_and_entries_are_plain_files(self, exports: Path) -> None:
        path = write_sample(exports)
        with zipfile.ZipFile(path) as archive:
            infos = {info.filename: info for info in archive.infolist()}
        assert infos["embeddings.f32"].compress_type == zipfile.ZIP_STORED
        assert infos["chunks.jsonl"].compress_type == zipfile.ZIP_DEFLATED
        assert infos[MANIFEST_NAME].compress_type == zipfile.ZIP_DEFLATED
        for info in infos.values():
            mode = info.external_attr >> 16
            assert mode & 0o170000 == 0o100000, info.filename  # S_IFREG: never a symlink

    def test_the_result_names_the_published_file(self, exports: Path) -> None:
        with write_bundle(exports, "rag", created_at=WHEN) as writer:
            with writer.member("documents.jsonl") as sink:
                sink.write(b"{}\n")
            writer.seal(**seal_defaults())
        result = writer.result
        assert result.name == "varagity-rag-20261108-1312.zip"
        assert result.path == exports / result.name
        assert result.size_bytes == result.path.stat().st_size
        assert result.manifest == read_manifest(result.path)
        assert sorted(item.name for item in exports.iterdir()) == [result.name]

    def test_read_manifest_opens_only_the_manifest_member(
        self, exports: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_sample(exports)
        opened: list[str] = []
        real_open = zipfile.ZipFile.open

        def spy(self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any) -> Any:
            opened.append(name if isinstance(name, str) else name.filename)
            return real_open(self, name, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", spy)
        read_manifest(path)
        assert opened == [MANIFEST_NAME]


class TestFailedWrites:
    def test_an_exception_mid_write_leaves_no_bundle(self, exports: Path) -> None:
        with (
            pytest.raises(RuntimeError, match="the store fell over"),
            write_bundle(exports, "rag", created_at=WHEN) as writer,
        ):
            with writer.member("documents.jsonl") as sink:
                sink.write(b"{}\n")
            raise RuntimeError("the store fell over")
        assert list(exports.iterdir()) == []

    def test_an_unsealed_bundle_is_never_published(self, exports: Path) -> None:
        with (
            pytest.raises(RuntimeError, match="never sealed"),
            write_bundle(exports, "rag", created_at=WHEN) as writer,
            writer.member("documents.jsonl") as sink,
        ):
            sink.write(b"{}\n")
        assert list(exports.iterdir()) == []
        with pytest.raises(RuntimeError, match="not been published"):
            _ = writer.result

    def test_member_rules(self, exports: Path) -> None:
        with write_bundle(exports, "rag", created_at=WHEN) as writer:
            with pytest.raises(ValueError, match="reserved or already written"):
                writer.member(MANIFEST_NAME).__enter__()
            with writer.member("a.jsonl") as sink:
                sink.write(b"x")
            with pytest.raises(ValueError, match="reserved or already written"):
                writer.member("a.jsonl").__enter__()
            writer.seal(**seal_defaults())
            with pytest.raises(RuntimeError, match="sealed"):
                writer.member("late.jsonl").__enter__()
            with pytest.raises(RuntimeError, match="already sealed"):
                writer.seal(**seal_defaults())

    def test_a_name_claimed_mid_write_is_never_replaced(self, exports: Path) -> None:
        with write_bundle(exports, "rag", created_at=WHEN) as writer:
            with writer.member("documents.jsonl") as sink:
                sink.write(b"{}\n")
            squatter = exports / "varagity-rag-20261108-1312.zip"
            squatter.write_bytes(b"someone else's file")
            writer.seal(**seal_defaults())
        assert squatter.read_bytes() == b"someone else's file"
        assert writer.result.name == "varagity-rag-20261108-1312-2.zip"


class TestNamesAndPartials:
    def test_names_stamp_utc_and_never_collide(self, exports: Path) -> None:
        exports.mkdir(parents=True)
        local = datetime(2026, 11, 8, 6, 12, tzinfo=UTC).astimezone()
        assert bundle_name(exports, "rag", now=local) == "varagity-rag-20261108-0612.zip"
        (exports / "varagity-rag-20261108-0612.zip").write_bytes(b"")
        (exports / ".varagity-rag-20261108-0612-2.zip.partial").write_bytes(b"")
        assert bundle_name(exports, "rag", now=local) == "varagity-rag-20261108-0612-3.zip"
        assert bundle_name(exports, "graph", now=local) == "varagity-graph-20261108-0612.zip"

    def test_same_minute_exports_get_suffixes(self, exports: Path) -> None:
        names = [write_sample(exports).name for _ in range(3)]
        assert names == [
            "varagity-rag-20261108-1312.zip",
            "varagity-rag-20261108-1312-2.zip",
            "varagity-rag-20261108-1312-3.zip",
        ]

    def test_partials_are_never_listed(self, exports: Path) -> None:
        write_sample(exports)
        (exports / ".varagity-rag-20261108-1400.zip.partial").write_bytes(b"in flight")
        assert [listing.name for listing in list_bundles(exports)] == [
            "varagity-rag-20261108-1312.zip"
        ]

    def test_an_in_flight_bundle_is_invisible_until_published(self, exports: Path) -> None:
        with write_bundle(exports, "rag", created_at=WHEN) as writer:
            with writer.member("documents.jsonl") as sink:
                sink.write(b"{}\n")
            assert list_bundles(exports) == []
            writer.seal(**seal_defaults())
        assert [listing.name for listing in list_bundles(exports)] == [writer.result.name]

    def test_the_next_export_sweeps_only_its_own_corpus_partials(self, exports: Path) -> None:
        exports.mkdir(parents=True)
        stale = exports / ".varagity-rag-20261101-0900.zip.partial"
        stale.write_bytes(b"crashed")
        other = exports / ".varagity-graph-20261101-0900.zip.partial"
        other.write_bytes(b"another corpus")
        write_sample(exports)
        assert not stale.exists()
        assert other.exists()

    def test_sweep_reports_what_it_removed(self, exports: Path) -> None:
        assert sweep_partials(exports, "rag") == 0  # no folder yet
        exports.mkdir(parents=True)
        (exports / ".varagity-rag-1.zip.partial").write_bytes(b"")
        (exports / ".varagity-rag-2.zip.partial").write_bytes(b"")
        assert sweep_partials(exports, "rag") == 2

    @pytest.mark.parametrize(
        ("name", "ok"),
        [
            ("varagity-rag-20261108-1312.zip", True),
            ("hand-copied.ZIP", True),
            (".varagity-rag-20261108-1312.zip.partial", False),
            (".hidden.zip", False),
            ("notes.txt", False),
            ("../escape.zip", False),
            ("a\\b.zip", False),
            ("", False),
        ],
    )
    def test_bundle_names(self, name: str, ok: bool) -> None:
        assert is_bundle_name(name) is ok


class TestReadManifestRefusals:
    def test_not_a_zip(self, tmp_path: Path) -> None:
        path = tmp_path / "fake.zip"
        path.write_bytes(b"this is not a zip")
        with pytest.raises(BundleInvalid, match="not a readable zip"):
            read_manifest(path)

    def test_a_zip_without_a_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("readme.txt", "hello")
        with pytest.raises(BundleInvalid, match="has no bundle.json"):
            read_manifest(path)

    def test_a_manifest_that_is_not_a_varagity_one(self, tmp_path: Path) -> None:
        path = tmp_path / "other.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(MANIFEST_NAME, '{"format": "something-else", "version": 1}')
        with pytest.raises(BundleInvalid, match="not a varagity manifest"):
            read_manifest(path)

    def test_an_absurdly_large_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bundle_module, "_MAX_MANIFEST_BYTES", 16)
        path = write_sample(tmp_path / "exports")
        with pytest.raises(BundleInvalid, match="not a manifest"):
            read_manifest(path)


class TestListing:
    def test_a_missing_folder_lists_nothing(self, exports: Path) -> None:
        assert list_bundles(exports) == []

    def test_newest_first_with_manifests(self, exports: Path) -> None:
        older = write_sample(exports, when=datetime(2026, 10, 19, 9, 4, tzinfo=UTC))
        newer = write_sample(exports)
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        listings = list_bundles(exports)
        assert [listing.name for listing in listings] == [newer.name, older.name]
        assert listings[0].size_bytes == newer.stat().st_size
        assert listings[0].modified_at == datetime.fromtimestamp(2_000_000, tz=UTC)
        assert isinstance(listings[0].manifest, BundleManifest)

    def test_a_foreign_zip_is_listed_without_a_manifest(self, exports: Path) -> None:
        exports.mkdir(parents=True)
        with zipfile.ZipFile(exports / "photos.zip", "w") as archive:
            archive.writestr("a.jpg", b"...")
        (exports / "notes.txt").write_text("not a zip")
        listings = list_bundles(exports)
        assert [(listing.name, listing.manifest) for listing in listings] == [("photos.zip", None)]

    def test_a_symlink_leading_out_of_the_folder_is_not_listed(
        self, exports: Path, tmp_path: Path
    ) -> None:
        outside = write_sample(tmp_path / "elsewhere")
        exports.mkdir(parents=True)
        (exports / "link.zip").symlink_to(outside)
        assert list_bundles(exports) == []


class TestLocations:
    @pytest.fixture
    def roots(self, tmp_path: Path, settings_env: Callable[..., None]) -> Iterator[Path]:
        settings_env(DOCS_PATH=str(tmp_path / "docs"), GRAPH_DOCS_PATH=str(tmp_path / "graph-docs"))
        yield tmp_path

    def test_each_corpus_keeps_bundles_in_its_own_reserved_folder(self, roots: Path) -> None:
        assert corpus_root("rag") == roots / "docs"
        assert exports_dir("rag") == roots / "docs" / "varagity-exports"
        assert exports_dir("graph") == roots / "graph-docs" / "varagity-exports"

    def test_display_paths_read_from_the_checkout_root(self, roots: Path) -> None:
        assert display_path("rag") == "docs/varagity-exports/"
        assert display_path("rag", "b.zip") == "docs/varagity-exports/b.zip"
        assert display_path("graph", "g.zip") == "graph-docs/varagity-exports/g.zip"
