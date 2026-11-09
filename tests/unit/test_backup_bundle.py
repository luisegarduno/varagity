"""Unit tests for the backup bundle container (ADR-019).

Bundles are real zips in ``tmp_path`` — the ``test_graph_manifest.py``
pattern — so these exercise the actual write path: members streamed and
hashed, the manifest sealed last, the ``.partial`` → ``os.replace`` publish,
and the one-member manifest read the listing depends on.
"""

import hashlib
import json
import os
import stat
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
    BundleDamaged,
    BundleIncompatible,
    BundleInvalid,
    BundleManifest,
    BundleVerdict,
    bundle_name,
    corpus_root,
    display_path,
    exports_dir,
    inspect_bundle,
    is_bundle_name,
    is_graph_file_name,
    list_bundles,
    open_bundle,
    read_manifest,
    sweep_partials,
    verify_bundle,
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
        assert [listing.name for listing in list_bundles(exports, "rag")] == [
            "varagity-rag-20261108-1312.zip"
        ]

    def test_an_in_flight_bundle_is_invisible_until_published(self, exports: Path) -> None:
        with write_bundle(exports, "rag", created_at=WHEN) as writer:
            with writer.member("documents.jsonl") as sink:
                sink.write(b"{}\n")
            assert list_bundles(exports, "rag") == []
            writer.seal(**seal_defaults())
        assert [listing.name for listing in list_bundles(exports, "rag")] == [writer.result.name]

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
        with pytest.raises(BundleInvalid, match="fake.zip: Not a readable zip"):
            read_manifest(path)

    def test_a_zip_without_a_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("readme.txt", "hello")
        with pytest.raises(BundleInvalid, match="plain.zip: No bundle.json"):
            read_manifest(path)

    def test_a_manifest_that_is_not_a_varagity_one(self, tmp_path: Path) -> None:
        path = tmp_path / "other.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(MANIFEST_NAME, '{"format": "something-else", "version": 1}')
        with pytest.raises(BundleInvalid, match="other.zip: .* not a Varagity manifest"):
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
        assert list_bundles(exports, "rag") == []

    def test_newest_first_with_manifests(self, exports: Path) -> None:
        older = write_sample(exports, when=datetime(2026, 10, 19, 9, 4, tzinfo=UTC))
        newer = write_sample(exports)
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        listings = list_bundles(exports, "rag")
        assert [listing.name for listing in listings] == [newer.name, older.name]
        assert listings[0].size_bytes == newer.stat().st_size
        assert listings[0].modified_at == datetime.fromtimestamp(2_000_000, tz=UTC)
        assert isinstance(listings[0].manifest, BundleManifest)

    def test_a_foreign_zip_is_listed_without_a_manifest(self, exports: Path) -> None:
        exports.mkdir(parents=True)
        with zipfile.ZipFile(exports / "photos.zip", "w") as archive:
            archive.writestr("a.jpg", b"...")
        (exports / "notes.txt").write_text("not a zip")
        listings = list_bundles(exports, "rag")
        assert [(listing.name, listing.manifest) for listing in listings] == [("photos.zip", None)]

    def test_a_symlink_leading_out_of_the_folder_is_not_listed(
        self, exports: Path, tmp_path: Path
    ) -> None:
        outside = write_sample(tmp_path / "elsewhere")
        exports.mkdir(parents=True)
        (exports / "link.zip").symlink_to(outside)
        assert list_bundles(exports, "rag") == []


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


# ── the restore verdicts (phase 2) ────────────────────────────────────────

DIM = 4


@pytest.fixture
def stack(settings_env: Callable[..., None]) -> None:
    """This stack's embedding space — the one the sample bundles were made in."""
    settings_env(EMBEDDING_MODEL="test-model", EMBEDDING_DIM=DIM)


def rag_members(chunks: int = 2) -> dict[str, bytes]:
    """A RAG bundle's three data members with a consistent layout."""
    return {
        "documents.jsonl": b'{"doc_id":"a"}\n',
        "chunks.jsonl": b"".join(b'{"chunk_id":"a::%d"}\n' % i for i in range(chunks)),
        "embeddings.f32": bytes(range(chunks * DIM * 4)),
    }


def craft(
    path: Path,
    *,
    members: dict[str, bytes] | None = None,
    manifest: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    extra: tuple[tuple[zipfile.ZipInfo | str, bytes], ...] = (),
) -> Path:
    """Hand-build a bundle zip: data members, any extra entries, then bundle.json.

    The manifest records every member's real size and sha256 unless
    ``records`` (or a ``manifest`` override of ``members``) says otherwise.
    """
    contents = rag_members() if members is None else members
    document: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "corpus": "rag",
        "created_at": WHEN.isoformat(),
        "embedding_model": "test-model",
        "embedding_dim": DIM,
        "stale": False,
        "counts": {"documents": 1, "chunks": 2},
        "ingest_settings": {"CHUNK_SIZE": "400"},
        "members": records
        if records is not None
        else [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in contents.items()
        ],
    }
    document.update(manifest or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        for entry, data in extra:
            archive.writestr(entry, data)
        archive.writestr(MANIFEST_NAME, json.dumps(document))
    return path


def verdict_of(path: Path, corpus: str = "rag") -> BundleVerdict:
    return inspect_bundle(path, corpus)  # type: ignore[arg-type]


class TestVerdicts:
    def test_a_sound_bundle_is_restorable(self, exports: Path, stack: None) -> None:
        verdict = verdict_of(write_sample(exports))
        assert verdict.restorable is True
        assert verdict.reason is None
        assert verdict.manifest is not None and verdict.manifest.counts["chunks"] == 2
        assert verdict.manifest_or_raise() is verdict.manifest

    def test_another_format_version_is_incompatible_not_unreadable(
        self, tmp_path: Path, stack: None
    ) -> None:
        # A later layout may not validate as this one: the version decides first.
        path = craft(tmp_path / "v2.zip", manifest={"version": 2, "corpus": "everything"})
        verdict = verdict_of(path)
        assert isinstance(verdict.problem, BundleIncompatible)
        assert verdict.reason == "Bundle format version 2 — this stack reads version 1."
        assert verdict.manifest is None
        with pytest.raises(BundleIncompatible):
            verdict.manifest_or_raise()

    def test_a_bundle_of_the_other_corpus_is_incompatible(
        self, tmp_path: Path, stack: None
    ) -> None:
        verdict = verdict_of(craft(tmp_path / "g.zip", manifest={"corpus": "graph"}))
        assert isinstance(verdict.problem, BundleIncompatible)
        assert verdict.reason == "A message graph backup — it cannot restore the document corpus."
        assert verdict.manifest is not None  # still described: counts, dates, …

    def test_another_embedding_model_is_incompatible(self, tmp_path: Path, stack: None) -> None:
        verdict = verdict_of(craft(tmp_path / "m.zip", manifest={"embedding_model": "BAAI/bge-m3"}))
        assert isinstance(verdict.problem, BundleIncompatible)
        assert verdict.reason == ("Embedded with BAAI/bge-m3 — this stack embeds with test-model.")

    def test_another_embedding_dimension_is_incompatible(self, tmp_path: Path, stack: None) -> None:
        # A consistent 8-dimensional bundle: the layout holds, the stack's dim doesn't.
        members = {**rag_members(), "embeddings.f32": bytes(2 * 8 * 4)}
        verdict = verdict_of(
            craft(tmp_path / "d.zip", members=members, manifest={"embedding_dim": 8})
        )
        assert isinstance(verdict.problem, BundleIncompatible)
        assert verdict.reason == ("8-dimensional embeddings — this stack embeds into 4 dimensions.")

    @pytest.mark.parametrize(
        ("entry", "why"),
        [
            ("../escape.txt", "it climbs out of the bundle"),
            ("/etc/cron.d/evil", "an absolute path"),
            ("C:/windows/evil", "an absolute path"),
            ("a\\b.txt", "a backslash in its path"),
            ("a//b.txt", "an empty or '.' path segment"),
        ],
    )
    def test_unsafe_member_paths_are_invalid(
        self, tmp_path: Path, stack: None, entry: str, why: str
    ) -> None:
        path = craft(tmp_path / "evil.zip", extra=((entry, b"payload"),))
        verdict = verdict_of(path)
        assert type(verdict.problem) is BundleInvalid
        assert verdict.reason == f"Unsafe member {entry!r}: {why}."

    def test_a_symlink_member_is_invalid(self, tmp_path: Path, stack: None) -> None:
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        verdict = verdict_of(craft(tmp_path / "link.zip", extra=((link, b"/etc/passwd"),)))
        assert verdict.reason == "Unsafe member 'link': a symlink."

    def test_a_member_outside_the_allowlist_is_invalid(self, tmp_path: Path, stack: None) -> None:
        members = {**rag_members(), "notes.txt": b"hi"}
        verdict = verdict_of(craft(tmp_path / "extra.zip", members=members))
        assert verdict.reason == "Unexpected member(s): notes.txt."

    def test_a_member_the_manifest_records_but_the_zip_lacks_is_invalid(
        self, tmp_path: Path, stack: None
    ) -> None:
        members = rag_members()
        records = [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
        ]
        del members["chunks.jsonl"]
        verdict = verdict_of(craft(tmp_path / "short.zip", members=members, records=records))
        assert verdict.reason == "Missing member(s): chunks.jsonl."

    def test_an_allowed_member_the_manifest_does_not_record_is_invalid(
        self, tmp_path: Path, stack: None
    ) -> None:
        members = rag_members()
        records = [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
            if name != "embeddings.f32"
        ]
        del members["embeddings.f32"]
        verdict = verdict_of(craft(tmp_path / "unrecorded.zip", members=members, records=records))
        assert verdict.reason == "bundle.json does not record embeddings.f32."

    def test_a_duplicated_entry_is_invalid(self, tmp_path: Path, stack: None) -> None:
        with pytest.warns(UserWarning, match="Duplicate name"):
            path = craft(tmp_path / "dup.zip", extra=(("chunks.jsonl", b"again"),))
        assert verdict_of(path).reason == "The zip holds 'chunks.jsonl' more than once."

    def test_a_member_the_manifest_lists_twice_is_invalid(
        self, tmp_path: Path, stack: None
    ) -> None:
        members = rag_members()
        records = [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
        ]
        path = craft(tmp_path / "twice.zip", members=members, records=[*records, records[0]])
        assert verdict_of(path).reason == "bundle.json lists 'documents.jsonl' more than once."

    def test_a_recorded_size_that_differs_from_the_zips_is_invalid(
        self, tmp_path: Path, stack: None
    ) -> None:
        members = rag_members()
        records = [
            {
                "name": name,
                "size_bytes": len(data) + (1 if name == "chunks.jsonl" else 0),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in members.items()
        ]
        verdict = verdict_of(craft(tmp_path / "size.zip", members=members, records=records))
        assert type(verdict.problem) is BundleInvalid
        assert verdict.reason is not None
        size = len(members["chunks.jsonl"])
        assert verdict.reason == (
            f"chunks.jsonl is {size} bytes in the zip but {size + 1} in bundle.json — the "
            "bundle is damaged."
        )

    def test_a_vector_matrix_that_disagrees_with_the_chunk_count_is_invalid(
        self, tmp_path: Path, stack: None
    ) -> None:
        verdict = verdict_of(
            craft(tmp_path / "rows.zip", manifest={"counts": {"documents": 1, "chunks": 3}})
        )
        assert verdict.reason == (
            "embeddings.f32 is 32 bytes, but 3 chunk(s) of 4 float32s take 48 — the bundle is "
            "damaged."
        )

    def test_missing_counts_are_invalid(self, tmp_path: Path, stack: None) -> None:
        verdict = verdict_of(craft(tmp_path / "counts.zip", manifest={"counts": {}}))
        assert verdict.reason == "bundle.json does not record its document and chunk counts."

    def test_unreadable_files_get_a_reason_not_an_exception(self, tmp_path: Path) -> None:
        junk = tmp_path / "junk.zip"
        junk.write_bytes(b"not a zip at all")
        verdict = verdict_of(junk)
        assert verdict.manifest is None and verdict.reason is not None
        assert verdict.reason.startswith("Not a readable zip file")
        plain = tmp_path / "plain.zip"
        with zipfile.ZipFile(plain, "w") as archive:
            archive.writestr("photo.jpg", b"...")
        assert verdict_of(plain).reason == "No bundle.json — not a Varagity backup."
        broken = tmp_path / "broken.zip"
        with zipfile.ZipFile(broken, "w") as archive:
            archive.writestr(MANIFEST_NAME, "{not json")
        assert verdict_of(broken).reason == "Its bundle.json is not a Varagity manifest (not JSON)."

    def test_inspection_never_opens_a_data_member(
        self, exports: Path, stack: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_sample(exports)
        opened: list[str] = []
        real_open = zipfile.ZipFile.open

        def spy(self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any) -> Any:
            opened.append(name if isinstance(name, str) else name.filename)
            return real_open(self, name, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", spy)
        assert verdict_of(path).restorable
        assert opened == [MANIFEST_NAME]

    def test_listings_carry_each_bundles_verdict(self, exports: Path, stack: None) -> None:
        good = write_sample(exports)
        bad = craft(exports / "other-model.zip", manifest={"embedding_model": "BAAI/bge-m3"})
        os.utime(good, (2_000_000, 2_000_000))
        os.utime(bad, (1_000_000, 1_000_000))
        listings = list_bundles(exports, "rag")
        assert [(listing.name, listing.verdict.restorable) for listing in listings] == [
            (good.name, True),
            (bad.name, False),
        ]
        assert listings[0].mtime_ns == 2_000_000 * 10**9
        assert listings[1].manifest is not None  # an incompatible bundle is still described


class TestVerification:
    def test_a_sound_bundle_verifies_and_returns_its_manifest(
        self, exports: Path, stack: None
    ) -> None:
        path = write_sample(exports)
        assert verify_bundle(path, "rag") == read_manifest(path)

    def test_verification_raises_the_cheap_verdicts_problem(
        self, tmp_path: Path, stack: None
    ) -> None:
        path = craft(tmp_path / "m.zip", manifest={"embedding_model": "BAAI/bge-m3"})
        with pytest.raises(BundleIncompatible, match="Embedded with BAAI/bge-m3"):
            verify_bundle(path, "rag")

    def test_a_member_whose_bytes_changed_is_damaged(self, tmp_path: Path, stack: None) -> None:
        """★ Same sizes, re-zipped CRCs: only the sha256 can tell."""
        members = rag_members()
        records = [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
        ]
        tampered = {**members, "chunks.jsonl": members["chunks.jsonl"].replace(b"a::1", b"a::7")}
        path = craft(tmp_path / "tampered.zip", members=tampered, records=records)
        assert verdict_of(path).restorable  # the cheap verdict can't see it…
        with pytest.raises(BundleDamaged, match="chunks.jsonl does not match its recorded sha256"):
            verify_bundle(path, "rag")  # …the hash does

    def test_a_crc_failure_is_damage_too(self, tmp_path: Path, stack: None) -> None:
        """Bytes flipped inside the zip file itself (a stored member, so the CRC trips)."""
        path = write_sample(tmp_path / "exports")
        raw = bytearray(path.read_bytes())
        marker = bytes(range(32))  # embeddings.f32's stored payload in write_sample
        offset = raw.index(marker)
        raw[offset] ^= 0xFF
        path.write_bytes(bytes(raw))
        with pytest.raises(BundleDamaged, match="embeddings.f32 could not be read"):
            verify_bundle(path, "rag")

    def test_member_streams_read_lines_and_check_on_exit(self, exports: Path, stack: None) -> None:
        path = write_sample(exports)
        manifest = read_manifest(path)
        with open_bundle(path, manifest) as source:
            with source.member("chunks.jsonl") as stream:
                assert [line for line in stream] == [
                    b'{"chunk_id":"a::0"}\n',
                    b'{"chunk_id":"a::1"}\n',
                ]
            with source.member("embeddings.f32") as stream:
                assert stream.read(4) == bytes(range(4))  # the rest is drained by the check
            with pytest.raises(BundleInvalid, match="Missing member"):
                source.member("nope.jsonl").__enter__()

    def test_a_stream_abandoned_by_an_error_is_not_checked(
        self, exports: Path, stack: None
    ) -> None:
        path = write_sample(exports)
        with (
            open_bundle(path, read_manifest(path)) as source,
            pytest.raises(RuntimeError, match="the consumer gave up"),
            source.member("chunks.jsonl") as stream,
        ):
            stream.readline()
            raise RuntimeError("the consumer gave up")  # not a BundleDamaged

    def test_an_unreadable_file_cannot_be_opened(self, tmp_path: Path) -> None:
        junk = tmp_path / "junk.zip"
        junk.write_bytes(b"nope")
        manifest = BundleManifest.model_validate(
            {
                "format": BUNDLE_FORMAT,
                "version": BUNDLE_VERSION,
                "corpus": "rag",
                "created_at": WHEN.isoformat(),
                "embedding_model": "m",
                "embedding_dim": DIM,
                "stale": False,
                "counts": {},
                "members": [],
            }
        )
        with pytest.raises(BundleInvalid, match="Not a readable zip"), open_bundle(junk, manifest):
            pass


# ── the graph allowlist (phase 3) ─────────────────────────────────────────

ENGINE = {"name": "lightrag", "package": "lightrag-hku", "version": "1.5.4"}


def graph_members() -> dict[str, bytes]:
    """A few of a LightRAG workdir's files, under the engine's folder."""
    return {
        "lightrag/graph_chunk_entity_relation.graphml": b"<graphml/>",
        "lightrag/kv_store_doc_status.json": b'{"t1::2026-01-01": {"status": "processed"}}',
        "lightrag/vdb_entities.json": b'{"embedding_dim": 1024, "data": [], "matrix": ""}',
        "lightrag/varagity_manifest.json": b'{"version": 1, "docs": {}}',
    }


def graph_craft(path: Path, **kwargs: Any) -> Path:
    """Hand-build a graph bundle (the RAG ``craft`` with a graph manifest)."""
    members = kwargs.pop("members", None)
    manifest = {
        "corpus": "graph",
        "engine": ENGINE,
        "counts": {"thread_days": 1, "messages": 2},
        "ingest_settings": None,
        **kwargs.pop("manifest", {}),
    }
    return craft(
        path, members=graph_members() if members is None else members, manifest=manifest, **kwargs
    )


class TestGraphVerdicts:
    def test_a_sound_graph_bundle_is_restorable_and_verifies(
        self, tmp_path: Path, stack: None
    ) -> None:
        path = graph_craft(tmp_path / "g.zip")
        verdict = verdict_of(path, "graph")
        assert verdict.restorable, verdict.reason
        assert verdict.manifest is not None
        assert verdict.manifest.engine is not None
        assert verdict.manifest.engine.version == "1.5.4"
        assert verify_bundle(path, "graph").counts == {"thread_days": 1, "messages": 2}

    def test_no_member_is_individually_required(self, tmp_path: Path, stack: None) -> None:
        """A workdir holds only the namespaces its engine has written so far."""
        members = {"lightrag/kv_store_doc_status.json": b"{}"}
        assert verdict_of(graph_craft(tmp_path / "g.zip", members=members), "graph").restorable

    def test_a_graph_bundle_must_record_its_engine(self, tmp_path: Path, stack: None) -> None:
        path = graph_craft(tmp_path / "g.zip", manifest={"engine": None})
        verdict = verdict_of(path, "graph")
        assert type(verdict.problem) is BundleInvalid
        assert verdict.reason == "bundle.json does not record the graph engine that wrote it."

    @pytest.mark.parametrize("name", ["..", "light rag", ".lightrag", "a/b", ""])
    def test_an_engine_name_that_cannot_be_a_folder_is_invalid(
        self, tmp_path: Path, stack: None, name: str
    ) -> None:
        path = graph_craft(tmp_path / "g.zip", manifest={"engine": {**ENGINE, "name": name}})
        assert verdict_of(path, "graph").reason == (
            f"bundle.json records the graph engine {name!r}, which is not a folder name."
        )

    @pytest.mark.parametrize(
        "entry",
        [
            "kv_store_full_docs.json",  # outside the engine folder
            "other/kv_store_full_docs.json",  # another folder
            "lightrag/nested/x.json",  # flat files only
            "lightrag/.hidden.json",  # never hidden
            "lightrag/with space.json",  # a name an engine never writes
            "documents.jsonl",  # a RAG member
        ],
    )
    def test_members_must_be_flat_safe_files_in_the_engine_folder(
        self, tmp_path: Path, stack: None, entry: str
    ) -> None:
        path = graph_craft(tmp_path / "g.zip", members={**graph_members(), entry: b"x"})
        verdict = verdict_of(path, "graph")
        assert type(verdict.problem) is BundleInvalid
        assert verdict.reason == f"Unexpected member(s): {entry}."

    def test_a_file_the_manifest_never_hashed_is_invalid(self, tmp_path: Path, stack: None) -> None:
        """An unrecorded member could never be verified, so it could never restore."""
        members = graph_members()
        records = [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
            if name != "lightrag/vdb_entities.json"
        ]
        path = graph_craft(tmp_path / "g.zip", members=members, records=records)
        assert verdict_of(path, "graph").reason == (
            "bundle.json does not record lightrag/vdb_entities.json."
        )

    def test_a_hand_rezipped_bundles_folder_entry_is_tolerated(
        self, tmp_path: Path, stack: None
    ) -> None:
        """``zip -r`` adds a ``lightrag/`` directory entry; it holds nothing to verify."""
        path = graph_craft(tmp_path / "g.zip", extra=((zipfile.ZipInfo("lightrag/"), b""),))
        assert verdict_of(path, "graph").restorable

    def test_only_the_engine_folder_entry_is_tolerated(self, tmp_path: Path, stack: None) -> None:
        path = graph_craft(tmp_path / "g.zip", extra=((zipfile.ZipInfo("other/"), b""),))
        assert verdict_of(path, "graph").reason == "Unexpected member(s): other/."

    def test_a_recorded_size_that_differs_from_the_zips_is_damage(
        self, tmp_path: Path, stack: None
    ) -> None:
        members = graph_members()
        records = [
            {
                "name": name,
                "size_bytes": len(data) + (1 if name.endswith(".graphml") else 0),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in members.items()
        ]
        path = graph_craft(tmp_path / "g.zip", members=members, records=records)
        reason = verdict_of(path, "graph").reason
        assert reason is not None and reason.endswith("— the bundle is damaged.")

    def test_a_graph_bundle_from_another_embedding_model_is_incompatible(
        self, tmp_path: Path, stack: None
    ) -> None:
        path = graph_craft(tmp_path / "g.zip", manifest={"embedding_model": "BAAI/bge-m3"})
        verdict = verdict_of(path, "graph")
        assert isinstance(verdict.problem, BundleIncompatible)
        assert verdict.reason == "Embedded with BAAI/bge-m3 — this stack embeds with test-model."

    def test_the_corpora_cannot_stand_in_for_each_other(self, tmp_path: Path, stack: None) -> None:
        graph = graph_craft(tmp_path / "g.zip")
        assert verdict_of(graph, "rag").reason == (
            "A message graph backup — it cannot restore the document corpus."
        )
        rag = craft(tmp_path / "r.zip")
        assert verdict_of(rag, "graph").reason == (
            "A document corpus backup — it cannot restore the message graph."
        )

    def test_the_rag_allowlist_is_untouched_by_the_graph_rule(
        self, tmp_path: Path, stack: None
    ) -> None:
        path = craft(tmp_path / "r.zip", extra=(("lightrag/vdb_entities.json", b"{}"),))
        assert verdict_of(path, "rag").reason == "Unexpected member(s): lightrag/vdb_entities.json."

    def test_a_tampered_graph_member_is_damaged(self, tmp_path: Path, stack: None) -> None:
        members = graph_members()
        records = [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
        ]
        tampered = {**members, "lightrag/graph_chunk_entity_relation.graphml": b"<GRAPHML/>"}
        path = graph_craft(tmp_path / "g.zip", members=tampered, records=records)
        assert verdict_of(path, "graph").restorable  # same sizes: only the hash can tell
        with pytest.raises(BundleDamaged, match="graph_chunk_entity_relation.graphml"):
            verify_bundle(path, "graph")

    def test_the_engine_format_round_trips_through_the_writer(self, exports: Path) -> None:
        from varagity.graph.records import EngineFormat

        engine = EngineFormat(**ENGINE)
        with write_bundle(exports.parent / "graph-exports", "graph", created_at=WHEN) as writer:
            with writer.member("lightrag/kv_store_doc_status.json") as sink:
                sink.write(b"{}")
            writer.seal(
                **seal_defaults(counts={"thread_days": 1}, ingest_settings=None, engine=engine)
            )
        manifest = read_manifest(writer.result.path)
        assert manifest.engine == engine
        assert writer.result.name == "varagity-graph-20261108-1312.zip"

    def test_a_rag_manifest_records_no_engine(self, exports: Path) -> None:
        assert read_manifest(write_sample(exports)).engine is None


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("kv_store_llm_response_cache.json", True),
        ("graph_chunk_entity_relation.graphml", True),
        ("varagity_graph_summary.json", True),
        ("lightrag", True),
        ("v1.5-x_y", True),
        (".hidden", False),
        ("..", False),
        (".", False),
        ("", False),
        ("a/b.json", False),
        ("a\\b.json", False),
        ("with space.json", False),
        ("ünïcode.json", False),
        ("x" * 255, True),
        ("x" * 256, False),
    ],
)
def test_graph_file_names(name: str, ok: bool) -> None:
    assert is_graph_file_name(name) is ok
