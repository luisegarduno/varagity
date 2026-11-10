"""Unit tests for the shared path-containment helper (spec_v3 §5.2).

:func:`varagity.paths.resolve_contained` is the single backstop the upload,
corpus-list, delete, and preview sites share, so it carries dedicated tests
for each branch of "resolve a path and confirm it stays under the root".
"""

import os
from pathlib import Path, PurePath, PurePosixPath

import pytest

from varagity.paths import (
    EXPORTS_DIRNAME,
    fsync_directory,
    is_exports_path,
    relative_to_root,
    resolve_contained,
)


def test_contained_file_returns_the_resolved_path(tmp_path: Path) -> None:
    """A file beneath the root resolves to its real path."""
    root = tmp_path / "corpus"
    (root / "q3").mkdir(parents=True)
    target = root / "q3" / "notes.md"
    target.write_text("hi")
    assert resolve_contained(target, root.resolve()) == target.resolve()


def test_the_root_itself_is_contained(tmp_path: Path) -> None:
    """The boundary is inclusive: the root resolves to itself."""
    root = tmp_path / "corpus"
    root.mkdir()
    assert resolve_contained(root, root.resolve()) == root.resolve()


def test_path_outside_root_returns_none(tmp_path: Path) -> None:
    """A path outside the root is rejected (it need not even exist)."""
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "secret.md"
    assert resolve_contained(outside, root.resolve()) is None


def test_symlink_escaping_the_root_returns_none(tmp_path: Path) -> None:
    """A link inside the root pointing out of it resolves outside → None."""
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = root / "escape"
    escape.symlink_to(outside)
    assert resolve_contained(escape / "loot.md", root.resolve()) is None


def test_unresolvable_path_oserror_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve() raising OSError counts as not-contained, never propagates."""
    root = tmp_path / "corpus"
    root.mkdir()

    def _raise(*args: object, **kwargs: object) -> Path:
        raise OSError("simulated dangling symlink in the prefix")

    monkeypatch.setattr(Path, "resolve", _raise)
    assert resolve_contained(root / "x.md", root) is None


class TestRelativeToRoot:
    def test_a_contained_source_is_relative_posix(self, tmp_path: Path) -> None:
        root = (tmp_path / "corpus").resolve()
        (root / "q3").mkdir(parents=True)
        assert relative_to_root(str(root / "q3" / "notes.md"), root) == "q3/notes.md"

    def test_a_source_need_not_exist(self, tmp_path: Path) -> None:
        """A backup records relative paths for documents whose files are gone."""
        root = (tmp_path / "corpus").resolve()
        root.mkdir()
        assert relative_to_root(root / "deleted.pdf", root) == "deleted.pdf"

    def test_an_outside_source_has_none(self, tmp_path: Path) -> None:
        root = (tmp_path / "corpus").resolve()
        root.mkdir()
        assert relative_to_root("/app/docs/a.md", root) is None


class TestIsExportsPath:
    @pytest.mark.parametrize(
        "relative",
        [
            "varagity-exports",
            "varagity-exports/varagity-rag-20261108-1312.zip",
            "varagity-exports/deep/x.md",
            "Varagity-Exports/x.md",  # case-insensitive filesystems
            PurePosixPath("varagity-exports/x.md"),
            Path("varagity-exports"),
        ],
    )
    def test_the_reserved_folder_and_its_contents(self, relative: str | PurePath) -> None:
        assert is_exports_path(relative) is True

    @pytest.mark.parametrize(
        "relative",
        [
            "",
            "notes.md",
            "varagity-exports.zip",  # a file beside it, not the folder
            "q3/varagity-exports/x.md",  # nested: ordinary corpus
            "my-varagity-exports/x.md",
        ],
    )
    def test_everything_else(self, relative: str) -> None:
        assert is_exports_path(relative) is False


def test_the_reserved_folder_name() -> None:
    assert EXPORTS_DIRNAME == "varagity-exports"


class TestFsyncDirectory:
    def test_syncs_a_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        synced: list[int] = []
        real_fsync = os.fsync
        monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
        fsync_directory(tmp_path)
        assert len(synced) == 1

    def test_is_best_effort(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rename already happened: a directory that can't be synced is no failure."""
        fsync_directory(tmp_path / "missing")  # can't be opened

        def refuse(fd: int) -> None:
            raise OSError("EINVAL")

        monkeypatch.setattr(os, "fsync", refuse)
        fsync_directory(tmp_path)  # opened, refuses the sync
