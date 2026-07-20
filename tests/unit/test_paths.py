"""Unit tests for the shared path-containment helper (spec_v3 §5.2).

:func:`varagity.paths.resolve_contained` is the single backstop the upload,
corpus-list, delete, and preview sites share, so it carries dedicated tests
for each branch of "resolve a path and confirm it stays under the root".
"""

from pathlib import Path

import pytest

from varagity.paths import resolve_contained


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
