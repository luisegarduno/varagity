"""Filesystem path containment for the corpus routes (spec_v3 §5.2).

A stored ``source`` path — or an upload's declared target — is *data*, not
authority: before the API writes a file, lists it relative to the corpus,
unlinks it, or serves a preview of it, the path is re-resolved and confirmed
to still land inside ``DOCS_PATH``. :func:`resolve_contained` is the single
implementation the upload, corpus-list, delete, and preview sites share.

This module also names the one folder inside each corpus root that is *not*
corpus: :data:`EXPORTS_DIRNAME`, where backup bundles live (ADR-019).
Discovery and the upload routes consult :func:`is_exports_path` rather than
importing the backup package, so the rule "a bundle is never a source" has
one definition for every reader of a corpus folder to share.
"""

from pathlib import Path, PurePath

EXPORTS_DIRNAME = "varagity-exports"
"""The reserved folder each corpus root keeps its backup bundles in.

It sits inside the host bind mounts (``./docs``, ``./graph-docs``), so the
bundles survive ``docker compose down -v`` beside the sources they back up —
and nothing that reads a corpus may treat it as part of one.
"""


def resolve_contained(path: Path, root: Path) -> Path | None:
    """Resolve ``path`` and return it only when it stays under ``root``.

    The shared containment backstop (spec_v3 §5.2 rule 6): symlinks are
    followed, so a link inside the corpus that points out of it is caught,
    and an unresolvable path (a symlink loop, a vanished prefix) counts as
    *not* contained rather than raising.

    ``root`` is compared verbatim — each caller passes exactly the boundary
    it means (already ``resolve()``d, or freshly resolved at the call site),
    so this never second-guesses which form of the root to check against.

    Args:
        path: The candidate path (need not exist; resolved non-strictly).
        root: The containment boundary the resolved path is checked against.

    Returns:
        The resolved path when it is ``root`` or lives beneath it; ``None``
        when it escapes the boundary or cannot be resolved.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return None
    return resolved if resolved.is_relative_to(root) else None


def relative_to_root(path: str | Path, root: Path) -> str | None:
    """A path's POSIX form relative to ``root``, when it resolves inside it.

    The corpus table's grouping key (folder uploads keep their structure,
    and the GUI folds the flat document list back into those folders) and a
    backup's portable form of a document's ``source``: only the server knows
    the corpus root, so the relative path is computed here rather than
    string-guessed by a client.

    Args:
        path: The (usually absolute) path recorded at ingest time.
        root: The resolved corpus directory.

    Returns:
        The POSIX-separated relative path, or ``None`` when the path doesn't
        resolve inside ``root`` (ingested from elsewhere — such documents
        list flat, and a backup keeps their absolute path instead).
    """
    resolved = resolve_contained(Path(path), root)
    if resolved is None:
        return None
    return resolved.relative_to(root).as_posix()


def is_exports_path(relative: str | PurePath) -> bool:
    """Report whether a corpus-relative path is the reserved exports folder.

    True for the folder itself and for anything beneath it — only the
    *first* segment decides, so a ``varagity-exports`` folder nested deeper
    in the corpus is ordinary corpus. The name is matched case-insensitively:
    on a case-insensitive filesystem (Docker Desktop's macOS binds)
    ``Varagity-Exports/`` *is* the reserved folder, and a differently-cased
    twin must not smuggle an upload into it — the same posture the upload
    route takes toward Windows' reserved device names.

    Args:
        relative: A path relative to a corpus root — a discovered file's
            ``relative_to(root)``, an upload's sanitized relative path, or a
            flat upload's bare file name.

    Returns:
        ``True`` when the path's first segment is :data:`EXPORTS_DIRNAME`.
    """
    parts = PurePath(relative).parts
    return bool(parts) and parts[0].casefold() == EXPORTS_DIRNAME
