"""Filesystem path containment for the corpus routes (spec_v3 §5.2).

A stored ``source`` path — or an upload's declared target — is *data*, not
authority: before the API writes a file, lists it relative to the corpus,
unlinks it, or serves a preview of it, the path is re-resolved and confirmed
to still land inside ``DOCS_PATH``. :func:`resolve_contained` is the single
implementation the upload, corpus-list, delete, and preview sites share.
"""

from pathlib import Path


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
