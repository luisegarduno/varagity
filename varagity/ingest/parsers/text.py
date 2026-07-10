"""Plain-text parser for ``.txt`` and ``.md`` (spec §9.2).

Reads UTF-8, normalizes newlines, and repairs hyphen-broken words via
:func:`remove_hyphen_space` (carried over from the reference implementation).
"""

import re
from pathlib import Path

from varagity.ingest.parsers.base import RawDocument, register

# Line-broken hyphenation: "frame-\nwork" (possibly with trailing/leading
# whitespace around the newline) → "framework".
_HYPHEN_NEWLINE = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
# Mid-line split: "frame- work" → "framework".
_HYPHEN_SPACE = re.compile(r"(\w)-[ \t]+(\w)")


def remove_hyphen_space(text: str) -> str:
    r"""Rejoin words split by a hyphen at a line break or before a space.

    The reference implementation's fix for extraction artifacts like
    ``frame-\nwork`` and ``frame- work`` (both → ``framework``).

    Args:
        text: Text with possible hyphenation artifacts (newlines already
            normalized to ``\n``).

    Returns:
        The text with split words rejoined.
    """
    text = _HYPHEN_NEWLINE.sub(r"\1\2", text)
    return _HYPHEN_SPACE.sub(r"\1\2", text)


@register("text")
class TextParser:
    """Parser for the ``text_like`` bucket (``.txt`` / ``.md``)."""

    def extract(self, path: Path, verbose: int | None = None) -> RawDocument:
        """Read and normalize a text file.

        Args:
            path: The ``.txt`` / ``.md`` file to read (UTF-8).
            verbose: Console verbosity (0–2); accepted for interface
                uniformity — this parser renders nothing itself.

        Returns:
            The extracted document; ``page`` is ``None`` (not paginated).

        Raises:
            UnicodeDecodeError: If the file is not valid UTF-8.
            OSError: If the file cannot be read.
        """
        raw = path.read_text(encoding="utf-8")
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        text = remove_hyphen_space(text)
        return RawDocument(
            text=text,
            source_meta={
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "page": None,
            },
        )
