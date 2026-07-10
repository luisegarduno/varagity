"""Parser protocol and registry (the spec §5.1 registry pattern).

Each parser module defines one implementation decorated with
``@register("name")``; callers resolve implementations with
:func:`get_parser` and never import concrete parsers directly.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RawDocument:
    """A parsed source document, before chunking.

    Attributes:
        text: The full extracted text (normalized by the parser).
        source_meta: Provenance seeded into every chunk's metadata:
            ``source`` (absolute path), ``file_name``, ``file_type``, and
            ``page`` (``None`` for non-paginated formats).
    """

    text: str
    source_meta: dict[str, Any]


class Parser(Protocol):
    """Interface every parser implements."""

    def extract(self, path: Path, verbose: int | None = None) -> RawDocument:
        """Extract text and provenance from a source file.

        Args:
            path: The file to parse.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The extracted document.
        """
        ...


PARSER_REGISTRY: dict[str, Parser] = {}


def register[T: type[Any]](name: str) -> Callable[[T], T]:
    """Class decorator registering a parser instance under ``name``.

    Args:
        name: Registry key (matches the discovery bucket, e.g. ``"text"``).

    Returns:
        The decorator, which instantiates and registers the class unchanged.
    """

    def deco(cls: T) -> T:
        PARSER_REGISTRY[name] = cls()
        return cls

    return deco


def get_parser(name: str) -> Parser:
    """Look up a registered parser by name.

    Args:
        name: Registry key (e.g. ``"text"``, ``"pdf"``).

    Returns:
        The registered parser instance.

    Raises:
        KeyError: If no parser is registered under ``name`` (message lists
            the available ones).
    """
    if name not in PARSER_REGISTRY:
        raise KeyError(f"Unknown parser {name!r}. Available: {list(PARSER_REGISTRY)}")
    return PARSER_REGISTRY[name]
