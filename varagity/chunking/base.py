"""Chunking-strategy protocol and registry (the spec §5.1 sketch, typed).

Each strategy module defines one implementation decorated with
``@register("name")``; callers resolve the configured strategy with
``get_chunker(settings.CHUNKING_STRATEGY)``.
"""

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.documents import Document


class ChunkingStrategy(Protocol):
    """Interface every chunking strategy implements."""

    def split(
        self, text: str, *, source_meta: dict[str, Any], verbose: int | None = None
    ) -> list[Document]:
        """Split a document's text into chunks.

        Args:
            text: The full document text.
            source_meta: Provenance seeded into every chunk's metadata
                (``source``, ``file_name``, ``file_type``, ``page``).
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The chunks, each with metadata seeded plus its ``chunk_index``.
        """
        ...


CHUNKER_REGISTRY: dict[str, ChunkingStrategy] = {}


def register[T: type[Any]](name: str) -> Callable[[T], T]:
    """Class decorator registering a chunking strategy instance under ``name``.

    Args:
        name: Registry key (the ``CHUNKING_STRATEGY`` env value).

    Returns:
        The decorator, which instantiates and registers the class unchanged.
    """

    def deco(cls: T) -> T:
        CHUNKER_REGISTRY[name] = cls()
        return cls

    return deco


def get_chunker(name: str) -> ChunkingStrategy:
    """Look up a registered chunking strategy by name.

    Args:
        name: Registry key (e.g. ``"recursive_character"``).

    Returns:
        The registered strategy instance.

    Raises:
        KeyError: If no strategy is registered under ``name`` (message lists
            the available ones).
    """
    if name not in CHUNKER_REGISTRY:
        raise KeyError(f"Unknown chunking strategy {name!r}. Available: {list(CHUNKER_REGISTRY)}")
    return CHUNKER_REGISTRY[name]
