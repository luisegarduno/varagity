"""Retriever protocol and registry (the spec §5.1 registry pattern).

Each retrieval method module defines one implementation decorated with
``@register("name")``; callers resolve the configured method with
``get_retriever(settings.RETRIEVAL_METHOD)``. Phase 4 registers ``semantic``;
``bm25`` and ``hybrid`` land in Phase 6 as new files — no caller edits.
"""

from collections.abc import Callable
from typing import Any, Protocol

from varagity.stores.records import RetrievedChunk


class Retriever(Protocol):
    """Interface every retrieval method implements."""

    def retrieve(self, query: str, k: int, verbose: int | None = None) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks for a query.

        Args:
            query: The user's query, unformatted (each method owns its own
                query encoding — e.g. e5 query mode for ``semantic``).
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The top-k chunks, best first, each carrying its ``score`` and the
            full persisted metadata record.
        """
        ...


RETRIEVER_REGISTRY: dict[str, Retriever] = {}


def register[T: type[Any]](name: str) -> Callable[[T], T]:
    """Class decorator registering a retriever instance under ``name``.

    Args:
        name: Registry key (the ``RETRIEVAL_METHOD`` env value).

    Returns:
        The decorator, which instantiates and registers the class unchanged.
    """

    def deco(cls: T) -> T:
        RETRIEVER_REGISTRY[name] = cls()
        return cls

    return deco


def get_retriever(name: str) -> Retriever:
    """Look up a registered retrieval method by name.

    Args:
        name: Registry key (e.g. ``"semantic"``).

    Returns:
        The registered retriever instance.

    Raises:
        KeyError: If no retriever is registered under ``name`` (message lists
            the available ones — note ``bm25``/``hybrid`` pass config
            validation but are not registered until Phase 6).
    """
    if name not in RETRIEVER_REGISTRY:
        raise KeyError(f"Unknown retrieval method {name!r}. Available: {list(RETRIEVER_REGISTRY)}")
    return RETRIEVER_REGISTRY[name]
