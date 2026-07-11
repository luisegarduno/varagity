"""Retriever protocol and registry (the spec §5.1 registry pattern).

Each retrieval method module defines one implementation decorated with
``@register("name")``; callers resolve the configured method with
``get_retriever(settings.RETRIEVAL_METHOD)``. v1 registers ``semantic``,
``bm25``, and ``hybrid``; adding a method later (e.g. a reranking retriever,
post-v1) means one new file plus its import line — no caller edits.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from varagity.stores.records import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """Interface every retrieval method implements.

    ``runtime_checkable`` because the protocol appears in Prefect flow
    signatures (``varagity.pipeline.query_flow``): Prefect builds a pydantic
    parameter schema from the annotations at decoration time, which requires
    types usable with ``isinstance``.
    """

    def encode_query(self, query: str, verbose: int | None = None) -> list[float] | None:
        """Encode a query into the vector its ``retrieve`` would use.

        Split out of ``retrieve`` so callers that track pipeline stages
        (``varagity.pipeline.query_flow``, spec §10.1 step 2) can run query
        embedding as its own stage and pass the result back via
        ``query_vector``.

        Args:
            query: The user's query, unformatted (each method owns its own
                query encoding — e.g. e5 query mode for ``semantic``).
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The query embedding, or ``None`` for methods that never encode
            queries (``bm25``).
        """
        ...

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks for a query.

        Args:
            query: The user's query, unformatted (each method owns its own
                query encoding — e.g. e5 query mode for ``semantic``).
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.
            query_vector: Pre-computed :meth:`encode_query` output; when
                omitted, methods that need a vector encode the query
                themselves. Methods that don't (``bm25``) ignore it.

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
            the available ones).
    """
    if name not in RETRIEVER_REGISTRY:
        raise KeyError(f"Unknown retrieval method {name!r}. Available: {list(RETRIEVER_REGISTRY)}")
    return RETRIEVER_REGISTRY[name]
