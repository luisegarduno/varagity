"""Sparse (BM25) retrieval over Elasticsearch (spec §10.1 step 3, §11.3).

A thin wrapper over :class:`~varagity.stores.bm25_store.ElasticsearchBM25`:
search the contextual BM25 index, then hydrate the full chunk records from
pgvector by the ``(doc_id, original_index)`` identity — Elasticsearch only
stores the identity and text fields, and grounded answers need the full
metadata (``source`` for citations, ``context`` for the prompt).
"""

import logging

from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_retrieve
from varagity.retrieval.base import register
from varagity.stores.bm25_store import BM25Hit, ElasticsearchBM25
from varagity.stores.records import RetrievedChunk
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)


def hydrate(
    scored_keys: list[tuple[tuple[str, int], float]],
    store: ContextualVectorDB,
) -> list[RetrievedChunk]:
    """Hydrate identity tuples into full chunk records (spec §11.4).

    Shared by the bm25 and hybrid retrievers.

    Args:
        scored_keys: ``((doc_id, original_index), score)`` pairs, best first;
            the score becomes the returned chunk's score.
        store: The vector store holding the full rows.

    Returns:
        The hydrated chunks in the given order. A key missing from pgvector
        (the stores disagree — e.g. a partially failed ingest) is dropped
        with a warning rather than surfacing an uncitable result.
    """
    rows = store.fetch_by_identity([key for key, _ in scored_keys])
    chunks: list[RetrievedChunk] = []
    for key, score in scored_keys:
        row = rows.get(key)
        if row is None:
            logger.warning(
                "chunk %r is indexed in Elasticsearch but missing from pgvector — "
                "dropping it (re-run `ingest` to reconcile the stores)",
                key,
            )
            continue
        chunks.append(row.model_copy(update={"score": score}))
    return chunks


@register("bm25")
class BM25Retriever:
    """Retriever backed by the contextual BM25 index.

    The registry instantiates it without arguments (no I/O at import time);
    dependencies then resolve from settings per call — fresh store
    connections per query keep a long-lived chat session robust. Tests and
    the eval harness inject their own stores instead.
    """

    def __init__(
        self,
        *,
        bm25: ElasticsearchBM25 | None = None,
        store: ContextualVectorDB | None = None,
    ) -> None:
        """Create the retriever.

        Args:
            bm25: BM25 store to search; opened from settings per call when
                omitted.
            store: Vector store for hydration; opened from settings per call
                when omitted.
        """
        self._bm25 = bm25
        self._store = store

    def encode_query(self, query: str, verbose: int | None = None) -> None:
        """Return ``None``: BM25 searches raw text and never encodes queries.

        Args:
            query: The user's query (unused).
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            Always ``None`` (the ``query_vector`` of a bm25-only pipeline
            state stays empty, spec §10.1).

        Raises:
            ValueError: If ``verbose`` is invalid.
        """
        check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        return None

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks by BM25 relevance.

        Args:
            query: The user's query, passed to Elasticsearch unformatted.
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.
            query_vector: Accepted for protocol compatibility; unused (BM25
                is purely lexical).

        Returns:
            The top-k chunks, best first, with BM25 scores and full hydrated
            metadata.

        Raises:
            ValueError: If ``verbose`` is invalid.
            elastic_transport.ConnectionError: If Elasticsearch is still
                unreachable after retries.
            psycopg.OperationalError: If the vector store is unreachable.
        """
        verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        hits = self._search_bm25(query, k, verbose)
        scored_keys = [((hit.doc_id, hit.original_index), hit.score) for hit in hits]
        if self._store is not None:
            chunks = hydrate(scored_keys, self._store)
        else:
            with ContextualVectorDB() as store:
                chunks = hydrate(scored_keys, store)
        v_retrieve(chunks, verbose)
        return chunks

    def _search_bm25(self, query: str, k: int, verbose: int) -> list[BM25Hit]:
        """Run the BM25 search on the injected or a per-call store.

        Args:
            query: The user's query.
            k: Number of hits to request.
            verbose: Validated console verbosity.

        Returns:
            The BM25 hits, best first.
        """
        if self._bm25 is not None:
            return self._bm25.search(query, k, verbose=verbose)
        with ElasticsearchBM25() as bm25:
            return bm25.search(query, k, verbose=verbose)
