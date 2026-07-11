"""Hybrid retrieval — weighted reciprocal-rank fusion (spec §10.1, §11.4).

The cookbook's fusion, verbatim in structure: over-retrieve ``k * 10`` from
**each** store, combine by weighted reciprocal rank (``weight * 1/(rank+1)``
— only rank positions matter, raw scores from the two stores are never
compared), dedupe on the ``(doc_id, original_index)`` identity, take the
top-k, and hydrate full rows from pgvector. This is the v1 default
``RETRIEVAL_METHOD`` (≈49% tier of the Anthropic retrieval-quality ladder);
re-ranking would slot in after fusion, post-v1.
"""

from collections import defaultdict
from contextlib import ExitStack

from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_retrieve
from varagity.models.embeddings import EmbeddingsClient
from varagity.models.registry import get_model
from varagity.retrieval.base import register
from varagity.retrieval.bm25 import hydrate
from varagity.stores.bm25_store import ElasticsearchBM25
from varagity.stores.records import RetrievedChunk
from varagity.stores.vector_store import ContextualVectorDB

# Spec §11.4: pull the top k·10 from each retriever before fusing.
OVERSAMPLE = 10

Identity = tuple[str, int]


def fuse(
    semantic_keys: list[Identity],
    bm25_keys: list[Identity],
    *,
    semantic_weight: float,
    bm25_weight: float,
    k: int,
) -> list[tuple[Identity, float]]:
    """Weighted reciprocal-rank fusion of two ranked lists (spec §11.4).

    Each occurrence of an identity contributes ``weight * 1/(rank+1)`` for
    its rank in that list; an identity present in both lists accumulates both
    contributions (which is also what dedupes it). Ties keep semantic-list
    order (``sorted`` is stable and the semantic list is scored first).

    Args:
        semantic_keys: ``(doc_id, original_index)`` identities from the
            semantic arm, best first.
        bm25_keys: Identities from the BM25 arm, best first.
        semantic_weight: Weight of the semantic arm.
        bm25_weight: Weight of the BM25 arm.
        k: Number of fused results to keep.

    Returns:
        The top-k ``(identity, fused_score)`` pairs, best first. The maximum
        possible score is ``semantic_weight + bm25_weight`` (rank 0 in both
        lists).
    """
    scores: defaultdict[Identity, float] = defaultdict(float)
    for rank, key in enumerate(semantic_keys):
        scores[key] += semantic_weight * 1.0 / (rank + 1)
    for rank, key in enumerate(bm25_keys):
        scores[key] += bm25_weight * 1.0 / (rank + 1)
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [(key, scores[key]) for key in top]


@register("hybrid")
class HybridRetriever:
    """Retriever fusing the pgvector and Elasticsearch stores.

    The registry instantiates it without arguments (no I/O at import time);
    dependencies then resolve from settings per call — fresh store
    connections per query keep a long-lived chat session robust. Tests and
    the eval harness inject their own stores/client instead.
    """

    def __init__(
        self,
        *,
        store: ContextualVectorDB | None = None,
        bm25: ElasticsearchBM25 | None = None,
        embeddings: EmbeddingsClient | None = None,
    ) -> None:
        """Create the retriever.

        Args:
            store: Vector store (semantic arm + hydration); opened from
                settings per call when omitted.
            bm25: BM25 store (sparse arm); opened from settings per call
                when omitted.
            embeddings: Embeddings client for query encoding; resolved via
                the model registry when omitted.
        """
        self._store = store
        self._bm25 = bm25
        self._embeddings = embeddings

    def retrieve(self, query: str, k: int, verbose: int | None = None) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks by weighted rank fusion.

        Args:
            query: The user's query; the semantic arm embeds it in e5 query
                mode, the BM25 arm searches it unformatted.
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The top-k chunks, best first, with fused scores (maximum 1.0
            given weights summing to 1.0) and full hydrated metadata.

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If query embedding still fails after retries.
            elastic_transport.ConnectionError: If Elasticsearch is still
                unreachable after retries.
            psycopg.OperationalError: If the vector store is unreachable.
        """
        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        embeddings = self._embeddings if self._embeddings is not None else get_model("embedding")
        query_vector = embeddings.embed_query(query, verbose=verbose)

        with ExitStack() as stack:
            store = self._store
            if store is None:
                store = stack.enter_context(ContextualVectorDB())
            bm25 = self._bm25
            if bm25 is None:
                bm25 = stack.enter_context(ElasticsearchBM25())

            semantic_results = store.search(query_vector, k * OVERSAMPLE, verbose=0)
            bm25_hits = bm25.search(query, k * OVERSAMPLE, verbose=0)
            fused = fuse(
                [(chunk.doc_id, chunk.original_index) for chunk in semantic_results],
                [(hit.doc_id, hit.original_index) for hit in bm25_hits],
                semantic_weight=settings.SEMANTIC_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
                k=k,
            )
            chunks = hydrate(fused, store)
        v_retrieve(chunks, verbose)
        return chunks
