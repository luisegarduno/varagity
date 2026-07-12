"""Hybrid retrieval — weighted reciprocal-rank fusion (spec §10.1, §11.4).

The cookbook's fusion, verbatim in structure: over-retrieve ``k * 10`` from
**each** store, combine by weighted reciprocal rank (``weight * 1/(rank+1)``
— only rank positions matter, raw scores from the two stores are never
compared), dedupe on the ``(doc_id, original_index)`` identity, take the
top-k, and hydrate full rows from pgvector. This is the default
``RETRIEVAL_METHOD`` (≈49% tier of the Anthropic retrieval-quality ladder);
the ``reranked`` method (spec_v2 §5) composes this retriever and
cross-encodes its output for the ≈67% tier.
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
from varagity.stores.records import RetrievalTrace, RetrievedChunk
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


def fuse_with_traces(
    semantic: list[tuple[Identity, float]],
    bm25: list[tuple[Identity, float]],
    *,
    semantic_weight: float,
    bm25_weight: float,
    k: int,
) -> tuple[list[tuple[Identity, float]], dict[Identity, RetrievalTrace]]:
    """Fuse two *scored* ranked lists, keeping the per-arm ranks (spec_v2 §9.2).

    The trace-building sibling of :func:`fuse`: fusion math is identical
    (delegated), but the per-arm rank/score maps :func:`fuse` builds and
    discards are preserved into a :class:`~varagity.stores.records
    .RetrievalTrace` per fused survivor — the "why it ranked here" data the
    provenance panel renders. Raw arm scores still never enter the fusion
    math; they ride along for display only.

    Args:
        semantic: ``(identity, cosine_score)`` pairs from the semantic arm,
            best first.
        bm25: ``(identity, bm25_score)`` pairs from the BM25 arm, best first.
        semantic_weight: Weight of the semantic arm.
        bm25_weight: Weight of the BM25 arm.
        k: Number of fused results to keep.

    Returns:
        The :func:`fuse` result (top-k ``(identity, fused_score)`` pairs,
        best first) plus a per-identity trace for each survivor, with
        1-based per-arm/fused ranks and ``final_rank == fused_rank`` (the
        rerank stage, when active, overwrites the final fields downstream).
    """
    fused = fuse(
        [key for key, _ in semantic],
        [key for key, _ in bm25],
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        k=k,
    )
    semantic_arm = {key: (rank, score) for rank, (key, score) in enumerate(semantic, start=1)}
    bm25_arm = {key: (rank, score) for rank, (key, score) in enumerate(bm25, start=1)}
    traces: dict[Identity, RetrievalTrace] = {}
    for fused_rank, (key, fused_score) in enumerate(fused, start=1):
        semantic_entry = semantic_arm.get(key)
        bm25_entry = bm25_arm.get(key)
        traces[key] = RetrievalTrace(
            semantic_rank=semantic_entry[0] if semantic_entry else None,
            semantic_score=semantic_entry[1] if semantic_entry else None,
            bm25_rank=bm25_entry[0] if bm25_entry else None,
            bm25_score=bm25_entry[1] if bm25_entry else None,
            fused_score=fused_score,
            fused_rank=fused_rank,
            final_rank=fused_rank,
        )
    return fused, traces


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

    def encode_query(self, query: str, verbose: int | None = None) -> list[float]:
        """Embed a query in e5 query mode for the semantic arm (spec §9.5).

        Args:
            query: The user's query, unformatted.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The query embedding (the BM25 arm searches the raw text).

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If query embedding still fails after retries.
        """
        verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
        embeddings = self._embeddings if self._embeddings is not None else get_model("embedding")
        return embeddings.embed_query(query, verbose=verbose)

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks by weighted rank fusion.

        Args:
            query: The user's query; the semantic arm embeds it in e5 query
                mode, the BM25 arm searches it unformatted.
            k: Number of chunks to return.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.
            query_vector: Pre-computed :meth:`encode_query` output; encoded
                here when omitted.

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
        if query_vector is None:
            query_vector = self.encode_query(query, verbose=verbose)

        with ExitStack() as stack:
            store = self._store
            if store is None:
                store = stack.enter_context(ContextualVectorDB())
            bm25 = self._bm25
            if bm25 is None:
                bm25 = stack.enter_context(ElasticsearchBM25())

            semantic_results = store.search(query_vector, k * OVERSAMPLE, verbose=0)
            bm25_hits = bm25.search(query, k * OVERSAMPLE, verbose=0)
            fused, traces = fuse_with_traces(
                [((chunk.doc_id, chunk.original_index), chunk.score) for chunk in semantic_results],
                [((hit.doc_id, hit.original_index), hit.score) for hit in bm25_hits],
                semantic_weight=settings.SEMANTIC_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
                k=k,
            )
            chunks = hydrate(fused, store, traces=traces)
        v_retrieve(chunks, verbose)
        return chunks
