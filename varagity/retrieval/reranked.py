"""Cross-encoder re-ranking over a base retriever (spec_v2 §5 — the ≈67% tier).

Re-ranking *composes* an existing retrieval method rather than replacing
fusion: over-fetch a wide candidate pool (``RERANK_CANDIDATES``) from the
configured ``RERANK_BASE_METHOD``, cross-encode every candidate's original
``content`` against the query (the contextual blurb already did its job at
the embedding/BM25 stage), and keep the ``RERANK_TOP_N`` most relevant —
the Anthropic cookbook's 150→20 over-fetch, scaled to this corpus.

``RERANK_ENABLED`` is a kill switch orthogonal to method selection: when
``false`` the retriever degrades to its base method's ranking (logged), so
a GUI toggle and the eval baseline both work without renaming the method.
"""

import logging
import time

from varagity.config import get_settings
from varagity.debug.show import check_verbose, v_retrieve
from varagity.models.registry import get_model
from varagity.models.rerank import RerankClient, RerankResult
from varagity.observability import metrics
from varagity.retrieval.base import Retriever, get_retriever, register
from varagity.stores.records import RetrievalTrace, RetrievedChunk

logger = logging.getLogger(__name__)


def apply_rerank(
    candidates: list[RetrievedChunk],
    scored: list[RerankResult],
) -> list[RetrievedChunk]:
    """Reorder candidates by cross-encoder relevance, filling the rerank trace.

    Each result's ``index`` points into ``candidates`` (which arrive
    best-first from the base retriever, so ``index + 1`` is the pre-rerank
    rank). The surviving chunk's ``score`` becomes the cross-encoder
    relevance — the "final score" the provenance panel shows — while the
    base retriever's fused score/ranks stay intact on the trace.

    Args:
        candidates: The base retriever's candidate pool, best first.
        scored: Cross-encoder judgments for (a subset of) the candidates.

    Returns:
        The judged candidates, most relevant first, each with
        ``rerank_score``, ``rerank_delta`` (pre-rank − post-rank: + moved
        up, − moved down), and ``final_rank`` recorded on its trace. A
        candidate without a base trace (an injected test fake) gets one
        built from its pre-rerank score and rank.
    """
    ordered = sorted(scored, key=lambda result: result.relevance_score, reverse=True)
    reranked: list[RetrievedChunk] = []
    for final_rank, result in enumerate(ordered, start=1):
        candidate = candidates[result.index]
        pre_rank = result.index + 1
        trace = candidate.trace
        if trace is None:
            trace = RetrievalTrace(
                fused_score=candidate.score, fused_rank=pre_rank, final_rank=pre_rank
            )
        trace = trace.model_copy(
            update={
                "rerank_score": result.relevance_score,
                "rerank_delta": pre_rank - final_rank,
                "final_rank": final_rank,
            }
        )
        reranked.append(
            candidate.model_copy(update={"score": result.relevance_score, "trace": trace})
        )
    return reranked


@register("reranked")
class RerankedRetriever:
    """Base retriever → cross-encoder rerank → top-N (spec_v2 §5.2).

    The registry instantiates it without arguments (no I/O at import time);
    the base retriever and rerank client then resolve from settings per
    call, so fusion improvements flow through and a config change needs no
    restart. Tests and the eval harness inject their own instead.
    """

    def __init__(
        self,
        *,
        base: Retriever | None = None,
        rerank: RerankClient | None = None,
    ) -> None:
        """Create the retriever.

        Args:
            base: Retriever producing the candidate pool; resolved from
                ``settings.RERANK_BASE_METHOD`` per call when omitted.
            rerank: Cross-encoder client; resolved via the model registry
                (``get_model("rerank")``) per call when omitted.
        """
        self._base = base
        self._rerank = rerank

    def _base_retriever(self) -> Retriever:
        """Resolve the composed base retriever (injected or from settings).

        Returns:
            The base retriever instance.

        Raises:
            KeyError: If ``settings.RERANK_BASE_METHOD`` names an
                unregistered method (config validation makes this
                unreachable for env-sourced settings).
        """
        if self._base is not None:
            return self._base
        return get_retriever(get_settings().RERANK_BASE_METHOD)

    def encode_query(self, query: str, verbose: int | None = None) -> list[float] | None:
        """Encode a query exactly as the base retriever would (spec_v2 §5.2).

        Args:
            query: The user's query, unformatted.
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.

        Returns:
            The base retriever's query encoding (``None`` for a ``bm25``
            base — nothing to encode).

        Raises:
            ValueError: If ``verbose`` is invalid.
            openai.APIError: If query embedding still fails after retries.
        """
        return self._base_retriever().encode_query(query, verbose)

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve a wide candidate pool, cross-encode it, keep the top-N.

        The pool is ``max(RERANK_CANDIDATES, k)`` from the base retriever;
        the cut is ``min(k, RERANK_TOP_N)`` (re-ranking narrows — it never
        returns more than the caller asked for). With ``RERANK_ENABLED``
        off, the base ranking passes through the same cut, logged as a
        degradation.

        Args:
            query: The user's query; the base retriever owns its encoding,
                the cross-encoder scores the raw pair.
            k: Number of chunks the caller wants (bounds the ``top_n`` cut).
            verbose: Console verbosity (0–2); defaults to
                ``settings.DEFAULT_VERBOSE``.
            query_vector: Pre-computed :meth:`encode_query` output; passed
                through to the base retriever.

        Returns:
            The ``min(k, RERANK_TOP_N)`` most relevant chunks, best first,
            with cross-encoder scores and the full rank provenance trace
            (or the base ranking when the kill switch is off).

        Raises:
            ValueError: If ``verbose`` is invalid, or the rerank request is
                permanently rejected (e.g. ``RERANK_MODEL`` is not a served
                cross-encoder).
            httpx.HTTPError: If the rerank request still fails after
                retries.
        """
        settings = get_settings()
        verbose = check_verbose(settings.DEFAULT_VERBOSE if verbose is None else verbose)
        top_n = min(k, settings.RERANK_TOP_N)
        pool = max(settings.RERANK_CANDIDATES, k)
        candidates = self._base_retriever().retrieve(
            query, k=pool, verbose=0, query_vector=query_vector
        )
        if not settings.RERANK_ENABLED:
            logger.info(
                "RERANK_ENABLED=false — degrading to the %r base method's ranking",
                settings.RERANK_BASE_METHOD,
            )
            chunks = candidates[:top_n]
        else:
            client = self._rerank if self._rerank is not None else get_model("rerank")
            rerank_started = time.perf_counter()
            scored = client.rerank(query, [chunk.content for chunk in candidates], verbose=0)
            chunks = apply_rerank(candidates, scored)[:top_n]
            # The rerank sub-stage's share of retrieval (spec_v2 §6.2 —
            # "is it earning its latency?"); the flow's `retrieve`
            # observation includes it.
            metrics.observe_query_stage("rerank", "reranked", time.perf_counter() - rerank_started)
        v_retrieve(chunks, verbose)
        return chunks
