"""Unit tests for the reranked retriever (spec_v2 §5.2) and apply_rerank."""

import logging
from collections.abc import Callable

import pytest

from varagity.models.rerank import RerankResult
from varagity.retrieval import RETRIEVER_REGISTRY, get_retriever
from varagity.retrieval.reranked import RerankedRetriever, apply_rerank
from varagity.stores.records import RetrievalTrace, RetrievedChunk


def _chunk(i: int, score: float, *, trace: RetrievalTrace | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc0000000000aaa::{i}",
        doc_id="doc0000000000aaa",
        original_index=i,
        content=f"chunk content {i}",
        context=None,
        metadata={"source": "/abs/corpus/a.md", "file_name": "a.md", "page": None},
        score=score,
        trace=trace,
    )


def _traced_chunk(i: int, score: float, fused_rank: int) -> RetrievedChunk:
    return _chunk(
        i,
        score,
        trace=RetrievalTrace(
            semantic_rank=fused_rank,
            semantic_score=score,
            fused_score=score,
            fused_rank=fused_rank,
            final_rank=fused_rank,
        ),
    )


class FakeBase:
    """Records retrieve/encode calls; returns planted candidates."""

    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self.candidates = candidates
        self.retrieve_calls: list[dict[str, object]] = []
        self.encoded: list[str] = []

    def encode_query(self, query: str, verbose: int | None = None) -> list[float]:
        self.encoded.append(query)
        return [0.25, -0.25]

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        self.retrieve_calls.append({"query": query, "k": k, "query_vector": query_vector})
        return self.candidates[:k]


class FakeRerank:
    """Records rerank calls; returns planted judgments."""

    def __init__(self, results: list[RerankResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        verbose: int | None = None,
    ) -> list[RerankResult]:
        self.calls.append((query, documents))
        return self.results


@pytest.fixture
def rerank_settings(settings_env: Callable[..., None]) -> Callable[..., None]:
    """Pin the rerank knobs (enabled, small pool) for retriever tests."""

    def _pin(**overrides: object) -> None:
        values: dict[str, object] = {
            "RERANK_ENABLED": "true",
            "RERANK_BASE_METHOD": "hybrid",
            "RERANK_CANDIDATES": 5,
            "RERANK_TOP_N": 3,
            "TOP_K": 10,
        }
        values.update(overrides)
        settings_env(**values)

    return _pin


class TestRegistry:
    def test_reranked_is_registered(self) -> None:
        assert "reranked" in RETRIEVER_REGISTRY
        assert isinstance(get_retriever("reranked"), RerankedRetriever)


class TestApplyRerank:
    def test_reorders_and_records_deltas(self) -> None:
        """Item at pre-rerank rank 5 promoted to rank 1 ⇒ delta +4."""
        candidates = [_traced_chunk(i, 1.0 - i / 10, fused_rank=i + 1) for i in range(5)]
        scored = [  # cross-encoder loves the last candidate, hates the first
            RerankResult(index=4, relevance_score=0.99),
            RerankResult(index=0, relevance_score=0.20),
            RerankResult(index=1, relevance_score=0.15),
            RerankResult(index=2, relevance_score=0.10),
            RerankResult(index=3, relevance_score=0.05),
        ]
        reranked = apply_rerank(candidates, scored)

        assert [chunk.original_index for chunk in reranked] == [4, 0, 1, 2, 3]
        winner = reranked[0]
        assert winner.trace is not None
        assert winner.trace.rerank_delta == +4  # 5 → 1
        assert winner.trace.rerank_score == 0.99
        assert winner.trace.final_rank == 1
        assert winner.score == 0.99  # final score = cross-encoder relevance
        demoted = reranked[1]
        assert demoted.trace is not None
        assert demoted.trace.rerank_delta == -1  # 1 → 2
        assert demoted.trace.final_rank == 2

    def test_base_trace_fields_survive(self) -> None:
        """Fusion ranks/scores stay intact; only the rerank fields change."""
        candidates = [_traced_chunk(0, 0.8, fused_rank=1), _traced_chunk(1, 0.6, fused_rank=2)]
        scored = [
            RerankResult(index=1, relevance_score=0.9),
            RerankResult(index=0, relevance_score=0.4),
        ]
        reranked = apply_rerank(candidates, scored)
        assert reranked[0].trace is not None
        assert reranked[0].trace.fused_rank == 2  # pre-rerank rank preserved
        assert reranked[0].trace.fused_score == 0.6
        assert reranked[0].trace.semantic_rank == 2

    def test_unsorted_server_results_are_sorted_by_relevance(self) -> None:
        candidates = [_traced_chunk(i, 0.5, fused_rank=i + 1) for i in range(3)]
        scored = [  # deliberately not sorted
            RerankResult(index=0, relevance_score=0.1),
            RerankResult(index=2, relevance_score=0.9),
            RerankResult(index=1, relevance_score=0.5),
        ]
        assert [c.original_index for c in apply_rerank(candidates, scored)] == [2, 1, 0]

    def test_traceless_candidate_gets_a_trace_built(self) -> None:
        """An injected fake without a base trace still yields rerank provenance."""
        candidates = [_chunk(0, 0.7), _chunk(1, 0.3)]
        scored = [
            RerankResult(index=1, relevance_score=0.8),
            RerankResult(index=0, relevance_score=0.2),
        ]
        reranked = apply_rerank(candidates, scored)
        assert reranked[0].trace is not None
        assert reranked[0].trace.fused_score == 0.3  # built from the pre-rerank score
        assert reranked[0].trace.rerank_delta == +1


class TestRerankedRetriever:
    def test_over_fetches_pool_reranks_content_and_cuts_top_n(
        self, rerank_settings: Callable[..., None]
    ) -> None:
        rerank_settings()
        base = FakeBase([_traced_chunk(i, 1.0 - i / 10, fused_rank=i + 1) for i in range(5)])
        client = FakeRerank([RerankResult(index=i, relevance_score=float(5 - i)) for i in range(5)])
        retriever = RerankedRetriever(base=base, rerank=client)

        result = retriever.retrieve("q", k=3, verbose=0)

        # The base was asked for the widened pool, not the caller's k.
        assert base.retrieve_calls == [{"query": "q", "k": 5, "query_vector": None}]
        # The cross-encoder saw the original chunk text (content, not blurbs).
        assert client.calls == [("q", [f"chunk content {i}" for i in range(5)])]
        # RERANK_TOP_N=3 cut applied after reranking.
        assert len(result) == 3
        assert all(chunk.trace is not None and chunk.trace.rerank_score for chunk in result)

    def test_pool_widens_to_k_when_k_exceeds_candidates(
        self, rerank_settings: Callable[..., None]
    ) -> None:
        rerank_settings(RERANK_CANDIDATES=2, RERANK_TOP_N=2)
        base = FakeBase([_traced_chunk(i, 0.5, fused_rank=i + 1) for i in range(8)])
        client = FakeRerank([RerankResult(index=0, relevance_score=1.0)])
        RerankedRetriever(base=base, rerank=client).retrieve("q", k=8, verbose=0)
        assert base.retrieve_calls[0]["k"] == 8  # max(RERANK_CANDIDATES, k)

    def test_never_returns_more_than_k(self, rerank_settings: Callable[..., None]) -> None:
        """The protocol's k bounds the cut even when RERANK_TOP_N exceeds it."""
        rerank_settings(RERANK_TOP_N=5, RERANK_CANDIDATES=5)
        base = FakeBase([_traced_chunk(i, 0.5, fused_rank=i + 1) for i in range(5)])
        client = FakeRerank([RerankResult(index=i, relevance_score=float(5 - i)) for i in range(5)])
        result = RerankedRetriever(base=base, rerank=client).retrieve("q", k=2, verbose=0)
        assert len(result) == 2

    def test_kill_switch_degrades_to_base_ranking_and_logs(
        self, rerank_settings: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        rerank_settings(RERANK_ENABLED="false")
        base = FakeBase([_traced_chunk(i, 1.0 - i / 10, fused_rank=i + 1) for i in range(5)])
        client = FakeRerank([RerankResult(index=4, relevance_score=9.9)])
        retriever = RerankedRetriever(base=base, rerank=client)

        with caplog.at_level(logging.INFO):
            result = retriever.retrieve("q", k=3, verbose=0)

        assert client.calls == []  # the cross-encoder was never called
        assert [chunk.original_index for chunk in result] == [0, 1, 2]  # base order
        assert len(result) == 3  # the RERANK_TOP_N cut still applies
        assert all(chunk.trace is not None and chunk.trace.rerank_score is None for chunk in result)
        assert any("RERANK_ENABLED=false" in record.message for record in caplog.records)

    def test_encode_query_delegates_to_base(self, rerank_settings: Callable[..., None]) -> None:
        rerank_settings()
        base = FakeBase([])
        retriever = RerankedRetriever(base=base, rerank=FakeRerank([]))
        assert retriever.encode_query("what powers Aurora?", verbose=0) == [0.25, -0.25]
        assert base.encoded == ["what powers Aurora?"]

    def test_invalid_verbose_raises_before_retrieval(
        self, rerank_settings: Callable[..., None]
    ) -> None:
        rerank_settings()
        base = FakeBase([_traced_chunk(0, 0.5, fused_rank=1)])
        retriever = RerankedRetriever(base=base, rerank=FakeRerank([]))
        with pytest.raises(ValueError, match="verbose"):
            retriever.retrieve("q", k=1, verbose=7)
        assert base.retrieve_calls == []

    def test_base_resolves_from_settings_when_not_injected(
        self, rerank_settings: Callable[..., None]
    ) -> None:
        """RERANK_BASE_METHOD names the composed retriever (registry lookup)."""
        rerank_settings(RERANK_BASE_METHOD="semantic")
        from varagity.retrieval.semantic import SemanticRetriever

        retriever = RerankedRetriever()
        assert isinstance(retriever._base_retriever(), SemanticRetriever)
