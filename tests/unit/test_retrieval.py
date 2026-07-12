"""Unit tests for the retriever registry and the store-backed retrieval methods."""

from collections.abc import Callable

import pytest

from varagity.retrieval import RETRIEVER_REGISTRY, get_retriever
from varagity.retrieval.bm25 import BM25Retriever, hydrate
from varagity.retrieval.hybrid import OVERSAMPLE, HybridRetriever, fuse, fuse_with_traces
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.bm25_store import BM25Hit
from varagity.stores.records import RetrievalTrace, RetrievedChunk


def _chunk(i: int, score: float, doc_id: str = "doc0000000000aaa") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}::{i}",
        doc_id=doc_id,
        original_index=i,
        content=f"chunk content {i}",
        context=None,
        metadata={"source": f"/abs/corpus/{doc_id}.md", "file_name": f"{doc_id}.md", "page": None},
        score=score,
    )


def _hit(i: int, score: float, doc_id: str = "doc0000000000aaa") -> BM25Hit:
    return BM25Hit(
        doc_id=doc_id,
        original_index=i,
        content=f"chunk content {i}",
        contextualized_content=f"chunk content {i}",
        score=score,
    )


class FakeEmbeddings:
    """Records queries; returns a fixed vector."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str, verbose: int | None = None) -> list[float]:
        self.queries.append(query)
        return [0.5, -0.5, 0.25]


class FakeStore:
    """Records searches; returns planted chunks; hydrates from them too."""

    def __init__(
        self,
        chunks: list[RetrievedChunk],
        search_results: list[RetrievedChunk] | None = None,
    ) -> None:
        self.chunks = chunks  # the hydration pool (what "exists in pgvector")
        self.search_results = chunks if search_results is None else search_results
        self.searches: list[tuple[list[float], int]] = []
        self.hydrate_calls: list[list[tuple[str, int]]] = []

    def search(
        self, query_vector: list[float], k: int, verbose: int | None = None
    ) -> list[RetrievedChunk]:
        self.searches.append((query_vector, k))
        return self.search_results[:k]

    def fetch_by_identity(
        self, keys: list[tuple[str, int]]
    ) -> dict[tuple[str, int], RetrievedChunk]:
        self.hydrate_calls.append(list(keys))
        by_key = {(c.doc_id, c.original_index): c for c in self.chunks}
        return {key: by_key[key].model_copy(update={"score": 0.0}) for key in keys if key in by_key}


class FakeBM25:
    """Records searches; returns planted BM25 hits."""

    def __init__(self, hits: list[BM25Hit]) -> None:
        self.hits = hits
        self.searches: list[tuple[str, int]] = []

    def search(self, query: str, k: int, verbose: int | None = None) -> list[BM25Hit]:
        self.searches.append((query, k))
        return self.hits[:k]


class TestRegistry:
    def test_all_three_methods_registered(self) -> None:
        """The full spec §10.1 vocabulary resolves (Phase 6 complete)."""
        assert {"semantic", "bm25", "hybrid"} <= set(RETRIEVER_REGISTRY)
        assert isinstance(get_retriever("semantic"), SemanticRetriever)
        assert isinstance(get_retriever("bm25"), BM25Retriever)
        assert isinstance(get_retriever("hybrid"), HybridRetriever)

    def test_unknown_method_raises_listing_available(self) -> None:
        with pytest.raises(KeyError, match="semantic"):
            get_retriever("definitely-not-a-retriever")


class TestSemanticRetriever:
    def test_embeds_query_and_searches_store(self) -> None:
        chunks = [_chunk(0, 0.9), _chunk(1, 0.7)]
        embeddings = FakeEmbeddings()
        store = FakeStore(chunks)
        retriever = SemanticRetriever(store=store, embeddings=embeddings)  # type: ignore[arg-type]

        result = retriever.retrieve("what powers Aurora?", k=2, verbose=0)

        # The raw query reaches the embeddings client (which owns e5 query
        # mode — instruction wrapping is asserted in the embeddings tests).
        assert embeddings.queries == ["what powers Aurora?"]
        assert store.searches == [([0.5, -0.5, 0.25], 2)]
        # The store's results come back unchanged apart from the attached trace.
        assert [c.model_copy(update={"trace": None}) for c in result] == chunks

    def test_fills_single_arm_trace(self) -> None:
        """The cosine ranking is the ranking: fused == semantic, bm25 absent."""
        store = FakeStore([_chunk(0, 0.9), _chunk(1, 0.7)])
        retriever = SemanticRetriever(store=store, embeddings=FakeEmbeddings())  # type: ignore[arg-type]
        result = retriever.retrieve("q", k=2, verbose=0)
        assert result[0].trace == RetrievalTrace(
            semantic_rank=1,
            semantic_score=0.9,
            fused_score=0.9,
            fused_rank=1,
            final_rank=1,
        )
        assert result[1].trace is not None
        assert result[1].trace.semantic_rank == 2
        assert result[1].trace.bm25_rank is None
        assert result[1].trace.rerank_score is None

    def test_k_is_passed_through(self) -> None:
        store = FakeStore([_chunk(i, 1.0 - i / 10) for i in range(5)])
        retriever = SemanticRetriever(store=store, embeddings=FakeEmbeddings())  # type: ignore[arg-type]
        assert len(retriever.retrieve("q", k=3, verbose=0)) == 3
        assert store.searches[0][1] == 3

    def test_invalid_verbose_raises_before_embedding(self) -> None:
        embeddings = FakeEmbeddings()
        retriever = SemanticRetriever(
            store=FakeStore([]),
            embeddings=embeddings,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="verbose"):
            retriever.retrieve("q", k=1, verbose=9)
        assert embeddings.queries == []

    def test_verbose_2_renders_match_panels(self) -> None:
        from varagity.debug.show import console

        retriever = SemanticRetriever(
            store=FakeStore([_chunk(0, 0.88)]),
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        )
        with console.capture() as capture:
            retriever.retrieve("q", k=1, verbose=2)
        out = capture.get()
        assert "Retrieved" in out
        assert "chunk content 0" in out
        assert "0.8800" in out


class TestFusionMath:
    """Weighted reciprocal-rank fusion on synthetic ranked lists (spec §11.4)."""

    A, B, C = ("docA", 0), ("docB", 1), ("docC", 2)

    def test_weights_applied_at_rank_positions(self) -> None:
        fused = fuse(
            [self.A, self.B],  # semantic: A rank 0, B rank 1
            [self.B, self.C],  # bm25:     B rank 0, C rank 1
            semantic_weight=0.8,
            bm25_weight=0.2,
            k=10,
        )
        assert fused == [
            (self.A, pytest.approx(0.8)),  # 0.8 * 1/1
            (self.B, pytest.approx(0.8 / 2 + 0.2)),  # 0.8 * 1/2 + 0.2 * 1/1
            (self.C, pytest.approx(0.2 / 2)),  # 0.2 * 1/2
        ]

    def test_dedupes_on_identity(self) -> None:
        """A chunk in both lists appears once, with accumulated score."""
        fused = fuse([self.A], [self.A], semantic_weight=0.8, bm25_weight=0.2, k=10)
        assert fused == [(self.A, pytest.approx(1.0))]

    def test_top_k_cut(self) -> None:
        fused = fuse([self.A, self.B, self.C], [], semantic_weight=1.0, bm25_weight=0.0, k=2)
        assert [key for key, _ in fused] == [self.A, self.B]

    def test_rank_only_no_raw_scores(self) -> None:
        """Fusion sees identities only — a huge raw BM25 score cannot leak in.

        With equal weights, rank 0 in either list contributes identically.
        """
        by_semantic = fuse([self.A], [self.B], semantic_weight=0.5, bm25_weight=0.5, k=2)
        assert by_semantic[0][1] == by_semantic[1][1] == pytest.approx(0.5)

    def test_semantic_first_on_ties(self) -> None:
        """Stable sort keeps the semantic arm's entry ahead on equal scores."""
        fused = fuse([self.A], [self.B], semantic_weight=0.5, bm25_weight=0.5, k=2)
        assert [key for key, _ in fused] == [self.A, self.B]

    def test_empty_lists(self) -> None:
        assert fuse([], [], semantic_weight=0.8, bm25_weight=0.2, k=5) == []


class TestFuseWithTraces:
    """The trace-building sibling keeps the per-arm ranks fuse() discards."""

    A, B, C = ("docA", 0), ("docB", 1), ("docC", 2)

    def _fused(self) -> tuple[list, dict]:
        return fuse_with_traces(
            [(self.A, 0.91), (self.B, 0.55)],  # semantic arm with cosine scores
            [(self.B, 9.0), (self.C, 4.0)],  # bm25 arm with ES scores
            semantic_weight=0.8,
            bm25_weight=0.2,
            k=10,
        )

    def test_fusion_result_matches_fuse(self) -> None:
        """Fusion math is delegated — identical ranking and scores."""
        fused, _ = self._fused()
        assert fused == fuse(
            [self.A, self.B], [self.B, self.C], semantic_weight=0.8, bm25_weight=0.2, k=10
        )

    def test_both_arm_ranks_and_scores_preserved(self) -> None:
        """B sits in both arms: rank 2 semantic, rank 1 bm25, raw scores kept."""
        _, traces = self._fused()
        trace_b = traces[self.B]
        assert trace_b.semantic_rank == 2
        assert trace_b.semantic_score == 0.55
        assert trace_b.bm25_rank == 1
        assert trace_b.bm25_score == 9.0
        assert trace_b.fused_score == pytest.approx(0.8 / 2 + 0.2)
        assert trace_b.fused_rank == 2  # A fused first (0.8), B second (0.6)
        assert trace_b.final_rank == 2

    def test_single_arm_survivors_leave_the_other_arm_none(self) -> None:
        _, traces = self._fused()
        trace_a = traces[self.A]  # semantic-only
        assert (trace_a.semantic_rank, trace_a.semantic_score) == (1, 0.91)
        assert trace_a.bm25_rank is None
        assert trace_a.bm25_score is None
        trace_c = traces[self.C]  # bm25-only
        assert trace_c.semantic_rank is None
        assert (trace_c.bm25_rank, trace_c.bm25_score) == (2, 4.0)

    def test_rerank_fields_start_empty(self) -> None:
        """Fusion never fills the rerank fields — that's the reranked stage."""
        _, traces = self._fused()
        assert all(t.rerank_score is None and t.rerank_delta is None for t in traces.values())

    def test_only_fused_survivors_get_traces(self) -> None:
        fused, traces = fuse_with_traces(
            [(self.A, 0.9), (self.B, 0.8)],
            [],
            semantic_weight=1.0,
            bm25_weight=0.0,
            k=1,
        )
        assert [key for key, _ in fused] == [self.A]
        assert set(traces) == {self.A}


class TestHydrate:
    def test_missing_identity_dropped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A chunk in ES but not pgvector is dropped, not surfaced uncitable."""
        store = FakeStore([_chunk(0, 0.9)])
        known = ("doc0000000000aaa", 0)
        ghost = ("doc0000000000zzz", 7)
        with caplog.at_level("WARNING"):
            chunks = hydrate([(known, 0.5), (ghost, 0.4)], store)  # type: ignore[arg-type]
        assert [(c.doc_id, c.original_index) for c in chunks] == [known]
        assert chunks[0].score == 0.5  # caller's score attached
        assert chunks[0].trace is None  # no traces given — the pre-v2 shape
        assert any("missing from pgvector" in r.message for r in caplog.records)

    def test_attaches_traces_by_identity(self) -> None:
        store = FakeStore([_chunk(0, 0.0), _chunk(1, 0.0)])
        keys = [("doc0000000000aaa", 0), ("doc0000000000aaa", 1)]
        traces = {
            keys[0]: RetrievalTrace(fused_score=0.8, fused_rank=1, final_rank=1),
            keys[1]: RetrievalTrace(fused_score=0.6, fused_rank=2, final_rank=2),
        }
        chunks = hydrate([(keys[0], 0.8), (keys[1], 0.6)], store, traces=traces)  # type: ignore[arg-type]
        assert chunks[0].trace == traces[keys[0]]
        assert chunks[1].trace == traces[keys[1]]

    def test_identity_absent_from_traces_hydrates_with_none(self) -> None:
        store = FakeStore([_chunk(0, 0.0)])
        key = ("doc0000000000aaa", 0)
        chunks = hydrate([(key, 0.5)], store, traces={})  # type: ignore[arg-type]
        assert chunks[0].trace is None


class TestBM25Retriever:
    def test_searches_es_and_hydrates_in_order(self) -> None:
        chunks = [_chunk(0, 0.0), _chunk(1, 0.0)]
        bm25 = FakeBM25([_hit(1, 7.5), _hit(0, 3.25)])  # ES order: chunk 1 first
        store = FakeStore(chunks)
        retriever = BM25Retriever(bm25=bm25, store=store)  # type: ignore[arg-type]

        result = retriever.retrieve("rare keyword", k=2, verbose=0)

        assert bm25.searches == [("rare keyword", 2)]
        # ES ranking preserved, ES scores attached, full metadata hydrated
        assert [c.original_index for c in result] == [1, 0]
        assert [c.score for c in result] == [7.5, 3.25]
        assert all(c.metadata["source"].endswith(".md") for c in result)

    def test_invalid_verbose_raises_before_search(self) -> None:
        bm25 = FakeBM25([])
        retriever = BM25Retriever(bm25=bm25, store=FakeStore([]))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="verbose"):
            retriever.retrieve("q", k=1, verbose=7)
        assert bm25.searches == []

    def test_fills_single_arm_trace(self) -> None:
        """The ES ranking is the ranking: fused == bm25, semantic absent."""
        bm25 = FakeBM25([_hit(1, 7.5), _hit(0, 3.25)])
        retriever = BM25Retriever(bm25=bm25, store=FakeStore([_chunk(0, 0.0), _chunk(1, 0.0)]))  # type: ignore[arg-type]
        result = retriever.retrieve("q", k=2, verbose=0)
        assert result[0].trace == RetrievalTrace(
            bm25_rank=1,
            bm25_score=7.5,
            fused_score=7.5,
            fused_rank=1,
            final_rank=1,
        )
        assert result[1].trace is not None
        assert result[1].trace.bm25_rank == 2
        assert result[1].trace.semantic_rank is None
        assert result[1].trace.rerank_score is None


class TestHybridRetriever:
    @pytest.fixture
    def pinned_weights(self, settings_env: Callable[..., None]) -> None:
        settings_env(SEMANTIC_WEIGHT="0.8", BM25_WEIGHT="0.2", TOP_K="10")

    def test_oversamples_fuses_and_hydrates(self, pinned_weights: None) -> None:
        semantic_chunks = [_chunk(0, 0.99), _chunk(1, 0.55)]
        bm25 = FakeBM25([_hit(1, 9.0), _hit(2, 4.0)])
        store = FakeStore(semantic_chunks + [_chunk(2, 0.0)])
        embeddings = FakeEmbeddings()
        retriever = HybridRetriever(
            store=store,  # type: ignore[arg-type]
            bm25=bm25,  # type: ignore[arg-type]
            embeddings=embeddings,  # type: ignore[arg-type]
        )

        result = retriever.retrieve("what powers Aurora?", k=2, verbose=0)

        # Both arms over-retrieve k * OVERSAMPLE (spec §11.4).
        assert store.searches == [([0.5, -0.5, 0.25], 2 * OVERSAMPLE)]
        assert bm25.searches == [("what powers Aurora?", 2 * OVERSAMPLE)]
        assert embeddings.queries == ["what powers Aurora?"]

        # Fused: chunk0 = 0.8; chunk1 = 0.8/2 + 0.2 = 0.6; chunk2 = 0.1 (cut).
        assert [c.original_index for c in result] == [0, 1]
        assert result[0].score == pytest.approx(0.8)
        assert result[1].score == pytest.approx(0.6)
        # top-k cut applied after fusion
        assert len(result) == 2

    def test_bm25_only_chunk_can_win(self, pinned_weights: None) -> None:
        """A chunk the semantic arm never returned is still retrievable."""
        bm25 = FakeBM25([_hit(5, 12.0)])
        store = FakeStore([_chunk(5, 0.0)], search_results=[])  # semantic arm finds nothing
        retriever = HybridRetriever(
            store=store,  # type: ignore[arg-type]
            bm25=bm25,  # type: ignore[arg-type]
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        )
        result = retriever.retrieve("rare term", k=3, verbose=0)
        assert [c.original_index for c in result] == [5]
        assert result[0].score == pytest.approx(0.2)  # bm25 weight * 1/1

    def test_traces_carry_per_arm_ranks_through_hydration(self, pinned_weights: None) -> None:
        """The spec_v2 §9.2 seam: fuse no longer discards the per-arm ranks."""
        semantic_chunks = [_chunk(0, 0.99), _chunk(1, 0.55)]
        bm25 = FakeBM25([_hit(1, 9.0), _hit(2, 4.0)])
        # Hydration pool holds all three; the semantic *arm* returns only 0–1.
        store = FakeStore(semantic_chunks + [_chunk(2, 0.0)], search_results=semantic_chunks)
        retriever = HybridRetriever(
            store=store,  # type: ignore[arg-type]
            bm25=bm25,  # type: ignore[arg-type]
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        )

        result = retriever.retrieve("q", k=3, verbose=0)

        # chunk 0: semantic-only, fused rank 1.
        trace0 = result[0].trace
        assert trace0 is not None
        assert (trace0.semantic_rank, trace0.semantic_score) == (1, 0.99)
        assert trace0.bm25_rank is None
        assert trace0.fused_score == pytest.approx(0.8)
        assert (trace0.fused_rank, trace0.final_rank) == (1, 1)
        # chunk 1: in both arms — both ranks and raw scores preserved.
        trace1 = result[1].trace
        assert trace1 is not None
        assert (trace1.semantic_rank, trace1.semantic_score) == (2, 0.55)
        assert (trace1.bm25_rank, trace1.bm25_score) == (1, 9.0)
        assert trace1.fused_score == pytest.approx(0.6)
        # chunk 2: bm25-only.
        trace2 = result[2].trace
        assert trace2 is not None
        assert trace2.semantic_rank is None
        assert (trace2.bm25_rank, trace2.bm25_score) == (2, 4.0)
        # No rerank stage on the hybrid path.
        assert trace1.rerank_score is None and trace1.rerank_delta is None
