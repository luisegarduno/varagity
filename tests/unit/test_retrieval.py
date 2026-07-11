"""Unit tests for the retriever registry and the three retrieval methods."""

from collections.abc import Callable

import pytest

from varagity.retrieval import RETRIEVER_REGISTRY, get_retriever
from varagity.retrieval.bm25 import BM25Retriever, hydrate
from varagity.retrieval.hybrid import OVERSAMPLE, HybridRetriever, fuse
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.bm25_store import BM25Hit
from varagity.stores.records import RetrievedChunk


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
        assert result == chunks

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
        assert any("missing from pgvector" in r.message for r in caplog.records)


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
