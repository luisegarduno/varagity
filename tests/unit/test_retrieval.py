"""Unit tests for the retriever registry and semantic retrieval."""

import pytest

from varagity.retrieval import RETRIEVER_REGISTRY, get_retriever
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.records import RetrievedChunk


def _chunk(i: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc0000000000aaa::{i}",
        doc_id="doc0000000000aaa",
        original_index=i,
        content=f"chunk content {i}",
        context=None,
        metadata={"source": "/abs/corpus/a.md", "file_name": "a.md", "page": None},
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
    """Records searches; returns planted chunks."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.searches: list[tuple[list[float], int]] = []

    def search(
        self, query_vector: list[float], k: int, verbose: int | None = None
    ) -> list[RetrievedChunk]:
        self.searches.append((query_vector, k))
        return self.chunks[:k]


class TestRegistry:
    def test_semantic_is_registered(self) -> None:
        assert "semantic" in RETRIEVER_REGISTRY
        assert isinstance(get_retriever("semantic"), SemanticRetriever)

    def test_unknown_method_raises_listing_available(self) -> None:
        with pytest.raises(KeyError, match="semantic"):
            get_retriever("definitely-not-a-retriever")

    def test_phase_6_methods_not_yet_registered(self) -> None:
        """bm25/hybrid pass config validation but land in Phase 6."""
        for name in ("bm25", "hybrid"):
            with pytest.raises(KeyError, match="Unknown retrieval method"):
                get_retriever(name)


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
