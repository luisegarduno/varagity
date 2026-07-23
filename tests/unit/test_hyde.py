"""Unit tests for the HyDE retriever (ADR-016)."""

import logging
from collections.abc import Callable

import pytest

from varagity.retrieval import RETRIEVER_REGISTRY, get_retriever
from varagity.retrieval.hyde import HYDE_PASSAGE_LABEL, HYDE_PROMPT, HydeRetriever
from varagity.stores.records import RetrievalTrace, RetrievedChunk

QUERY_VECTOR = [0.1, 0.2]  # what the base's own (e5 query-mode) encoding returns
PASSAGE_VECTOR = [0.9, 0.8]  # what the passage-mode embedding returns


def _chunk(i: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc0000000000aaa::{i}",
        doc_id="doc0000000000aaa",
        original_index=i,
        content=f"chunk content {i}",
        context=None,
        metadata={"source": "/abs/corpus/a.md", "file_name": "a.md", "page": None},
        score=score,
        trace=RetrievalTrace(fused_score=score, fused_rank=i + 1, final_rank=i + 1),
    )


class FakeBase:
    """Records retrieve/encode calls; returns planted candidates."""

    def __init__(self, candidates: list[RetrievedChunk] | None = None) -> None:
        self.candidates = candidates or []
        self.retrieve_calls: list[dict[str, object]] = []
        self.encoded: list[str] = []

    def encode_query(self, query: str, verbose: int | None = None) -> list[float]:
        self.encoded.append(query)
        return QUERY_VECTOR

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


class FakeLLM:
    """Records generate calls; returns planted text (or raises)."""

    def __init__(self, response: str = "a hypothetical passage", *, error: bool = False) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self.error:
            raise RuntimeError("llm down")
        return self.response


class FakeEmbeddings:
    """Records both e5 modes so tests can assert which one HyDE used."""

    def __init__(self) -> None:
        self.passages: list[list[str]] = []
        self.queries: list[str] = []

    def embed_passages(self, texts: list[str], verbose: int | None = None) -> list[list[float]]:
        self.passages.append(list(texts))
        return [PASSAGE_VECTOR for _ in texts]

    def embed_query(self, query: str, verbose: int | None = None) -> list[float]:
        self.queries.append(query)
        return QUERY_VECTOR


@pytest.fixture
def hyde_settings(settings_env: Callable[..., None]) -> Callable[..., None]:
    """Pin the HyDE knobs (enabled, hybrid base, small caps) for retriever tests."""

    def _pin(**overrides: object) -> None:
        values: dict[str, object] = {
            "HYDE_ENABLED": "true",
            "HYDE_BASE_METHOD": "hybrid",
            "HYDE_MODEL_TYPE": "default",
            "HYDE_MAX_TOKENS": 256,
            "HYDE_MAX_CHARS": 200,
        }
        values.update(overrides)
        settings_env(**values)

    return _pin


class TestRegistry:
    def test_hyde_is_registered(self) -> None:
        assert "hyde" in RETRIEVER_REGISTRY
        assert isinstance(get_retriever("hyde"), HydeRetriever)


class TestPrompt:
    def test_template_ends_with_the_echo_strip_label(self) -> None:
        """The label the retriever strips must stay the prompt's last token."""
        assert HYDE_PROMPT.endswith(HYDE_PASSAGE_LABEL)

    def test_template_has_the_query_slot(self) -> None:
        assert "{query}" in HYDE_PROMPT


class TestEncodeQuery:
    def test_generates_then_embeds_the_passage_in_passage_mode(
        self, hyde_settings: Callable[..., None]
    ) -> None:
        """The cleaned passage is embedded via embed_passages — never embed_query."""
        hyde_settings()
        llm = FakeLLM("<think>let me think</think>PASSAGE: Aurora runs on a fusion core.")
        embeddings = FakeEmbeddings()
        base = FakeBase()
        retriever = HydeRetriever(base=base, llm=llm, embeddings=embeddings)  # type: ignore[arg-type]

        vector = retriever.encode_query("what powers Aurora?", verbose=0)

        assert vector == PASSAGE_VECTOR
        # <think> stripped AND the completion-priming label echo stripped.
        assert embeddings.passages == [["Aurora runs on a fusion core."]]
        assert embeddings.queries == []  # passage mode, not e5 query mode
        assert base.encoded == []  # no fallback happened
        [call] = llm.calls
        assert call["max_tokens"] == 256  # HYDE_MAX_TOKENS reaches the client
        [message] = call["messages"]  # type: ignore[misc]
        assert "what powers Aurora?" in message["content"]

    def test_llm_failure_falls_back_to_the_base_encoding(
        self, hyde_settings: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        hyde_settings()
        base = FakeBase()
        retriever = HydeRetriever(
            base=base,
            llm=FakeLLM(error=True),
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        )
        with caplog.at_level(logging.WARNING):
            vector = retriever.encode_query("q", verbose=0)
        assert vector == QUERY_VECTOR
        assert base.encoded == ["q"]
        assert any("HyDE LLM call failed" in record.message for record in caplog.records)

    def test_empty_passage_after_think_strip_falls_back(
        self, hyde_settings: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unclosed <think> cleans to nothing — the raw query must be searched."""
        hyde_settings()
        base = FakeBase()
        retriever = HydeRetriever(
            base=base,
            llm=FakeLLM("<think>still reasoning when the cap hit"),
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        )
        with caplog.at_level(logging.WARNING):
            vector = retriever.encode_query("q", verbose=0)
        assert vector == QUERY_VECTOR
        assert any("empty passage" in record.message for record in caplog.records)

    def test_overlong_passage_falls_back(
        self, hyde_settings: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        hyde_settings(HYDE_MAX_CHARS=10)
        base = FakeBase()
        retriever = HydeRetriever(
            base=base,
            llm=FakeLLM("way more than ten characters of hypothetical text"),
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        )
        with caplog.at_level(logging.WARNING):
            vector = retriever.encode_query("q", verbose=0)
        assert vector == QUERY_VECTOR
        assert any("HYDE_MAX_CHARS" in record.message for record in caplog.records)

    def test_kill_switch_degrades_to_base_encoding_and_logs(
        self, hyde_settings: Callable[..., None], caplog: pytest.LogCaptureFixture
    ) -> None:
        hyde_settings(HYDE_ENABLED="false")
        base = FakeBase()
        llm = FakeLLM()
        retriever = HydeRetriever(base=base, llm=llm, embeddings=FakeEmbeddings())  # type: ignore[arg-type]
        with caplog.at_level(logging.INFO):
            vector = retriever.encode_query("q", verbose=0)
        assert llm.calls == []  # the LLM was never called
        assert vector == QUERY_VECTOR
        assert base.encoded == ["q"]
        assert any("HYDE_ENABLED=false" in record.message for record in caplog.records)


class TestRetrieve:
    def test_base_gets_the_original_query_with_the_passage_vector(
        self, hyde_settings: Callable[..., None]
    ) -> None:
        """Dense-arm-only substitution: text stays the user's words."""
        hyde_settings()
        base = FakeBase([_chunk(i, 1.0 - i / 10) for i in range(5)])
        retriever = HydeRetriever(
            base=base,
            llm=FakeLLM("Aurora runs on a fusion core."),
            embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        )

        result = retriever.retrieve("what powers Aurora?", k=3, verbose=0)

        assert base.retrieve_calls == [
            {"query": "what powers Aurora?", "k": 3, "query_vector": PASSAGE_VECTOR}
        ]
        # Chunks and their traces pass through untouched — the base's
        # ranking is the ranking.
        assert result == base.candidates[:3]

    def test_provided_query_vector_skips_generation(
        self, hyde_settings: Callable[..., None]
    ) -> None:
        """The flow's embed stage already generated; retrieve must not re-pay."""
        hyde_settings()
        base = FakeBase([_chunk(0, 0.9)])
        llm = FakeLLM()
        retriever = HydeRetriever(base=base, llm=llm, embeddings=FakeEmbeddings())  # type: ignore[arg-type]

        retriever.retrieve("q", k=1, verbose=0, query_vector=PASSAGE_VECTOR)

        assert llm.calls == []
        assert base.retrieve_calls == [{"query": "q", "k": 1, "query_vector": PASSAGE_VECTOR}]

    def test_invalid_verbose_raises_before_any_work(
        self, hyde_settings: Callable[..., None]
    ) -> None:
        hyde_settings()
        base = FakeBase([_chunk(0, 0.9)])
        llm = FakeLLM()
        retriever = HydeRetriever(base=base, llm=llm, embeddings=FakeEmbeddings())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="verbose"):
            retriever.retrieve("q", k=1, verbose=7)
        assert llm.calls == []
        assert base.retrieve_calls == []


class TestBaseResolution:
    def test_base_resolves_from_settings_when_not_injected(
        self, hyde_settings: Callable[..., None]
    ) -> None:
        """HYDE_BASE_METHOD names the composed retriever (registry lookup)."""
        hyde_settings(HYDE_BASE_METHOD="semantic")
        from varagity.retrieval.semantic import SemanticRetriever

        retriever = HydeRetriever()
        assert isinstance(retriever._base_retriever(), SemanticRetriever)
