"""Unit tests for context formatting and answer generation (spec §15.2)."""

from collections.abc import Callable, Sequence

import pytest

from varagity.generation.answer import (
    ANSWER_PROMPT,
    answer_query,
    format_context,
    generate_answer,
)
from varagity.stores.records import RetrievedChunk


def _chunk(
    i: int, content: str, *, context: str | None = None, source: str = "/abs/corpus/a.md"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc0000000000aaa::{i}",
        doc_id="doc0000000000aaa",
        original_index=i,
        content=content,
        context=context,
        metadata={"source": source, "file_name": source.rsplit("/", 1)[-1], "page": None},
        score=0.9 - i / 10,
    )


class StubLLM:
    """Records generate() calls; returns a scripted response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[Sequence[dict[str, str]]] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        self.calls.append(messages)
        return self.response


class TestFormatContext:
    def test_null_context_renders_empty(self) -> None:
        """The [CONTEXT] line is present but empty until Phase 5 (stable format)."""
        block = format_context([_chunk(0, "some content")])
        assert block == "[SOURCE]:  /abs/corpus/a.md\n[CONTEXT]: \n[CONTENT]: some content"

    def test_context_blurb_rendered_when_present(self) -> None:
        block = format_context([_chunk(0, "the content", context="This chunk situates X.")])
        assert block == (
            "[SOURCE]:  /abs/corpus/a.md\n[CONTEXT]: This chunk situates X.\n[CONTENT]: the content"
        )

    def test_chunks_joined_blank_line_separated_in_order(self) -> None:
        blocks = format_context([_chunk(0, "first"), _chunk(1, "second", source="/abs/b.txt")])
        first, second = blocks.split("\n\n")
        assert "[CONTENT]: first" in first
        assert "[SOURCE]:  /abs/b.txt" in second
        assert "[CONTENT]: second" in second

    def test_empty_retrieval_formats_to_empty_string(self) -> None:
        assert format_context([]) == ""


class TestGenerateAnswer:
    def test_prompt_carries_grounding_scaffold_and_context(self) -> None:
        llm = StubLLM("Grounded answer. [SOURCE]: /abs/corpus/a.md")
        chunks = [_chunk(0, "Lantern produces 4.2 megawatts at peak.")]
        generate_answer("What powers Aurora?", chunks, llm=llm, verbose=0)  # type: ignore[arg-type]

        assert len(llm.calls) == 1
        (message,) = llm.calls[0]
        assert message["role"] == "user"
        prompt = message["content"]
        # spec §10.2 scaffold, verbatim
        assert "You are Varagity, a retrieval-augmented assistant." in prompt
        assert "Answer the user's QUESTION using ONLY the CONTEXT below." in prompt
        assert "If the answer is not contained in the context, say you don't know" in prompt
        assert "Cite the [SOURCE] of any facts you use." in prompt
        # the retrieved evidence and the question are inside
        assert "Lantern produces 4.2 megawatts at peak." in prompt
        assert f"<context>\n{format_context(chunks)}\n</context>" in prompt
        assert "QUESTION: What powers Aurora?" in prompt
        assert prompt.endswith("ANSWER:")

    def test_answer_is_think_stripped(self) -> None:
        llm = StubLLM("<think>scanning chunks…</think>The reactor is Lantern.")
        answer = generate_answer("q", [_chunk(0, "c")], llm=llm, verbose=0)  # type: ignore[arg-type]
        assert answer == "The reactor is Lantern."

    def test_dont_know_response_passes_through_unaltered(self) -> None:
        """The don't-know contract lives in the prompt (spec §15.2).

        The stub's admission must reach the caller verbatim.
        """
        llm = StubLLM("I don't know — the context does not mention this.")
        answer = generate_answer(
            "What is the airspeed of an unladen swallow?",
            [_chunk(0, "Lantern produces 4.2 megawatts.")],
            llm=llm,  # type: ignore[arg-type]
            verbose=0,
        )
        assert answer == "I don't know — the context does not mention this."
        prompt = llm.calls[0][0]["content"]
        assert "say you don't know — do not fabricate" in prompt

    def test_precomputed_formatted_context_is_used(self) -> None:
        llm = StubLLM("ok")
        generate_answer(
            "q",
            [_chunk(0, "ignored when precomputed")],
            llm=llm,  # type: ignore[arg-type]
            formatted_context="PRECOMPUTED BLOCK",
            verbose=0,
        )
        prompt = llm.calls[0][0]["content"]
        assert "PRECOMPUTED BLOCK" in prompt
        assert "ignored when precomputed" not in prompt

    def test_invalid_verbose_raises_before_llm_call(self) -> None:
        llm = StubLLM("ok")
        with pytest.raises(ValueError, match="verbose"):
            generate_answer("q", [], llm=llm, verbose=5)  # type: ignore[arg-type]
        assert llm.calls == []


class FakeRetriever:
    """Records retrieve() calls; returns planted chunks."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int, verbose: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self.chunks


class TestAnswerQuery:
    def test_state_dict_threads_all_fields(self) -> None:
        chunks = [_chunk(0, "evidence")]
        retriever = FakeRetriever(chunks)
        llm = StubLLM("<think>…</think>The answer.")

        state = answer_query(
            "the question?",
            retriever=retriever,
            llm=llm,
            k=4,
            verbose=0,  # type: ignore[arg-type]
        )

        assert retriever.calls == [("the question?", 4)]
        assert state["query"] == "the question?"
        assert state["query_vector"] is None  # encapsulated by the retrieval seam in Phase 4
        assert state["retrieved"] == chunks
        assert state["formatted_context"] == format_context(chunks)
        assert state["answer"] == "The answer."

    def test_k_defaults_to_settings_top_k(self, settings_env: Callable[..., None]) -> None:
        settings_env(TOP_K=7)
        retriever = FakeRetriever([])
        answer_query("q", retriever=retriever, llm=StubLLM("x"), verbose=0)  # type: ignore[arg-type]
        assert retriever.calls == [("q", 7)]

    def test_on_retrieved_hook_fires_before_generation(self) -> None:
        events: list[str] = []
        chunks = [_chunk(0, "c")]

        class OrderedLLM(StubLLM):
            def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                events.append("generate")
                return super().generate(messages, **kwargs)

        answer_query(
            "q",
            retriever=FakeRetriever(chunks),  # type: ignore[arg-type]
            llm=OrderedLLM("x"),  # type: ignore[arg-type]
            verbose=0,
            on_retrieved=lambda got: events.append(f"retrieved:{len(got)}"),
        )
        assert events == ["retrieved:1", "generate"]

    def test_unregistered_default_method_raises_key_error(
        self, settings_env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """answer_query propagates a registry miss for the configured method."""
        from varagity.retrieval import RETRIEVER_REGISTRY

        settings_env(RETRIEVAL_METHOD="bm25")
        monkeypatch.delitem(RETRIEVER_REGISTRY, "bm25")
        with pytest.raises(KeyError, match="Unknown retrieval method"):
            answer_query("q", llm=StubLLM("x"), verbose=0)  # type: ignore[arg-type]

    def test_prompt_contains_the_retrieved_evidence(self) -> None:
        llm = StubLLM("cited answer")
        answer_query(
            "where is the corridor?",
            retriever=FakeRetriever([_chunk(0, "a 1.8-kilometer strip of kelp")]),  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            verbose=0,
        )
        assert "a 1.8-kilometer strip of kelp" in llm.calls[0][0]["content"]


def test_answer_prompt_is_the_spec_verbatim_template() -> None:
    assert ANSWER_PROMPT == (
        "You are Varagity, a retrieval-augmented assistant.\n"
        "Answer the user's QUESTION using ONLY the CONTEXT below.\n"
        "If the answer is not contained in the context, say you don't know — do not fabricate.\n"
        "Cite the [SOURCE] of any facts you use.\n"
        "\n"
        "<context>\n"
        "{formatted_context}\n"
        "</context>\n"
        "\n"
        "QUESTION: {query}\n"
        "ANSWER:"
    )
