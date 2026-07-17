"""Unit tests for generate_answer_stream (the streaming generation seam).

A scripted fake LLM stands in for llama.cpp; the tests cover delta
classification and ordering, the abort seam, usage capture, and the
orphaned-closer reconciliation (streamed display is best-effort, the
returned answer is clean_response-exact).
"""

from collections.abc import Callable, Iterator, Sequence

import pytest

from varagity.generation.answer import ANSWER_PROMPT, generate_answer_stream
from varagity.models.llm import GenerationTimings, clean_response
from varagity.models.stream import Kind
from varagity.stores.records import RetrievedChunk


def make_chunk(index: int = 0, content: str = "Planted fact.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc::{index}",
        doc_id="doc",
        original_index=index,
        content=content,
        context="a situating blurb",
        metadata={"source": "/docs/x.txt", "file_name": "x.txt", "file_type": "txt"},
        score=0.9,
    )


class StreamingFakeLLM:
    """Yields scripted deltas; records prompts and close() on the iterator."""

    def __init__(
        self,
        deltas: Sequence[str],
        usage: dict[str, int] | None = None,
        timings: Sequence[GenerationTimings] | None = None,
    ) -> None:
        self.deltas = list(deltas)
        self.usage = usage
        # One reading reported after each delta, llama.cpp-style (cumulative
        # counters); empty = a server that reports no timings.
        self.timings = list(timings or [])
        self.prompts: list[str] = []
        self.closed = False
        self.yielded: list[str] = []

    def generate_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
        on_usage: Callable[[object], None] | None = None,
        on_timings: Callable[[GenerationTimings], None] | None = None,
    ) -> Iterator[str]:
        self.prompts.append(messages[0]["content"])

        def gen() -> Iterator[str]:
            try:
                for index, delta in enumerate(self.deltas):
                    self.yielded.append(delta)
                    yield delta
                    if on_timings is not None and index < len(self.timings):
                        on_timings(self.timings[index])
                if self.usage is not None and on_usage is not None:

                    class _Usage:
                        prompt_tokens = self.usage["prompt_tokens"]
                        completion_tokens = self.usage["completion_tokens"]

                    on_usage(_Usage())
            finally:
                self.closed = True

        return gen()


def collect(
    llm: StreamingFakeLLM, *, should_abort: Callable[[], bool] | None = None
) -> tuple[list[tuple[Kind, str]], dict]:
    events: list[tuple[Kind, str]] = []
    result = generate_answer_stream(
        "What is planted?",
        [make_chunk()],
        on_delta=lambda kind, text: events.append((kind, text)),
        llm=llm,
        should_abort=should_abort,
        verbose=0,
    )
    return events, dict(result)


def test_answer_deltas_stream_in_order_and_reconcile() -> None:
    llm = StreamingFakeLLM(["The fact ", "is planted. ", "[SOURCE]: x.txt"])
    events, result = collect(llm)
    assert [kind for kind, _ in events] == ["answer", "answer", "answer"]
    assert "".join(text for _, text in events) == "The fact is planted. [SOURCE]: x.txt"
    assert result["answer"] == "The fact is planted. [SOURCE]: x.txt"
    assert result["reasoning"] == ""
    assert result["aborted"] is False


def test_reasoning_classified_and_captured() -> None:
    llm = StreamingFakeLLM(["<think>let me ", "check</think>", "Answer text."])
    events, result = collect(llm)
    assert events == [
        ("reasoning", "let me "),
        ("reasoning", "check"),
        ("answer", "Answer text."),
    ]
    assert result["reasoning"] == "let me check"
    assert result["answer"] == "Answer text."


def test_prompt_is_the_grounding_prompt() -> None:
    llm = StreamingFakeLLM(["ok"])
    collect(llm)
    expected_prefix = ANSWER_PROMPT.split("{formatted_context}")[0]
    assert llm.prompts[0].startswith(expected_prefix)
    assert "Planted fact." in llm.prompts[0]
    assert "What is planted?" in llm.prompts[0]


def test_orphan_closer_answer_is_clean_response_exact() -> None:
    deltas = ["step 1... ", "step 2...</think>", "The answer."]
    llm = StreamingFakeLLM(deltas)
    events, result = collect(llm)
    # Streamed display was best-effort (pre-closer text went out as answer)…
    assert ("answer", "step 1... ") in events
    # …but the persisted answer reconciles to clean_response semantics.
    assert result["answer"] == clean_response("".join(deltas)) == "The answer."


def test_abort_stops_midstream_and_closes_the_llm_stream() -> None:
    llm = StreamingFakeLLM(["a", "b", "c", "d"])
    seen: list[str] = []

    def should_abort() -> bool:
        return len(seen) >= 2

    events, result = collect_with_tracking(llm, seen, should_abort)
    assert result["aborted"] is True
    assert llm.closed
    assert len(llm.yielded) < 4  # generation stopped early, not drained


def collect_with_tracking(
    llm: StreamingFakeLLM, seen: list[str], should_abort: Callable[[], bool]
) -> tuple[list[tuple[Kind, str]], dict]:
    events: list[tuple[Kind, str]] = []

    def on_delta(kind: Kind, text: str) -> None:
        seen.append(text)
        events.append((kind, text))

    result = generate_answer_stream(
        "q",
        [make_chunk()],
        on_delta=on_delta,
        llm=llm,
        should_abort=should_abort,
        verbose=0,
    )
    return events, dict(result)


def test_stream_closed_on_normal_completion() -> None:
    llm = StreamingFakeLLM(["done."])
    collect(llm)
    assert llm.closed


def test_usage_captured_when_reported() -> None:
    llm = StreamingFakeLLM(["x"], usage={"prompt_tokens": 20, "completion_tokens": 5})
    _, result = collect(llm)
    assert result["usage"] == {"prompt_tokens": 20, "completion_tokens": 5}


def test_usage_none_when_unreported() -> None:
    llm = StreamingFakeLLM(["x"])
    _, result = collect(llm)
    assert result["usage"] is None


def test_final_rate_is_the_last_timings_reading() -> None:
    llm = StreamingFakeLLM(
        ["a", "b"],
        timings=[
            GenerationTimings(predicted_n=10, predicted_ms=100.0),
            GenerationTimings(predicted_n=24, predicted_ms=600.0),
        ],
    )
    _, result = collect(llm)
    assert result["tokens_per_second"] == pytest.approx(40.0)


def test_on_stats_forwards_every_reading_in_order() -> None:
    readings = [
        GenerationTimings(predicted_n=10, predicted_ms=100.0),
        GenerationTimings(predicted_n=20, predicted_ms=300.0),
    ]
    llm = StreamingFakeLLM(["a", "b"], timings=readings)
    seen: list[GenerationTimings] = []
    generate_answer_stream(
        "q",
        [make_chunk()],
        on_delta=lambda kind, text: None,
        llm=llm,
        on_stats=seen.append,
        verbose=0,
    )
    assert seen == readings


def test_rate_none_when_the_server_reports_no_timings() -> None:
    llm = StreamingFakeLLM(["x"])
    _, result = collect(llm)
    assert result["tokens_per_second"] is None


def test_unclosed_think_yields_empty_answer() -> None:
    llm = StreamingFakeLLM(["<think>never stops reasoning"])
    events, result = collect(llm)
    assert result["answer"] == ""
    assert result["reasoning"] == "never stops reasoning"
    assert all(kind == "reasoning" for kind, _ in events)


def test_invalid_verbose_rejected() -> None:
    llm = StreamingFakeLLM(["x"])
    with pytest.raises(ValueError, match="verbose"):
        generate_answer_stream("q", [make_chunk()], on_delta=lambda k, t: None, llm=llm, verbose=5)
