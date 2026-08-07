"""Unit tests for the graph query flow (spec_graphrag §4.3).

The flow runs for real under ``prefect_test_harness`` with the graph service
and the model client faked, because what is under test is the *composition*:
that a graph turn reuses the chunk path's condense and generation stages
verbatim (so the search/answer query split, the ``[SOURCE]`` contract, and
abort handling are inherited rather than re-implemented), that the evidence
reaches the caller before the first token, and that the answer is grounded in
the transcript days the graph actually cited.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from prefect.cache_policies import NO_CACHE
from prefect.testing.utilities import prefect_test_harness

from varagity.chat.base import PreparedQuery, Turn
from varagity.graph.records import (
    GraphEntity,
    GraphEvidence,
    GraphRelation,
    GraphRetrieval,
    TranscriptExcerpt,
)
from varagity.models.llm import GenerationTimings
from varagity.pipeline import graph_query_stream_flow
from varagity.pipeline.graph_flow import graph_retrieve_task

THREAD = "iMessage;-;+15125550101"
QUESTION = "What does Bob think about computers?"


@pytest.fixture(scope="module", autouse=True)
def prefect_harness() -> Iterator[None]:
    """Run every test in this module against an ephemeral Prefect API."""
    with prefect_test_harness():
        yield


@pytest.fixture(autouse=True)
def pinned_settings(settings_env: Callable[..., None]) -> None:
    """Hermetic query-path settings (no machine ``.env`` leakage)."""
    settings_env(
        CHAT_ENGINE="simple",
        GRAPH_QUERY_MODE="mix",
        LLM_CONTEXT_TOKENS=16384,
        MAX_TOKENS=2048,
        DEFAULT_VERBOSE=0,
    )


def excerpt(text: str = "[2016-03-04 18:22] Bob: I built the PC", **kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "doc_key": f"{THREAD}::2016-03-04",
        "thread_name": "Hardware Talk",
        "span": "2016-03-04",
        "text": text,
        "message_guids": ["g1", "g2"],
    }
    return TranscriptExcerpt(**{**defaults, **kwargs})


def retrieval(**kwargs: Any) -> GraphRetrieval:
    defaults: dict[str, Any] = {
        "evidence": GraphEvidence(
            entities=[GraphEntity(name="Bob", type="person")],
            relations=[
                GraphRelation(
                    source="Bob",
                    target="mechanical keyboard",
                    label="prefers",
                    description="Bob prefers mechanical keyboards",
                )
            ],
            message_guids=["g1", "g2"],
        ),
        "excerpts": [excerpt()],
        "mode": "mix",
    }
    return GraphRetrieval(**{**defaults, **kwargs})


class FakeService:
    """Records the retrievals it was asked for; returns a scripted one."""

    def __init__(self, payload: GraphRetrieval | None = None) -> None:
        self.payload = payload if payload is not None else retrieval()
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(
        self, question: str, *, mode: str | None = None, verbose: int = 0
    ) -> GraphRetrieval:
        self.calls.append((question, mode))
        return self.payload


class FakeLLM:
    """Streams scripted deltas; keeps every prompt it was handed."""

    def __init__(self, deltas: Sequence[str] | None = None) -> None:
        self.deltas = list(deltas or ["<think>weighing</think>Bob ", "built a PC."])
        self.prompts: list[str] = []

    def generate_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
        on_usage: Callable[[Any], None] | None = None,
        on_timings: Callable[[GenerationTimings], None] | None = None,
    ) -> Iterator[str]:
        self.prompts.append(messages[0]["content"])

        def gen() -> Iterator[str]:
            yield from self.deltas
            if on_usage is not None:
                on_usage(type("_Usage", (), {"prompt_tokens": 11, "completion_tokens": 7})())

        return gen()

    def generate(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        raise AssertionError("the streamed path must never call generate()")

    @property
    def prompt(self) -> str:
        """The first prompt the flow sent."""
        return self.prompts[0]


class CondensingEngine:
    """A chat engine double that always rewrites (the condense_context shape)."""

    def __init__(self, rewritten: str = "what does Bob think about computers") -> None:
        self.rewritten = rewritten
        self.history: list[Sequence[Turn]] = []

    def prepare(
        self, query: str, *, history: Sequence[Turn], llm: Any, verbose: int
    ) -> PreparedQuery:
        self.history.append(list(history))
        return PreparedQuery(
            search_query=self.rewritten,
            original_query=query,
            condensed=True,
            condense_latency_s=0.1,
        )


def run_flow(
    service: FakeService,
    llm: FakeLLM,
    *,
    query: str = QUESTION,
    engine: Any = None,
    deltas: list[tuple[str, str]] | None = None,
    seen: list[GraphRetrieval] | None = None,
    **kwargs: Any,
) -> Any:
    return graph_query_stream_flow(
        query,
        service,  # type: ignore[arg-type]
        engine=engine,
        llm=llm,  # type: ignore[arg-type]
        on_retrieved=None if seen is None else seen.append,
        on_delta=(lambda kind, text: None)
        if deltas is None
        else (lambda kind, text: deltas.append((kind, text))),
        **kwargs,
    )


class TestStaging:
    def test_evidence_reaches_the_caller_before_the_first_token(self) -> None:
        """★ The SSE protocol's rule, enforced where the callbacks fire."""
        order: list[str] = []
        service, llm = FakeService(), FakeLLM()
        graph_query_stream_flow(
            QUESTION,
            service,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            on_retrieved=lambda payload: order.append("retrieval"),
            on_delta=lambda kind, text: order.append(kind),
        )
        assert order[0] == "retrieval"
        assert "answer" in order
        assert order.index("retrieval") < order.index("answer")

    def test_the_state_carries_what_the_turn_persists(self) -> None:
        service, llm = FakeService(), FakeLLM()
        state = run_flow(service, llm)
        assert state["answer"] == "Bob built a PC."
        assert state["reasoning"] == "weighing"
        assert state["aborted"] is False
        assert state["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
        assert state["retrieval"] is service.payload
        assert state["query"] == QUESTION
        assert state["prepared"].original_query == QUESTION

    def test_the_retrieval_hook_gets_the_payload_the_state_keeps(self) -> None:
        seen: list[GraphRetrieval] = []
        service = FakeService()
        run_flow(service, FakeLLM(), seen=seen)
        assert seen == [service.payload]

    def test_a_client_abort_stops_generation_and_marks_the_turn(self) -> None:
        state = graph_query_stream_flow(
            QUESTION,
            FakeService(),  # type: ignore[arg-type]
            llm=FakeLLM(),  # type: ignore[arg-type]
            on_delta=lambda kind, text: None,
            should_abort=lambda: True,
        )
        assert state["aborted"] is True


class TestGrounding:
    def test_the_prompt_cites_the_transcript_days_the_graph_returned(self) -> None:
        """★ The citation contract: the label is what a chip resolves to."""
        llm = FakeLLM()
        run_flow(FakeService(), llm)
        assert "[SOURCE]:  Hardware Talk (2016-03-04)" in llm.prompt
        assert "I built the PC" in llm.prompt
        assert "- Bob prefers mechanical keyboards" in llm.prompt

    def test_the_answer_prompt_gets_the_users_own_words(self) -> None:
        """★ spec_v3 §4.2: the condensed form searches, the original answers."""
        engine = CondensingEngine()
        llm = FakeLLM()
        service = FakeService()
        state = run_flow(service, llm, engine=engine)
        assert service.calls == [("what does Bob think about computers", "mix")]
        assert QUESTION in llm.prompt
        assert "what does Bob think about computers" not in llm.prompt
        assert state["prepared"].condensed is True

    def test_history_reaches_the_chat_engine(self) -> None:
        engine = CondensingEngine()
        run_flow(
            FakeService(),
            FakeLLM(),
            engine=engine,
            history=[Turn(role="user", content="who is Bob?")],
        )
        assert [turn.content for turn in engine.history[0]] == ["who is Bob?"]

    def test_an_empty_retrieval_still_answers_from_an_empty_context(self) -> None:
        """No evidence is a "don't know", not a crash and not a fabrication."""
        service = FakeService(GraphRetrieval(evidence=GraphEvidence(), mode="mix"))
        llm = FakeLLM()
        state = run_flow(service, llm)
        assert state["formatted_context"] == ""
        assert "<context>\n\n</context>" in llm.prompt

    def test_the_context_is_bounded_by_the_window(self) -> None:
        """A transcript day is thousands of characters; the window is finite."""
        service = FakeService(retrieval(excerpts=[excerpt(text="x" * 200_000)]))
        state = run_flow(service, FakeLLM())
        # LLM_CONTEXT_TOKENS 16384 - MAX_TOKENS 2048 - 1024 headroom, ×3.
        assert len(state["formatted_context"]) <= (16384 - 2048 - 1024) * 3


class TestModeSelection:
    def test_the_mode_defaults_to_the_setting(self, settings_env: Callable[..., None]) -> None:
        settings_env(GRAPH_QUERY_MODE="global")
        service = FakeService()
        run_flow(service, FakeLLM())
        assert service.calls == [(QUESTION, "global")]

    def test_an_explicit_mode_wins(self) -> None:
        service = FakeService()
        run_flow(service, FakeLLM(), mode="local")
        assert service.calls == [(QUESTION, "local")]


class TestTracking:
    def test_the_flow_and_its_retrieval_stage_are_named_runs(self) -> None:
        assert graph_query_stream_flow.name == "graph-query-stream"
        assert graph_retrieve_task.name == "graph_retrieve"

    def test_the_query_path_carries_no_retries_and_no_caching(self) -> None:
        """★ Interactive: the clients retry HTTP themselves, and inputs are live."""
        assert not graph_retrieve_task.retries
        assert graph_retrieve_task.cache_policy is NO_CACHE
        assert graph_query_stream_flow.retries in (0, None)
