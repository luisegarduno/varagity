"""Unit tests for the shared graph answer synthesis (ADR-017 retrieval-only).

The engine finds evidence; this module writes the answer. What matters here
is therefore not an engine at all: that the rendered context is *complete*
(facts and transcript excerpts, decision #6), *bounded* (llama.cpp hard-500s
at the window), and that a synthesis failure is a scored miss rather than an
exception escaping into a harness run or a chat turn.
"""

import logging
from typing import Any

import pytest

from varagity.graph import answer as graph_answer
from varagity.graph.answer import (
    NO_EVIDENCE_ANSWER,
    SYNTHESIS_MAX_TOKENS,
    answer_context_max_chars,
    excerpts_block,
    facts_block,
    graph_answer_context,
    synthesis_context,
    synthesis_max_chars,
    synthesize,
    transcript_label,
)
from varagity.graph.records import (
    GraphCommunity,
    GraphEntity,
    GraphEvidence,
    GraphRelation,
    GraphRetrieval,
    TranscriptExcerpt,
)

THREAD = "iMessage;-;+15125550101"


class ScriptedLLM:
    """Records generate() calls; returns a scripted response or raises."""

    def __init__(self, response: str | Exception = "Bob prefers mechanical keyboards.") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append({"messages": list(messages), **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def evidence(**kwargs: Any) -> GraphEvidence:
    return GraphEvidence(
        entities=[GraphEntity(name="Bob", type="person")],
        relations=[
            GraphRelation(
                source="Bob",
                target="mechanical keyboard",
                label="prefers",
                description="Bob prefers mechanical keyboards",
            )
        ],
        **kwargs,
    )


def excerpt(
    text: str = "[2016-03-04 18:22] Bob: I built the PC", **kwargs: Any
) -> TranscriptExcerpt:
    defaults: dict[str, Any] = {
        "doc_key": f"{THREAD}::2016-03-04",
        "thread_name": "Hardware Talk",
        "span": "2016-03-04",
        "text": text,
    }
    return TranscriptExcerpt(**{**defaults, **kwargs})


def retrieval(
    *, excerpts: list[TranscriptExcerpt] | None = None, mode: str = "mix"
) -> GraphRetrieval:
    return GraphRetrieval(evidence=evidence(), excerpts=excerpts or [], mode=mode)


class TestFactsBlock:
    def test_relations_render_as_fact_lines(self) -> None:
        assert facts_block(evidence()) == "- Bob prefers mechanical keyboards"

    def test_a_relation_falls_back_to_its_label(self) -> None:
        payload = GraphEvidence(relations=[GraphRelation(source="a", target="b", label="knows")])
        assert facts_block(payload) == "- knows"

    def test_community_summaries_join_the_facts(self) -> None:
        payload = GraphEvidence(
            communities=[GraphCommunity(id="c1", title="Hardware", summary="PC talk")]
        )
        assert facts_block(payload) == "- community Hardware: PC talk"

    def test_facts_with_no_content_are_not_offered_to_the_synthesizer(self) -> None:
        payload = GraphEvidence(
            relations=[GraphRelation(source="a", target="b")],
            communities=[GraphCommunity(id="c1", summary="")],
        )
        assert facts_block(payload) == ""

    def test_entities_alone_are_not_facts(self) -> None:
        """Names without edges say nothing — the empty block skips the call."""
        assert facts_block(GraphEvidence(entities=[GraphEntity(name="Bob")])) == ""


class TestExcerptsBlock:
    def test_each_excerpt_is_labelled_with_its_thread_and_span(self) -> None:
        rendered = excerpts_block([excerpt()], max_chars=1000)
        assert rendered.splitlines() == [
            "[Hardware Talk (2016-03-04)]",
            "[2016-03-04 18:22] Bob: I built the PC",
        ]

    def test_excerpts_are_blank_line_separated_in_engine_order(self) -> None:
        rendered = excerpts_block(
            [excerpt(text="first"), excerpt(text="second", span="2016-03-05")], max_chars=1000
        )
        assert rendered.count("\n\n") == 1
        assert rendered.index("first") < rendered.index("second")

    def test_no_excerpts_render_nothing(self) -> None:
        assert excerpts_block([], max_chars=1000) == ""

    def test_packing_drops_from_the_tail_rather_than_overflowing(self) -> None:
        """★ The budget is the window: a dropped tail beats a hard 500."""
        one = excerpts_block([excerpt(text="x" * 100)], max_chars=1000)
        both = excerpts_block(
            [excerpt(text="x" * 100), excerpt(text="y" * 100)], max_chars=len(one) + 10
        )
        assert both == one
        assert "y" not in both

    def test_a_single_oversized_excerpt_is_truncated_not_dropped(self) -> None:
        rendered = excerpts_block([excerpt(text="x" * 5000)], max_chars=200)
        assert len(rendered) <= 200
        assert rendered.startswith("[Hardware Talk (2016-03-04)]")
        assert rendered.endswith("[…]")

    def test_a_budget_too_small_for_a_label_renders_nothing(self) -> None:
        assert excerpts_block([excerpt()], max_chars=5) == ""
        assert excerpts_block([excerpt()], max_chars=0) == ""

    def test_a_budget_smaller_than_the_elision_marker_still_fits(self) -> None:
        """The label plus a sliver of text — never a block over budget."""
        rendered = excerpts_block([excerpt(text="x" * 50)], max_chars=31)
        assert rendered == "[Hardware Talk (2016-03-04)]\nxx"

    def test_an_empty_excerpt_stops_the_packing(self) -> None:
        rendered = excerpts_block([excerpt(text="   \n  "), excerpt(text="real")], max_chars=1000)
        assert rendered == ""


class TestSynthesisContext:
    def test_facts_only_renders_the_fact_headed_section(self) -> None:
        """The fact-shaped engine's context — unchanged in spirit from Graphiti."""
        assert synthesis_context(evidence(), max_chars=1000) == (
            "FACTS:\n- Bob prefers mechanical keyboards"
        )

    def test_facts_and_excerpts_are_both_grounded_on(self) -> None:
        """★ Decision #6: the shipped path sees the whole retrieval payload."""
        rendered = synthesis_context(evidence(), [excerpt()], max_chars=1000)
        assert rendered.startswith("FACTS:\n- Bob prefers mechanical keyboards")
        assert "TRANSCRIPT EXCERPTS:\n[Hardware Talk (2016-03-04)]" in rendered
        assert "I built the PC" in rendered

    def test_excerpts_alone_still_ground_an_answer(self) -> None:
        rendered = synthesis_context(GraphEvidence(), [excerpt()], max_chars=1000)
        assert rendered.startswith("TRANSCRIPT EXCERPTS:")
        assert "FACTS:" not in rendered

    def test_an_empty_retrieval_renders_nothing(self) -> None:
        assert synthesis_context(GraphEvidence(), [], max_chars=1000) == ""

    @pytest.mark.parametrize("max_chars", [40, 120, 400, 4000])
    def test_the_whole_context_honors_the_budget(self, max_chars: int) -> None:
        rendered = synthesis_context(
            evidence(),
            [excerpt(text="x" * 900), excerpt(text="y" * 900, span="2016-03-05")],
            max_chars=max_chars,
        )
        assert len(rendered) <= max_chars

    def test_facts_are_kept_when_the_budget_only_fits_them(self) -> None:
        """Facts are small and name the people — excerpts drop first."""
        rendered = synthesis_context(evidence(), [excerpt(text="x" * 900)], max_chars=45)
        assert rendered.startswith("FACTS:")
        assert "TRANSCRIPT EXCERPTS:" not in rendered

    def test_a_zero_budget_renders_nothing_rather_than_a_bare_separator(self) -> None:
        assert synthesis_context(evidence(), [excerpt()], max_chars=0) == ""


class TestGraphAnswerContext:
    """The streamed path's renderer — same diet, chunk-RAG block shape."""

    def test_transcripts_render_as_citable_source_blocks(self) -> None:
        """★ The label is the citation contract: chips resolve to cards by it."""
        rendered = graph_answer_context(retrieval(excerpts=[excerpt()]), max_chars=1000)
        assert rendered.splitlines() == [
            "GRAPH FACTS (relations extracted from the transcripts below):",
            "- Bob prefers mechanical keyboards",
            "",
            "[SOURCE]:  Hardware Talk (2016-03-04)",
            "[CONTEXT]: ",
            "[CONTENT]: [2016-03-04 18:22] Bob: I built the PC",
        ]

    def test_the_source_label_is_what_the_evidence_cards_are_keyed_by(self) -> None:
        assert transcript_label(excerpt()) == "Hardware Talk (2016-03-04)"

    def test_the_facts_block_is_not_itself_citable(self) -> None:
        """A [SOURCE] there would produce chips with no card behind them."""
        rendered = graph_answer_context(retrieval(), max_chars=1000)
        assert "[SOURCE]" not in rendered

    def test_transcripts_alone_still_ground_an_answer(self) -> None:
        rendered = graph_answer_context(
            GraphRetrieval(evidence=GraphEvidence(), excerpts=[excerpt()], mode="mix"),
            max_chars=1000,
        )
        assert rendered.startswith("[SOURCE]:")

    def test_an_empty_retrieval_renders_nothing(self) -> None:
        """Which the grounding prompt turns into "I don't know", not a fabrication."""
        empty = GraphRetrieval(evidence=GraphEvidence(), excerpts=[], mode="mix")
        assert graph_answer_context(empty, max_chars=1000) == ""

    def test_facts_are_kept_when_the_budget_only_fits_them(self) -> None:
        rendered = graph_answer_context(retrieval(excerpts=[excerpt(text="x" * 900)]), max_chars=95)
        assert rendered.startswith("GRAPH FACTS")
        assert "[SOURCE]" not in rendered

    @pytest.mark.parametrize("max_chars", [40, 120, 400, 4000])
    def test_the_whole_context_honors_the_budget(self, max_chars: int) -> None:
        rendered = graph_answer_context(
            retrieval(
                excerpts=[
                    excerpt(text="x" * 900),
                    excerpt(text="y" * 900, span="2016-03-05"),
                ]
            ),
            max_chars=max_chars,
        )
        assert len(rendered) <= max_chars

    def test_it_grounds_on_exactly_what_the_harness_measures(self) -> None:
        """★ Same facts, same passages — only the block shape differs."""
        payload = retrieval(excerpts=[excerpt(), excerpt(text="second", span="2016-03-05")])
        streamed = graph_answer_context(payload, max_chars=4000)
        scored = synthesis_context(payload.evidence, payload.excerpts, max_chars=4000)
        for fragment in ("Bob prefers mechanical keyboards", "I built the PC", "second"):
            assert fragment in streamed
            assert fragment in scored


class TestContextBudgets:
    def test_the_budget_leaves_room_for_the_answer_and_the_prompt(self) -> None:
        assert synthesis_max_chars(16384) == (16384 - SYNTHESIS_MAX_TOKENS - 1024) * 3

    def test_a_tiny_window_still_gets_a_floor(self) -> None:
        assert synthesis_max_chars(1024) == 1000
        assert synthesis_max_chars(0) == 1000

    def test_the_streamed_budget_reserves_the_real_generation_cap(self) -> None:
        """MAX_TOKENS, not the synthesis cap — the streamed answer may be long."""
        assert answer_context_max_chars(16384, 8192) == (16384 - 8192 - 1024) * 3
        assert answer_context_max_chars(16384, 16384) == 1000


class TestSynthesize:
    def test_the_prompt_carries_the_context_and_the_verbatim_question(self) -> None:
        llm = ScriptedLLM()
        result = synthesize(llm, "What does Bob think about computers?", "FACTS:\n- he likes ARM")
        assert result == "Bob prefers mechanical keyboards."
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "FACTS:\n- he likes ARM" in prompt
        assert "What does Bob think about computers?" in prompt
        assert llm.calls[0]["max_tokens"] == SYNTHESIS_MAX_TOKENS

    def test_an_empty_context_answers_honestly_without_calling_the_model(self) -> None:
        llm = ScriptedLLM()
        assert synthesize(llm, "q", "   \n ") == NO_EVIDENCE_ANSWER
        assert llm.calls == []

    def test_a_failed_synthesis_answers_empty_rather_than_raising(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = ScriptedLLM(RuntimeError("model down"))
        with caplog.at_level(logging.WARNING, logger="varagity.graph.answer"):
            assert synthesize(llm, "q", "FACTS:\n- something") == ""
        assert "synthesis failed" in caplog.text

    def test_a_reasoning_stage_never_reaches_the_answer(self) -> None:
        """★ The same trap as condense/HyDE: generate() does not strip <think>."""
        llm = ScriptedLLM("<think>weighing facts</think>Bob likes ARM.")
        assert synthesize(llm, "q", "FACTS:\n- x") == "Bob likes ARM."

    def test_the_prompt_forbids_inventing_evidence(self) -> None:
        assert "Use ONLY the evidence below" in graph_answer.SYNTHESIS_PROMPT
