"""Unit tests for ThinkStreamSplitter — the streaming twin of clean_response.

Covers the same shapes TestCleanResponse covers (plain, balanced blocks,
multiple blocks, unclosed-at-EOF, orphaned closer) plus the streaming-only
concerns: tags split across token boundaries and hold-back flushing.
"""

from varagity.models.llm import clean_response
from varagity.models.stream import Kind, ThinkStreamSplitter


def run_splitter(deltas: list[str]) -> tuple[list[tuple[Kind, str]], ThinkStreamSplitter]:
    splitter = ThinkStreamSplitter()
    events: list[tuple[Kind, str]] = []
    for delta in deltas:
        events.extend(splitter.feed(delta))
    events.extend(splitter.finalize())
    return events, splitter


def answer_text(events: list[tuple[Kind, str]]) -> str:
    return "".join(text for kind, text in events if kind == "answer")


def reasoning_text(events: list[tuple[Kind, str]]) -> str:
    return "".join(text for kind, text in events if kind == "reasoning")


class TestCleanResponseParity:
    """One-shot feeds must partition exactly as clean_response strips."""

    def test_plain_response_all_answer(self) -> None:
        events, _ = run_splitter(["The answer is 42."])
        assert events == [("answer", "The answer is 42.")]

    def test_leading_think_block_split(self) -> None:
        raw = "<think>hmm, let me see</think>\n\nThe answer."
        events, _ = run_splitter([raw])
        assert reasoning_text(events) == "hmm, let me see"
        assert answer_text(events).strip() == clean_response(raw)

    def test_multiple_think_blocks(self) -> None:
        raw = "<think>a</think>First. <think>b</think>Second."
        events, _ = run_splitter([raw])
        assert reasoning_text(events) == "ab"
        assert answer_text(events) == "First. Second."
        assert answer_text(events) == clean_response(raw)

    def test_unclosed_think_dropped_from_answer(self) -> None:
        raw = "Partial. <think>still reasoning about"
        events, _ = run_splitter([raw])
        assert answer_text(events) == "Partial. "
        assert answer_text(events).strip() == clean_response(raw)
        assert reasoning_text(events) == "still reasoning about"

    def test_think_only_response_no_answer(self) -> None:
        events, _ = run_splitter(["<think>nothing but reasoning</think>"])
        assert answer_text(events) == ""
        assert reasoning_text(events) == "nothing but reasoning"

    def test_nested_open_tag_is_inert_reasoning_text(self) -> None:
        # clean_response's non-greedy regex: the first closer closes.
        raw = "<think>a <think> b</think>after"
        events, _ = run_splitter([raw])
        assert reasoning_text(events) == "a <think> b"
        assert answer_text(events) == "after"
        assert answer_text(events) == clean_response(raw)


class TestOrphanedCloser:
    def test_orphan_flagged_and_tag_dropped(self) -> None:
        raw = "step 1... step 2...</think>The answer."
        events, splitter = run_splitter([raw])
        assert splitter.saw_orphan_closer
        # Streaming can't reclassify already-emitted text; the flag tells
        # callers to reconcile via clean_response over the raw accumulation.
        assert answer_text(events) == "step 1... step 2...The answer."
        assert clean_response(raw) == "The answer."

    def test_no_orphan_flag_on_balanced_blocks(self) -> None:
        _, splitter = run_splitter(["<think>a</think>fine"])
        assert not splitter.saw_orphan_closer


class TestTokenBoundaries:
    def test_open_tag_split_across_deltas(self) -> None:
        events, _ = run_splitter(["<th", "ink>reason", "ing</think>Answer"])
        assert reasoning_text(events) == "reasoning"
        assert answer_text(events) == "Answer"

    def test_close_tag_split_across_deltas(self) -> None:
        events, _ = run_splitter(["<think>r</th", "ink>A"])
        assert reasoning_text(events) == "r"
        assert answer_text(events) == "A"

    def test_single_char_deltas(self) -> None:
        raw = "<think>ab</think>cd"
        events, _ = run_splitter(list(raw))
        assert reasoning_text(events) == "ab"
        assert answer_text(events) == "cd"

    def test_lone_angle_bracket_not_a_tag(self) -> None:
        events, _ = run_splitter(["a < b", " and c"])
        assert answer_text(events) == "a < b and c"

    def test_angle_prefix_that_never_completes_flushes_at_finalize(self) -> None:
        events, _ = run_splitter(["answer <thin"])
        # "<thin" is held back as a possible tag start until finalize.
        assert answer_text(events) == "answer <thin"

    def test_non_tag_angle_text_emitted_promptly(self) -> None:
        splitter = ThinkStreamSplitter()
        events = splitter.feed("value <x> more")
        assert ("answer", "value <x> more") in events

    def test_tag_chars_never_leak_into_output(self) -> None:
        events, _ = run_splitter(["<think>", "r", "</think>", "a"])
        assert "<" not in answer_text(events) + reasoning_text(events)


class TestStreamingBehavior:
    def test_answer_before_think_streams_immediately(self) -> None:
        splitter = ThinkStreamSplitter()
        assert splitter.feed("Sure! ") == [("answer", "Sure! ")]

    def test_reasoning_streams_without_waiting_for_closer(self) -> None:
        splitter = ThinkStreamSplitter()
        splitter.feed("<think>")
        assert splitter.feed("step one, ") == [("reasoning", "step one, ")]

    def test_empty_delta_is_harmless(self) -> None:
        splitter = ThinkStreamSplitter()
        assert splitter.feed("") == []
        splitter.feed("held <")
        assert splitter.feed("") == []  # hold-back must not double-emit

    def test_finalize_twice_returns_nothing_second_time(self) -> None:
        splitter = ThinkStreamSplitter()
        splitter.feed("text <th")
        assert splitter.finalize() != []
        assert splitter.finalize() == []
