"""Unit tests for the eval metrics and measurement aggregation (spec §16)."""

import pytest

from varagity.eval.datasets import GoldenChunkRef, ResolvedGoldenEntry
from varagity.eval.evaluate import (
    K_VALUES,
    measure_retriever,
    pass_at_k,
    recall_at_k,
)
from varagity.eval.ocr_benchmark import (
    character_error_rate,
    normalize_ocr_text,
    word_error_rate,
)
from varagity.stores.records import RetrievedChunk

GOLDEN = ["doc-a::0", "doc-a::1"]


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(GOLDEN, ["doc-a::0", "doc-a::1", "doc-b::0"], k=3) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(GOLDEN, ["doc-a::0", "doc-b::0", "doc-b::1"], k=3) == 0.5

    def test_zero_recall(self) -> None:
        assert recall_at_k(GOLDEN, ["doc-b::0", "doc-b::1"], k=2) == 0.0

    def test_k_cutoff_excludes_deeper_hits(self) -> None:
        retrieved = ["doc-b::0", "doc-b::1", "doc-b::2", "doc-b::3", "doc-b::4", "doc-a::0"]
        assert recall_at_k(["doc-a::0"], retrieved, k=5) == 0.0
        assert recall_at_k(["doc-a::0"], retrieved, k=6) == 1.0

    def test_k_beyond_retrieved_length_is_fine(self) -> None:
        assert recall_at_k(["doc-a::0"], ["doc-a::0"], k=20) == 1.0

    def test_empty_golden_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one golden"):
            recall_at_k([], ["doc-a::0"], k=5)

    def test_nonpositive_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(GOLDEN, ["doc-a::0"], k=0)

    def test_empty_retrieved_is_zero(self) -> None:
        assert recall_at_k(GOLDEN, [], k=5) == 0.0


class TestPassAtK:
    def test_all_found_passes(self) -> None:
        assert pass_at_k(GOLDEN, ["doc-a::1", "doc-a::0"], k=2) == 1.0

    def test_partial_fails(self) -> None:
        """pass@k is strict: one missing golden chunk fails the query."""
        assert pass_at_k(GOLDEN, ["doc-a::0", "doc-b::0"], k=2) == 0.0

    def test_zero_fails(self) -> None:
        assert pass_at_k(GOLDEN, ["doc-b::0"], k=1) == 0.0


def _entry(query: str, chunk_ids: list[str]) -> ResolvedGoldenEntry:
    refs = [
        GoldenChunkRef(
            rel_source=chunk_id.split("::")[0] + ".md",
            chunk_index=int(chunk_id.split("::")[1]),
        )
        for chunk_id in chunk_ids
    ]
    return ResolvedGoldenEntry(query=query, relevant=refs, chunk_ids=chunk_ids)


def _chunk(chunk_id: str) -> RetrievedChunk:
    doc_id, index = chunk_id.split("::")
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        original_index=int(index),
        content=f"content of {chunk_id}",
        context=None,
        metadata={},
        score=1.0,
    )


class ScriptedRetriever:
    """Returns a pre-scripted ranked list per query; records the calls."""

    def __init__(self, ranked_by_query: dict[str, list[str]]) -> None:
        self.ranked_by_query = ranked_by_query
        self.calls: list[tuple[str, int]] = []

    def encode_query(self, query: str, verbose: int | None = None) -> None:
        return None

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return [_chunk(chunk_id) for chunk_id in self.ranked_by_query[query][:k]]


class TestMeasureRetriever:
    def test_averages_and_ranks(self) -> None:
        entries = [
            _entry("q1", ["doc-a::0"]),  # found at rank 1 everywhere
            _entry("q2", ["doc-b::0", "doc-b::1"]),  # b::1 found only at rank 6
        ]
        retriever = ScriptedRetriever(
            {
                "q1": ["doc-a::0", "doc-x::0"],
                "q2": ["doc-b::0", "doc-x::0", "doc-x::1", "doc-x::2", "doc-x::3", "doc-b::1"],
            }
        )

        summary, ranks = measure_retriever(retriever, entries, k_values=(5, 10))

        # q1: recall 1.0 at both depths; q2: 0.5 at k=5 (b::1 at rank 6), 1.0 at k=10.
        assert summary["recall"]["5"] == pytest.approx(0.75)
        assert summary["recall"]["10"] == pytest.approx(1.0)
        # pass@k is strict: q2 fails at k=5, passes at k=10.
        assert summary["pass"]["5"] == pytest.approx(0.5)
        assert summary["pass"]["10"] == pytest.approx(1.0)
        # One retrieve per query, at max(k_values).
        assert retriever.calls == [("q1", 10), ("q2", 10)]
        # Golden ranks are 1-based, None when absent.
        assert ranks[0] == {"doc-a::0": 1}
        assert ranks[1] == {"doc-b::0": 1, "doc-b::1": 6}

    def test_absent_golden_rank_is_none(self) -> None:
        entries = [_entry("q", ["doc-a::0", "doc-a::1"])]
        retriever = ScriptedRetriever({"q": ["doc-a::0"]})
        summary, ranks = measure_retriever(retriever, entries, k_values=(5,))
        assert summary["recall"]["5"] == pytest.approx(0.5)
        assert ranks[0] == {"doc-a::0": 1, "doc-a::1": None}

    def test_default_k_values_are_the_spec_depths(self) -> None:
        entries = [_entry("q", ["doc-a::0"])]
        retriever = ScriptedRetriever({"q": ["doc-a::0"]})
        summary, _ = measure_retriever(retriever, entries)
        assert set(summary["recall"]) == {str(k) for k in K_VALUES}
        assert retriever.calls == [("q", max(K_VALUES))]

    def test_empty_entries_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one golden entry"):
            measure_retriever(ScriptedRetriever({}), [])

    def test_empty_k_values_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one k value"):
            measure_retriever(ScriptedRetriever({}), [_entry("q", ["doc-a::0"])], k_values=())


class TestNormalizeOcrText:
    def test_lowercases_and_collapses_whitespace(self) -> None:
        assert normalize_ocr_text("THE  Dredger\n\nMoorhen") == "the dredger moorhen"

    def test_strips_markdown_and_punctuation(self) -> None:
        assert normalize_ocr_text("## DIVE TEAM NOTE\n\n| cell | 4.6 |") == "dive team note cell 46"

    def test_empty_input(self) -> None:
        assert normalize_ocr_text("  \n ") == ""


class TestErrorRates:
    def test_identical_text_scores_zero(self) -> None:
        text = "The dredger Moorhen cleared the channel."
        assert word_error_rate(text, text) == 0.0
        assert character_error_rate(text, text) == 0.0

    def test_formatting_differences_score_zero(self) -> None:
        """Case, markdown markup, and spacing must not count as errors."""
        truth = "Dive Team Note.\nNo scour damage was found."
        hypothesis = "## DIVE TEAM NOTE\n\nNO SCOUR  DAMAGE WAS FOUND"
        assert word_error_rate(truth, hypothesis) == 0.0
        assert character_error_rate(truth, hypothesis) == 0.0

    def test_known_single_word_substitution(self) -> None:
        # 1 substitution over 4 reference words → WER 0.25.
        assert word_error_rate("the cat sat down", "the dog sat down") == pytest.approx(0.25)

    def test_cer_counts_character_edits(self) -> None:
        # "abcd" → "abed": 1 substitution over 4 reference chars.
        assert character_error_rate("abcd", "abed") == pytest.approx(0.25)

    def test_empty_truth_raises(self) -> None:
        with pytest.raises(ValueError, match="ground truth is empty"):
            word_error_rate("##  ", "anything")
        with pytest.raises(ValueError, match="ground truth is empty"):
            character_error_rate("", "anything")
