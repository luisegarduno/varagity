"""Unit tests for the eval metrics and measurement aggregation (spec §16)."""

import pytest

from varagity.eval.datasets import GoldenChunkRef, ResolvedGoldenEntry
from varagity.eval.evaluate import (
    K_VALUES,
    FactRef,
    FactResolvedEntry,
    measure_retriever,
    measure_retriever_facts,
    pass_at_k,
    pass_at_k_any,
    recall_at_k,
    recall_at_k_any,
    resolve_golden_by_fact,
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


class TestRecallAtKAny:
    def test_any_acceptable_id_satisfies_a_ref(self) -> None:
        acceptable = [["doc-a::0", "doc-a::1"], ["doc-b::4"]]
        assert recall_at_k_any(acceptable, ["doc-a::1", "doc-b::4"], k=2) == 1.0

    def test_unsatisfied_ref_counts_against(self) -> None:
        acceptable = [["doc-a::0"], ["doc-b::4"]]
        assert recall_at_k_any(acceptable, ["doc-a::0", "doc-x::0"], k=2) == 0.5

    def test_empty_acceptable_set_is_a_guaranteed_miss(self) -> None:
        acceptable: list[list[str]] = [["doc-a::0"], []]
        assert recall_at_k_any(acceptable, ["doc-a::0"], k=5) == 0.5

    def test_k_cutoff_applies(self) -> None:
        acceptable = [["doc-a::5"]]
        assert recall_at_k_any(acceptable, ["doc-x::0", "doc-a::5"], k=1) == 0.0
        assert recall_at_k_any(acceptable, ["doc-x::0", "doc-a::5"], k=2) == 1.0

    def test_empty_refs_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one golden ref"):
            recall_at_k_any([], ["doc-a::0"], k=5)

    def test_nonpositive_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k_any([["doc-a::0"]], ["doc-a::0"], k=0)


class TestPassAtKAny:
    def test_all_refs_satisfied_passes(self) -> None:
        acceptable = [["doc-a::0", "doc-a::1"], ["doc-b::4"]]
        assert pass_at_k_any(acceptable, ["doc-a::0", "doc-b::4"], k=2) == 1.0

    def test_one_unsatisfied_ref_fails(self) -> None:
        acceptable = [["doc-a::0"], ["doc-b::4"]]
        assert pass_at_k_any(acceptable, ["doc-a::0", "doc-x::0"], k=2) == 0.0


class _StubStore:
    """document_chunks stub: doc_id → [(chunk_id, content)]."""

    def __init__(self, chunks_by_doc: dict[str, list[tuple[str, str]]]) -> None:
        self.chunks_by_doc = chunks_by_doc
        self.calls: list[str] = []

    def document_chunks(self, doc_id: str) -> list[RetrievedChunk]:
        self.calls.append(doc_id)
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                original_index=index,
                content=content,
                context=None,
                metadata={},
                score=0.0,
            )
            for index, (chunk_id, content) in enumerate(self.chunks_by_doc.get(doc_id, []))
        ]


def _fact_entry(query: str, refs: list[tuple[str, int, str | None]]) -> ResolvedGoldenEntry:
    return ResolvedGoldenEntry(
        query=query,
        relevant=[
            GoldenChunkRef(rel_source=f"{doc}.md", chunk_index=index, fact=fact)
            for doc, index, fact in refs
        ],
        chunk_ids=[f"{doc}::{index}" for doc, index, _ in refs],
    )


class TestResolveGoldenByFact:
    def test_fact_matches_collect_every_containing_chunk(self) -> None:
        store = _StubStore(
            {
                "doc-a": [
                    ("doc-a::0", "The corridor is a 1.8-kilometer strip."),
                    ("doc-a::1", "strip. The 1.8-kilometer corridor dampens turbulence."),
                    ("doc-a::2", "Unrelated tail text."),
                ]
            }
        )
        entries = [_fact_entry("q", [("doc-a", 2, "1.8-kilometer")])]
        resolved = resolve_golden_by_fact(entries, store)  # type: ignore[arg-type]
        assert resolved[0].refs == [
            FactRef(label="1.8-kilometer", chunk_ids=["doc-a::0", "doc-a::1"])
        ]

    def test_matching_is_case_insensitive(self) -> None:
        store = _StubStore({"doc-a": [("doc-a::0", "CLEARED TO NINE METERS ON THE FOURTH")]})
        entries = [_fact_entry("q", [("doc-a", 0, "nine meters")])]
        resolved = resolve_golden_by_fact(entries, store)  # type: ignore[arg-type]
        assert resolved[0].refs[0].chunk_ids == ["doc-a::0"]

    def test_unmatched_fact_yields_empty_set_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        store = _StubStore({"doc-a": [("doc-a::0", "nothing relevant")]})
        entries = [_fact_entry("q", [("doc-a", 0, "purple submarine")])]
        with caplog.at_level(logging.WARNING, logger="varagity.eval.evaluate"):
            resolved = resolve_golden_by_fact(entries, store)  # type: ignore[arg-type]
        assert resolved[0].refs[0].chunk_ids == []
        assert any("purple submarine" in record.getMessage() for record in caplog.records)

    def test_factless_ref_falls_back_to_index_anchoring(self) -> None:
        store = _StubStore({"doc-a": [("doc-a::0", "text")]})
        entries = [_fact_entry("q", [("doc-a", 0, None)])]
        resolved = resolve_golden_by_fact(entries, store)  # type: ignore[arg-type]
        assert resolved[0].refs == [FactRef(label="doc-a::0", chunk_ids=["doc-a::0"])]
        assert store.calls == []  # no scan needed for an index-anchored ref

    def test_document_chunks_fetched_once_per_doc(self) -> None:
        store = _StubStore({"doc-a": [("doc-a::0", "alpha beta")]})
        entries = [_fact_entry("q", [("doc-a", 0, "alpha"), ("doc-a", 0, "beta")])]
        resolve_golden_by_fact(entries, store)  # type: ignore[arg-type]
        assert store.calls == ["doc-a"]


class TestMeasureRetrieverFacts:
    def test_any_of_scoring_and_best_ranks(self) -> None:
        entries = [
            FactResolvedEntry(
                query="q1",
                refs=[FactRef(label="fact-1", chunk_ids=["doc-a::0", "doc-a::1"])],
            ),
            FactResolvedEntry(
                query="q2",
                refs=[
                    FactRef(label="fact-2", chunk_ids=["doc-b::0"]),
                    FactRef(label="gone", chunk_ids=[]),
                ],
            ),
        ]
        retriever = ScriptedRetriever(
            {
                "q1": ["doc-x::0", "doc-a::1", "doc-a::0"],  # best acceptable at rank 2
                "q2": ["doc-b::0"],
            }
        )
        summary, ranks = measure_retriever_facts(retriever, entries, k_values=(1, 5))
        # q1: satisfied at k=5 only; q2: fact-2 at rank 1, "gone" never.
        assert summary["recall"]["1"] == pytest.approx(0.25)  # (0 + 0.5) / 2
        assert summary["recall"]["5"] == pytest.approx(0.75)  # (1 + 0.5) / 2
        assert summary["pass"]["5"] == pytest.approx(0.5)  # q1 passes; the empty ref fails q2
        assert retriever.calls == [("q1", 5), ("q2", 5)]
        assert ranks[0] == {"fact-1": 2}
        assert ranks[1] == {"fact-2": 1, "gone": None}

    def test_empty_entries_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one golden entry"):
            measure_retriever_facts(ScriptedRetriever({}), [])

    def test_empty_k_values_raise(self) -> None:
        entry = FactResolvedEntry(query="q", refs=[FactRef(label="f", chunk_ids=["doc-a::0"])])
        with pytest.raises(ValueError, match="at least one k value"):
            measure_retriever_facts(ScriptedRetriever({}), [entry], k_values=())


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
