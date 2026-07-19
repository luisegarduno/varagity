"""Unit tests for the multi-turn chat-engine eval (spec_v3 §4.9)."""

import json
from pathlib import Path
from typing import Any

import pytest

from varagity.chat.base import PreparedQuery, Turn
from varagity.chat.simple import SimpleChatEngine
from varagity.eval.datasets import (
    CONVERSATION_KINDS,
    ConversationFixture,
    GoldenEntry,
    load_conversations,
    resolve_golden,
)
from varagity.eval.evaluate import (
    CHAT_EVAL_RETRIEVAL_CONFIGS,
    CHAT_K_VALUES,
    CONVERSATIONS_DIR,
    EVAL_CORPUS,
    PINNED_EVAL_SETTINGS,
    FactRef,
    FactResolvedEntry,
    measure_chat_engine,
)
from varagity.stores.records import RetrievedChunk

VALID_FIXTURE: dict[str, Any] = {
    "name": "corridor",
    "turns": [
        {
            "query": "What is the corridor?",
            "assistant": "It is a strip of kelp.",
            "relevant": [{"rel_source": "a.md", "chunk_index": 0, "fact": "kelp"}],
        },
        {
            "query": "How long is it?",
            "kind": "pronoun",
            "relevant": [{"rel_source": "a.md", "chunk_index": 1, "fact": "1.8"}],
        },
    ],
}


def _write_fixture(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadConversations:
    def test_loads_in_file_name_order(self, tmp_path: Path) -> None:
        second = {**VALID_FIXTURE, "name": "second"}
        _write_fixture(tmp_path, "b-second.json", second)
        _write_fixture(tmp_path, "a-first.json", VALID_FIXTURE)
        loaded = load_conversations(tmp_path)
        assert [fixture.name for fixture in loaded] == ["corridor", "second"]
        assert loaded[0].turns[1].kind == "pronoun"
        assert loaded[0].turns[0].assistant == "It is a strip of kelp."

    def test_follow_up_without_kind_fails(self, tmp_path: Path) -> None:
        bad = json.loads(json.dumps(VALID_FIXTURE))
        del bad["turns"][1]["kind"]
        _write_fixture(tmp_path, "bad.json", bad)
        with pytest.raises(ValueError, match=r"(?s)bad\.json.*needs a kind tag"):
            load_conversations(tmp_path)

    def test_kind_on_first_turn_fails(self, tmp_path: Path) -> None:
        bad = json.loads(json.dumps(VALID_FIXTURE))
        bad["turns"][0]["kind"] = "pronoun"
        _write_fixture(tmp_path, "bad.json", bad)
        with pytest.raises(ValueError, match="standalone by construction"):
            load_conversations(tmp_path)

    def test_missing_assistant_on_non_final_turn_fails(self, tmp_path: Path) -> None:
        bad = json.loads(json.dumps(VALID_FIXTURE))
        del bad["turns"][0]["assistant"]
        _write_fixture(tmp_path, "bad.json", bad)
        with pytest.raises(ValueError, match="scripted assistant reply"):
            load_conversations(tmp_path)

    def test_single_turn_conversation_fails(self, tmp_path: Path) -> None:
        bad = json.loads(json.dumps(VALID_FIXTURE))
        bad["turns"] = bad["turns"][:1]
        _write_fixture(tmp_path, "bad.json", bad)
        with pytest.raises(ValueError, match="invalid conversation fixture"):
            load_conversations(tmp_path)

    def test_unknown_kind_fails(self, tmp_path: Path) -> None:
        bad = json.loads(json.dumps(VALID_FIXTURE))
        bad["turns"][1]["kind"] = "sarcastic"
        _write_fixture(tmp_path, "bad.json", bad)
        with pytest.raises(ValueError, match="invalid conversation fixture"):
            load_conversations(tmp_path)

    def test_duplicate_names_fail(self, tmp_path: Path) -> None:
        _write_fixture(tmp_path, "one.json", VALID_FIXTURE)
        _write_fixture(tmp_path, "two.json", VALID_FIXTURE)
        with pytest.raises(ValueError, match="already used by"):
            load_conversations(tmp_path)

    def test_invalid_json_names_the_file(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match=r"broken\.json: not valid JSON"):
            load_conversations(tmp_path)

    def test_empty_directory_fails(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no conversation fixtures"):
            load_conversations(tmp_path)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_conversations(tmp_path / "nope")


class TestShippedConversationFixtures:
    """The checked-in fixtures stay in sync with the eval corpus."""

    def test_load_and_resolve_against_the_fixtures_corpus(self) -> None:
        conversations = load_conversations(CONVERSATIONS_DIR)
        assert len(conversations) >= 8
        flat = [
            GoldenEntry(query=turn.query, relevant=turn.relevant)
            for conversation in conversations
            for turn in conversation.turns
        ]
        resolved = resolve_golden(flat, EVAL_CORPUS)  # every rel_source exists
        assert len(resolved) == len(flat)

    def test_every_ref_carries_a_fact(self) -> None:
        """The chat eval is fact-anchored; an index-only ref falls back silently."""
        for conversation in load_conversations(CONVERSATIONS_DIR):
            for turn in conversation.turns:
                for ref in turn.relevant:
                    assert ref.fact, (
                        f"{conversation.name!r} turn {turn.query!r} has a "
                        f"fact-less ref ({ref.rel_source})"
                    )

    def test_all_three_kinds_are_exercised(self) -> None:
        kinds = {
            turn.kind
            for conversation in load_conversations(CONVERSATIONS_DIR)
            for turn in conversation.turns
            if turn.kind is not None
        }
        assert kinds == set(CONVERSATION_KINDS)

    def test_sources_are_within_the_pinned_extensions(self) -> None:
        """A ref on a file type the pinned ingest skips can never resolve."""
        allowed = set(PINNED_EVAL_SETTINGS["ALLOWED_EXTENSIONS"].split(","))
        for conversation in load_conversations(CONVERSATIONS_DIR):
            for turn in conversation.turns:
                for ref in turn.relevant:
                    assert Path(ref.rel_source).suffix in allowed, (
                        f"{conversation.name!r} references {ref.rel_source}, which the "
                        f"pinned eval ingest ({allowed}) would skip"
                    )

    def test_text_source_facts_appear_verbatim_in_their_files(self) -> None:
        """Catch fact typos at unit time rather than as eval-time misses.

        Covers the .txt/.md refs only — PDF text extraction is far too slow
        for the unit suite; PDF facts mis-typed surface as loud
        ``unresolved_facts`` warnings at eval time instead.
        """
        texts: dict[str, str] = {}
        for conversation in load_conversations(CONVERSATIONS_DIR):
            for turn in conversation.turns:
                for ref in turn.relevant:
                    if not ref.rel_source.endswith((".txt", ".md")):
                        continue
                    if ref.rel_source not in texts:
                        texts[ref.rel_source] = (
                            (EVAL_CORPUS / ref.rel_source).read_text(encoding="utf-8").lower()
                        )
                    assert ref.fact is not None
                    assert ref.fact.lower() in texts[ref.rel_source], (
                        f"{conversation.name!r}: fact {ref.fact!r} does not appear in "
                        f"{ref.rel_source} — typo?"
                    )


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
        return [_chunk(chunk_id) for chunk_id in self.ranked_by_query.get(query, [])[:k]]


class ScriptedEngine:
    """Rewrites scripted queries; records every history it was handed."""

    def __init__(self, rewrites: dict[str, str]) -> None:
        self.rewrites = rewrites
        self.histories: list[list[Turn]] = []

    def prepare(self, query: str, *, history: Any, llm: Any, verbose: int) -> PreparedQuery:
        self.histories.append(list(history))
        if query in self.rewrites:
            return PreparedQuery(
                search_query=self.rewrites[query],
                original_query=query,
                condensed=True,
                condense_latency_s=0.25,
            )
        return PreparedQuery(
            search_query=query, original_query=query, condensed=False, condense_latency_s=None
        )


CONVERSATION = ConversationFixture.model_validate(VALID_FIXTURE)

# One fact-resolved entry per turn: turn 0 wants doc-a::0, turn 1 doc-a::1.
FACT_ENTRIES = [
    FactResolvedEntry(
        query="What is the corridor?", refs=[FactRef(label="kelp", chunk_ids=["doc-a::0"])]
    ),
    FactResolvedEntry(query="How long is it?", refs=[FactRef(label="1.8", chunk_ids=["doc-a::1"])]),
]

# The raw follow-up finds nothing; only the rewrite ranks the golden chunk.
RANKED = {
    "What is the corridor?": ["doc-a::0", "doc-x::0"],
    "How long is the corridor?": ["doc-a::1", "doc-x::0"],
    "How long is it?": ["doc-x::0", "doc-x::1"],
}


class TestMeasureChatEngine:
    def test_condensing_engine_beats_simple_through_the_harness(self) -> None:
        """Same fixture, same retriever — only the engine differs."""
        condensing = ScriptedEngine({"How long is it?": "How long is the corridor?"})
        condensed = measure_chat_engine(
            condensing,
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1, 5),
        )
        simple = measure_chat_engine(
            SimpleChatEngine(),
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1, 5),
        )
        assert condensed["summary"]["hybrid"]["follow_up"]["recall"]["1"] == 1.0
        assert simple["summary"]["hybrid"]["follow_up"]["recall"]["1"] == 0.0
        # Turn 0 is the identity split under both — the anchor never differs.
        assert condensed["summary"]["hybrid"]["all"]["recall"]["1"] == pytest.approx(1.0)
        assert simple["summary"]["hybrid"]["all"]["recall"]["1"] == pytest.approx(0.5)

    def test_history_threads_the_scripted_replies(self) -> None:
        engine = ScriptedEngine({})
        measure_chat_engine(
            engine,
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1,),
        )
        assert engine.histories[0] == []
        assert engine.histories[1] == [
            Turn(role="user", content="What is the corridor?"),
            Turn(role="assistant", content="It is a strip of kelp."),
        ]

    def test_search_query_drives_retrieval_and_is_recorded(self) -> None:
        retriever = ScriptedRetriever(RANKED)
        engine = ScriptedEngine({"How long is it?": "How long is the corridor?"})
        results = measure_chat_engine(
            engine,
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": retriever},
            llm=None,
            k_values=(1, 5),
        )
        # One retrieve per turn, with the engine's search query, at max(k).
        assert retriever.calls == [
            ("What is the corridor?", 5),
            ("How long is the corridor?", 5),
        ]
        record = results["conversations"][0]["turns"][1]
        assert record["query"] == "How long is it?"
        assert record["search_query"] == "How long is the corridor?"
        assert record["condensed"] is True
        assert record["kind"] == "pronoun"
        assert record["methods"]["hybrid"]["golden_ranks"] == {"1.8": 1}

    def test_by_kind_buckets_only_tagged_turns(self) -> None:
        results = measure_chat_engine(
            ScriptedEngine({}),
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1,),
        )
        by_kind = results["summary"]["hybrid"]["by_kind"]
        assert list(by_kind) == ["pronoun"]
        assert by_kind["pronoun"]["n_turns"] == 1
        assert results["summary"]["hybrid"]["follow_up"]["n_turns"] == 1
        assert results["summary"]["hybrid"]["all"]["n_turns"] == 2

    def test_condense_latency_aggregation(self) -> None:
        condensing = measure_chat_engine(
            ScriptedEngine({"How long is it?": "How long is the corridor?"}),
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1,),
        )
        assert condensing["condense"] == {
            "calls": 1,
            "mean_latency_s": 0.25,
            "max_latency_s": 0.25,
        }
        simple = measure_chat_engine(
            SimpleChatEngine(),
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1,),
        )
        assert simple["condense"] == {"calls": 0, "mean_latency_s": None, "max_latency_s": None}

    def test_unmatched_fact_is_a_guaranteed_miss(self) -> None:
        entries = [
            FACT_ENTRIES[0],
            FactResolvedEntry(query="How long is it?", refs=[FactRef(label="gone", chunk_ids=[])]),
        ]
        results = measure_chat_engine(
            ScriptedEngine({"How long is it?": "How long is the corridor?"}),
            [CONVERSATION],
            entries,
            {"hybrid": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1, 5),
        )
        follow_up = results["summary"]["hybrid"]["follow_up"]
        assert follow_up["recall"]["5"] == 0.0
        assert results["conversations"][0]["turns"][1]["methods"]["hybrid"]["golden_ranks"] == {
            "gone": None
        }

    def test_every_retrieval_config_is_scored(self) -> None:
        results = measure_chat_engine(
            ScriptedEngine({}),
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": ScriptedRetriever(RANKED), "reranked": ScriptedRetriever(RANKED)},
            llm=None,
            k_values=(1,),
        )
        record = results["conversations"][0]["turns"][0]
        assert set(record["methods"]) == {"hybrid", "reranked"}
        assert set(results["summary"]) == {"hybrid", "reranked"}

    def test_entry_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="fact-resolved entries for"):
            measure_chat_engine(
                ScriptedEngine({}),
                [CONVERSATION],
                FACT_ENTRIES[:1],
                {"hybrid": ScriptedRetriever(RANKED)},
                llm=None,
            )

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one conversation"):
            measure_chat_engine(
                ScriptedEngine({}), [], [], {"hybrid": ScriptedRetriever(RANKED)}, llm=None
            )
        with pytest.raises(ValueError, match="at least one retriever"):
            measure_chat_engine(ScriptedEngine({}), [CONVERSATION], FACT_ENTRIES, {}, llm=None)
        with pytest.raises(ValueError, match="at least one k value"):
            measure_chat_engine(
                ScriptedEngine({}),
                [CONVERSATION],
                FACT_ENTRIES,
                {"hybrid": ScriptedRetriever(RANKED)},
                llm=None,
                k_values=(),
            )

    def test_default_depths_are_the_chat_depths(self) -> None:
        retriever = ScriptedRetriever(RANKED)
        results = measure_chat_engine(
            ScriptedEngine({}),
            [CONVERSATION],
            FACT_ENTRIES,
            {"hybrid": retriever},
            llm=None,
        )
        assert retriever.calls[0][1] == max(CHAT_K_VALUES)
        scores = results["summary"]["hybrid"]["all"]
        assert set(scores["recall"]) == {str(k) for k in CHAT_K_VALUES}


class TestChatEvalConfig:
    def test_retrieval_configs_are_a_subset_of_the_sweep_vocabulary(self) -> None:
        """The chat-eval configs must name real registered methods."""
        from varagity.retrieval.base import RETRIEVER_REGISTRY

        assert set(CHAT_EVAL_RETRIEVAL_CONFIGS) <= set(RETRIEVER_REGISTRY)

    def test_chat_depths_are_shallower_than_the_matrix(self) -> None:
        """recall@20 over a ~16-chunk store is 1.0 and discriminates nothing."""
        assert max(CHAT_K_VALUES) <= 5
