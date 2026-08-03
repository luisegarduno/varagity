"""Unit tests for the graph-engine seam and the three bake-off adapters.

Not one engine library is installed here — that is the point. The registry
must be free (``import varagity.graph.engines`` pulls no engine), the pure
rendering must be exact (it is where most adapter correctness lives), the
evidence normalizers must map canned engine payloads and degrade rather than
raise on shapes they have never seen, and the session classes must be
drivable by doubles so the only untested surface left is the lazy-import
block inside each ``session()``.
"""

import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from varagity.graph.base import (
    GRAPH_ENGINE_REGISTRY,
    GraphEngine,
    GraphSession,
    get_graph_engine,
    register,
)
from varagity.graph.engines import cognee as cognee_adapter
from varagity.graph.engines import graphiti as graphiti_adapter
from varagity.graph.engines import lightrag as lightrag_adapter
from varagity.graph.records import BuildReport, GraphAnswer, GraphEvidence, GraphStats
from varagity.graph.render import (
    doc_guid_index,
    episode_payloads,
    guids_in_payload,
    merge_batches,
    thread_transcripts,
)
from varagity.graph.sources.base import MessageBatch, SourceMessage, Tapback

THREAD = "iMessage;-;+15125550101"
START = datetime(2016, 3, 4, 18, 22, tzinfo=UTC)


def message(
    guid: str,
    *,
    text: str = "hello",
    sender: str = "Bob",
    when: datetime = START,
    thread_id: str = THREAD,
    thread_name: str = "Hardware Talk",
    is_from_me: bool = False,
    tapbacks: Sequence[Tapback] = (),
) -> SourceMessage:
    return SourceMessage(
        guid=guid,
        thread_id=thread_id,
        thread_name=thread_name,
        sender_handle="" if is_from_me else "+15125550101",
        sender_name=sender,
        is_from_me=is_from_me,
        timestamp=when,
        text=text,
        tapbacks=list(tapbacks),
    )


def batch(*messages: SourceMessage, doc_id: str = "doc-1", path: str = "chat.db") -> MessageBatch:
    return MessageBatch(doc_id=doc_id, relative_path=path, messages=list(messages))


class TestRegistry:
    def test_all_three_bakeoff_seats_are_registered(self) -> None:
        assert set(GRAPH_ENGINE_REGISTRY) == {"lightrag", "cognee", "graphiti"}

    def test_get_graph_engine_returns_the_registered_instance(self) -> None:
        assert get_graph_engine("lightrag") is GRAPH_ENGINE_REGISTRY["lightrag"]

    def test_unknown_engine_raises_keyerror_listing_available(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_graph_engine("made_up")
        message_text = str(excinfo.value)
        assert "made_up" in message_text
        for name in ("lightrag", "cognee", "graphiti"):
            assert name in message_text

    def test_register_adds_an_instance_and_returns_the_class(self) -> None:
        @register("_test_probe")
        class ProbeEngine:
            def session(self, workdir: Path) -> Any:
                raise NotImplementedError

        try:
            assert isinstance(GRAPH_ENGINE_REGISTRY["_test_probe"], ProbeEngine)
            assert ProbeEngine.__name__ == "ProbeEngine"  # returned unchanged
        finally:
            del GRAPH_ENGINE_REGISTRY["_test_probe"]  # keep the registry pristine

    def test_adapters_satisfy_the_runtime_checkable_protocol(self) -> None:
        """Prefect's parameter-schema machinery needs isinstance to work."""
        for engine in GRAPH_ENGINE_REGISTRY.values():
            assert isinstance(engine, GraphEngine)

    def test_importing_the_registry_imports_no_engine_library(self) -> None:
        """★ The guard that keeps CI engine-free (plan decision #8).

        This module has already imported every adapter; if any of them
        touched its library at module level, the library would be in
        sys.modules by now.
        """
        import varagity.graph.engines  # noqa: F401  (the import under test)

        for library in ("lightrag", "cognee", "graphiti_core", "falkordb", "redislite"):
            assert library not in sys.modules


class TestMergeBatches:
    def test_overlapping_batches_yield_each_message_once(self) -> None:
        """★ The amendment's regression guard: a re-export is a superset, not a second copy."""
        older = batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))
        newer = batch(
            message("g1"),
            message("g2", when=START + timedelta(minutes=1)),
            message("g3", when=START + timedelta(minutes=2)),
            doc_id="doc-2",
            path="chat-2026.db",
        )
        merged = merge_batches([older, newer])
        assert [m.guid for m in merged] == ["g1", "g2", "g3"]

    def test_first_occurrence_wins(self) -> None:
        first = batch(message("g1", text="original"))
        second = batch(message("g1", text="edited"), doc_id="doc-2")
        assert merge_batches([first, second])[0].text == "original"

    def test_the_result_is_sorted_by_timestamp_then_guid(self) -> None:
        late = message("b", when=START + timedelta(hours=1))
        early_z = message("z", when=START)
        early_a = message("a", when=START)
        assert [m.guid for m in merge_batches([batch(late, early_z, early_a)])] == ["a", "z", "b"]

    def test_merging_nothing_is_empty(self) -> None:
        assert merge_batches([]) == []

    def test_duplicate_drops_are_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="varagity.graph.render"):
            merge_batches([batch(message("g1")), batch(message("g1"), doc_id="doc-2")])
        assert "duplicate guid" in caplog.text


class TestThreadTranscripts:
    def test_one_thread_day_renders_a_header_and_stamped_lines(self) -> None:
        docs = thread_transcripts(
            [
                message("g1", text="I built the PC"),
                message(
                    "g2",
                    text="nice",
                    sender="Me",
                    is_from_me=True,
                    when=START + timedelta(minutes=3),
                ),
            ]
        )
        (doc,) = docs
        assert doc.text.splitlines() == [
            "Thread: Hardware Talk (participants: Bob, Me)",
            "",
            "[2016-03-04 18:22] Bob: I built the PC",
            "[2016-03-04 18:25] Me: nice",
        ]
        assert doc.thread_id == THREAD
        assert doc.thread_name == "Hardware Talk"
        assert doc.message_guids == ["g1", "g2"]

    def test_the_doc_key_is_thread_plus_day_span_and_carries_no_doc_id(self) -> None:
        """★ The amendment: the same thread-days always render to the same key."""
        one_day = thread_transcripts([message("g1")])[0]
        assert one_day.doc_key == f"{THREAD}::2016-03-04"
        spanning = thread_transcripts(
            [message("g1"), message("g2", when=START + timedelta(days=2))]
        )[0]
        assert spanning.doc_key == f"{THREAD}::2016-03-04..2016-03-06"

    def test_the_same_messages_from_a_different_batch_render_identically(self) -> None:
        """A re-export's doc must be upsert-identical: same key, same text."""
        messages = [message("g1"), message("g2", when=START + timedelta(minutes=5))]
        first = thread_transcripts(merge_batches([batch(*messages)]))
        second = thread_transcripts(
            merge_batches([batch(*messages, doc_id="doc-2", path="chat-later.db")])
        )
        assert [doc.model_dump() for doc in first] == [doc.model_dump() for doc in second]

    def test_tapbacks_are_folded_beneath_their_message_in_a_stable_order(self) -> None:
        doc = thread_transcripts(
            [
                message(
                    "g1",
                    tapbacks=[
                        Tapback(kind="loved", sender_name="Carol"),
                        Tapback(kind="laughed", sender_name="Ada"),
                    ],
                )
            ]
        )[0]
        assert doc.text.splitlines()[-2:] == ["  [Ada laughed this]", "  [Carol loved this]"]

    def test_threads_are_kept_apart(self) -> None:
        docs = thread_transcripts(
            [message("g1"), message("g2", thread_id="other", thread_name="Crew")]
        )
        assert {doc.thread_id for doc in docs} == {THREAD, "other"}
        assert all(len(doc.message_guids) == 1 for doc in docs)

    def test_a_thread_without_a_name_falls_back_to_its_id(self) -> None:
        doc = thread_transcripts([message("g1", thread_name="")])[0]
        assert doc.thread_name == THREAD
        assert doc.text.startswith(f"Thread: {THREAD} (participants: Bob)")

    def test_participants_name_only_who_spoke_in_this_document(self) -> None:
        """A document's text may not depend on messages outside it."""
        docs = thread_transcripts(
            [
                message("g1", sender="Bob"),
                message("g2", sender="Carol", when=START + timedelta(days=1)),
            ],
            max_chars=1,
        )
        assert "participants: Bob)" in docs[0].text
        assert "participants: Carol)" in docs[1].text

    def test_a_tiny_cap_splits_every_day_into_its_own_document(self) -> None:
        docs = thread_transcripts(
            [message(f"g{day}", when=START + timedelta(days=day)) for day in range(3)],
            max_chars=1,
        )
        assert [doc.doc_key for doc in docs] == [
            f"{THREAD}::2016-03-0{4 + day}" for day in range(3)
        ]

    def test_days_pack_together_until_the_cap_and_split_on_a_day_boundary(self) -> None:
        # Each line is exactly 100 characters, so each day's block is 101.
        text = "x" * 76
        docs = thread_transcripts(
            [message(f"g{day}", text=text, when=START + timedelta(days=day)) for day in range(3)],
            max_chars=210,
        )
        assert [doc.message_guids for doc in docs] == [["g0", "g1"], ["g2"]]
        assert docs[0].doc_key == f"{THREAD}::2016-03-04..2016-03-05"

    def test_an_oversized_day_is_kept_whole_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A day is the atom — the engines chunk documents themselves."""
        with caplog.at_level(logging.INFO, logger="varagity.graph.render"):
            docs = thread_transcripts(
                [message("g1", text="x" * 500), message("g2", text="y" * 500)], max_chars=100
            )
        assert [doc.message_guids for doc in docs] == [["g1", "g2"]]
        assert "kept whole" in caplog.text

    def test_a_non_positive_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be positive"):
            thread_transcripts([message("g1")], max_chars=0)

    def test_no_messages_render_no_documents(self) -> None:
        assert thread_transcripts([]) == []


class TestEpisodePayloads:
    def test_one_episode_per_message_carries_guid_identity(self) -> None:
        (payload,) = episode_payloads([message("g1", text="I built the PC")])
        assert payload.name == "g1"
        assert payload.body == "Bob: I built the PC"
        assert payload.reference_time == START
        assert payload.source_description == "Hardware Talk"

    def test_reactions_ride_along_so_the_engines_get_the_same_diet(self) -> None:
        (payload,) = episode_payloads(
            [message("g1", tapbacks=[Tapback(kind="loved", sender_name="Carol")])]
        )
        assert payload.body.splitlines()[-1] == "  [Carol loved this]"

    def test_the_group_defaults_to_the_thread_and_can_be_pinned_corpus_wide(self) -> None:
        """Entity resolution happens within a partition — Q1 needs one corpus-wide group."""
        messages = [message("g1"), message("g2", thread_id="other")]
        assert [p.group_id for p in episode_payloads(messages)] == [THREAD, "other"]
        assert [p.group_id for p in episode_payloads(messages, group_id="varagity")] == [
            "varagity",
            "varagity",
        ]

    def test_payloads_are_chronological(self) -> None:
        messages = [message("late", when=START + timedelta(hours=2)), message("early")]
        assert [p.name for p in episode_payloads(messages)] == ["early", "late"]


class TestProvenanceWalk:
    INDEX = {f"{THREAD}::2016-03-04": ["g1", "g2"], f"{THREAD}::2016-03-05": ["g3"]}

    def test_an_index_is_built_from_rendered_documents(self) -> None:
        docs = thread_transcripts([message("g1"), message("g2", when=START + timedelta(minutes=1))])
        assert doc_guid_index(docs) == {f"{THREAD}::2016-03-04": ["g1", "g2"]}

    def test_an_exact_key_anywhere_in_the_payload_resolves(self) -> None:
        payload = {"chunks": [{"file_path": f"{THREAD}::2016-03-04"}]}
        assert guids_in_payload(payload, self.INDEX) == ["g1", "g2"]

    def test_a_key_embedded_in_a_path_or_a_separator_join_resolves(self) -> None:
        payload = [
            f"/work/documents/{THREAD}::2016-03-04.txt",
            f"{THREAD}::2016-03-05<SEP>{THREAD}::2016-03-04",
        ]
        assert guids_in_payload(payload, self.INDEX) == ["g1", "g2", "g3"]

    def test_keys_used_as_mapping_keys_resolve_too(self) -> None:
        assert guids_in_payload({f"{THREAD}::2016-03-05": 1}, self.INDEX) == ["g3"]

    def test_pydantic_payloads_are_walked(self) -> None:
        docs = thread_transcripts([message("g1")])
        assert guids_in_payload(docs, doc_guid_index(docs)) == ["g1"]

    def test_an_engine_that_cites_nothing_yields_no_provenance(self) -> None:
        assert guids_in_payload({"answer": "no idea"}, self.INDEX) == []
        assert guids_in_payload(None, self.INDEX) == []
        assert guids_in_payload(["", 7, None], self.INDEX) == []

    def test_guids_are_deduplicated_in_encounter_order(self) -> None:
        payload = [f"{THREAD}::2016-03-05", f"{THREAD}::2016-03-04", f"{THREAD}::2016-03-05"]
        assert guids_in_payload(payload, self.INDEX) == ["g3", "g1", "g2"]

    def test_a_cyclic_payload_does_not_recurse_forever(self) -> None:
        payload: dict[str, Any] = {"self": None}
        payload["self"] = payload
        assert guids_in_payload(payload, self.INDEX) == []

    def test_the_more_specific_key_wins_when_one_prefixes_another(self) -> None:
        index = {f"{THREAD}::2016-03-04": ["g1"], f"{THREAD}::2016-03-04..2016-03-06": ["g9"]}
        assert guids_in_payload([f"{THREAD}::2016-03-04..2016-03-06"], index) == ["g9"]


# --------------------------------------------------------------------------
# LightRAG
# --------------------------------------------------------------------------


@dataclass
class FakeQueryParam:
    """Stand-in for LightRAG's ``QueryParam``."""

    mode: str
    only_need_context: bool = False


class FakeGraphStore:
    def __init__(self, nodes: int = 3, edges: int = 2, fail: bool = False) -> None:
        self.nodes, self.edges, self.fail = nodes, edges, fail

    async def get_knowledge_graph(self, label: str) -> Any:
        if self.fail:
            raise RuntimeError("storage closed")
        return SimpleNamespace(nodes=["n"] * self.nodes, edges=["e"] * self.edges)


class FakeRag:
    """Stand-in for an initialized ``LightRAG`` instance."""

    def __init__(
        self,
        *,
        answer: str = "Bob loves mechanical keyboards.",
        context: Any = None,
        fail_insert: bool = False,
        fail_context: bool = False,
        fail_finalize: bool = False,
    ) -> None:
        self.answer = answer
        self.context = context
        self.fail_insert = fail_insert
        self.fail_context = fail_context
        self.fail_finalize = fail_finalize
        self.inserted: list[tuple[list[str], list[str]]] = []
        self.finalized = False
        self.chunk_entity_relation_graph = FakeGraphStore()

    async def ainsert(self, texts: list[str], *, ids: list[str], file_paths: list[str]) -> None:
        if self.fail_insert:
            raise RuntimeError("insert exploded")
        self.inserted.append((ids, file_paths))

    async def aquery(self, question: str, *, param: FakeQueryParam) -> Any:
        if param.only_need_context:
            if self.fail_context:
                raise RuntimeError("context exploded")
            return self.context
        return self.answer

    async def finalize_storages(self) -> None:
        if self.fail_finalize:
            raise RuntimeError("teardown exploded")
        self.finalized = True


LIGHTRAG_CONTEXT = {
    "entities": [
        {"entity_name": "Bob", "entity_type": "person", "description": "a friend"},
        {"no_name": True},
    ],
    "relationships": [
        {
            "src_id": "Bob",
            "tgt_id": "mechanical keyboard",
            "keywords": "prefers",
            "description": "Bob prefers mechanical keyboards",
        },
        {"src_id": "dangling"},
    ],
    "chunks": [{"file_path": f"{THREAD}::2016-03-04"}],
}


def lightrag_session(**kwargs: Any) -> Any:
    session = lightrag_adapter._LightRAGSession(FakeRag(**kwargs), FakeQueryParam)
    session.build([batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))])
    return session


class TestLightRAGAdapter:
    def test_build_inserts_transcripts_under_their_doc_keys(self) -> None:
        rag = FakeRag()
        session = lightrag_adapter._LightRAGSession(rag, FakeQueryParam)
        report = session.build(
            [
                batch(message("g1"), message("g2", when=START + timedelta(days=9))),
                batch(message("g3", thread_id="other", thread_name="Crew"), doc_id="doc-2"),
            ]
        )
        session.close()
        (ids, file_paths) = rag.inserted[0]
        # LightRAG's document id IS the transcript key, which is what makes a
        # re-inserted unchanged transcript a doc-status hit rather than a copy.
        assert ids == file_paths == [f"{THREAD}::2016-03-04..2016-03-13", "other::2016-03-04"]
        assert isinstance(report, BuildReport)
        assert report.messages_seen == 3
        assert report.failures == []
        assert report.wall_clock_s >= 0.0

    def test_a_second_build_over_an_overlapping_batch_sees_the_union(self) -> None:
        rag = FakeRag()
        session = lightrag_adapter._LightRAGSession(rag, FakeQueryParam)
        first = batch(message("g1"))
        second = batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))
        assert session.build([first, second]).messages_seen == 2
        session.close()

    def test_a_failed_insert_is_recorded_not_raised(self) -> None:
        session = lightrag_adapter._LightRAGSession(FakeRag(fail_insert=True), FakeQueryParam)
        report = session.build([batch(message("g1"))])
        session.close()
        assert report.failures and "insert exploded" in report.failures[0]

    def test_query_answers_with_normalized_evidence_and_provenance(self) -> None:
        session = lightrag_session(context=LIGHTRAG_CONTEXT)
        answer = session.query("What does Bob think about computers?")
        session.close()
        assert isinstance(answer, GraphAnswer)
        assert answer.mode == lightrag_adapter.PRIMARY_MODE
        assert answer.answer == "Bob loves mechanical keyboards."
        assert [e.name for e in answer.evidence.entities] == ["Bob"]
        assert answer.evidence.relations[0].target == "mechanical keyboard"
        assert answer.evidence.message_guids == ["g1", "g2"]
        assert answer.evidence.communities == []  # LightRAG has no community layer
        assert answer.latency_s >= 0.0

    def test_an_explicit_mode_overrides_the_primary(self) -> None:
        session = lightrag_session()
        assert session.query("q", mode="global").mode == "global"
        session.close()

    def test_a_reasoning_stage_never_reaches_the_answer(self) -> None:
        session = lightrag_session(answer="<think>hmm</think>Bob likes ARM.")
        assert session.query("q").answer == "Bob likes ARM."
        session.close()

    def test_failed_context_retrieval_still_answers(self, caplog: pytest.LogCaptureFixture) -> None:
        session = lightrag_session(fail_context=True)
        with caplog.at_level(logging.WARNING, logger="varagity.graph.engines.lightrag"):
            answer = session.query("q")
        session.close()
        assert answer.answer
        assert answer.evidence == GraphEvidence()
        assert "context retrieval failed" in caplog.text

    def test_stats_read_the_graph_storage(self) -> None:
        session = lightrag_session()
        assert session.stats() == GraphStats(entities=3, relations=2, communities=None)
        session.close()

    def test_unavailable_stats_are_reported_as_unknown_not_zero(self) -> None:
        session = lightrag_session()
        session._rag.chunk_entity_relation_graph = FakeGraphStore(fail=True)
        assert session.stats() == GraphStats(entities=None, relations=None, communities=None)
        session.close()

    def test_close_finalizes_storages_and_survives_a_failed_teardown(self) -> None:
        rag = FakeRag()
        lightrag_adapter._LightRAGSession(rag, FakeQueryParam).close()
        assert rag.finalized is True
        lightrag_adapter._LightRAGSession(FakeRag(fail_finalize=True), FakeQueryParam).close()

    @pytest.mark.parametrize(
        "context",
        [None, "not json at all", "[]", {"entities": "not a list"}, {"entities": [1, 2]}],
    )
    def test_unknown_context_shapes_degrade_instead_of_raising(self, context: Any) -> None:
        evidence = lightrag_adapter.evidence_from_context(context, {})
        assert evidence.entities == []
        assert evidence.relations == []
        assert evidence.message_guids == []

    def test_a_json_string_context_is_parsed(self) -> None:
        evidence = lightrag_adapter.evidence_from_context(
            '{"entities": [{"entity_name": "Jane"}]}', {}
        )
        assert [e.name for e in evidence.entities] == ["Jane"]

    def test_prose_context_is_kept_raw_for_the_adr_autopsy(self) -> None:
        evidence = lightrag_adapter.evidence_from_context("Bob said so.", {})
        assert evidence.raw == {"context": "Bob said so."}

    def test_the_generation_cap_is_clamped_to_the_context_window(self) -> None:
        """llama.cpp hard-500s at the window instead of stopping gracefully."""
        assert (
            lightrag_adapter.fit_max_tokens([{"role": "user", "content": "hi"}], 2048, 16384)
            == 2048
        )
        long_prompt = [{"role": "user", "content": "word " * 6000}]
        clamped = lightrag_adapter.fit_max_tokens(long_prompt, 2048, 8192)
        assert 256 <= clamped < 2048

    def test_the_clamp_never_returns_a_useless_cap(self) -> None:
        assert (
            lightrag_adapter.fit_max_tokens([{"role": "user", "content": "x " * 5000}], 2048, 512)
            == 256
        )

    def test_a_small_requested_cap_is_never_raised_by_the_floor(self) -> None:
        assert (
            lightrag_adapter.fit_max_tokens([{"role": "user", "content": None}], 128, 16384) == 128
        )

    def test_the_environment_pins_hold_the_engine_to_one_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in lightrag_adapter._ENV_PINS:
            monkeypatch.setenv(name, "stale")
        lightrag_adapter._pin_env()
        assert os.environ["MAX_ASYNC_LLM"] == "1"
        assert os.environ["MAX_PARALLEL_INSERT"] == "1"
        assert os.environ["LIGHTRAG_GRAPH_STORAGE"] == "NetworkXStorage"


class TestLightRAGModelFuncs:
    """The engine hooks: the <think> strip and the embedding shape."""

    class FakeCompletions:
        def __init__(self, content: str) -> None:
            self.content = content
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
            )

    class FakeEmbeddings:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.25])])

    def client(self, content: str) -> Any:
        completions = self.FakeCompletions(content)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=completions), embeddings=self.FakeEmbeddings()
        )

    def run(self, coroutine: Any) -> Any:
        import asyncio

        return asyncio.run(coroutine)

    def test_every_engine_completion_is_think_stripped(self) -> None:
        """★ The trap: LightRAG parses a delimiter grammar out of this string."""
        client = self.client('<think>reasoning</think>("entity"<|>Bob<|>person)')
        func = lightrag_adapter.make_llm_func(
            client, model="m", temperature=0.6, context_tokens=16384
        )
        assert self.run(func("extract entities")) == '("entity"<|>Bob<|>person)'

    def test_the_system_prompt_and_history_are_forwarded_in_order(self) -> None:
        client = self.client("ok")
        func = lightrag_adapter.make_llm_func(
            client, model="m", temperature=0.6, context_tokens=16384
        )
        self.run(
            func(
                "now this",
                system_prompt="be terse",
                history_messages=[{"role": "user", "content": "earlier"}],
                keyword_extraction=True,  # LightRAG passes stage flags through kwargs
            )
        )
        sent = client.chat.completions.calls[0]["messages"]
        assert [m["role"] for m in sent] == ["system", "user", "user"]
        assert sent[-1]["content"] == "now this"

    def test_an_empty_completion_is_an_empty_string(self) -> None:
        client = self.client("")
        func = lightrag_adapter.make_llm_func(
            client, model="m", temperature=0.6, context_tokens=16384
        )
        assert self.run(func("anything")) == ""

    def test_embeddings_come_back_in_order_without_a_query_prefix(self) -> None:
        """The recorded e5 deviation: LightRAG has one function for both sides."""
        client = self.client("unused")
        embed = lightrag_adapter.make_embedding_func(client, model="e5")
        assert self.run(embed(["a passage"])) == [[0.5, 0.25]]
        assert client.embeddings.calls[0]["input"] == ["a passage"]


# --------------------------------------------------------------------------
# cognee
# --------------------------------------------------------------------------


class FakeSearchTypes:
    GRAPH_COMPLETION = "graph-completion"
    CHUNKS = "chunks"


class FakeCogneeGraph:
    def __init__(self, nodes: int = 4, edges: int = 3) -> None:
        self.nodes, self.edges = nodes, edges

    async def get_graph_data(self) -> Any:
        return ["n"] * self.nodes, ["e"] * self.edges


class FakeCognee:
    """Stand-in for the ``cognee`` module.

    Ingestion is grouped, so the double counts *attempts* as well as
    successes: ``fail_add_groups``/``fail_cognify_groups`` fail one specific
    group (1-based) and leave the rest working, which is how the isolation
    guarantee is asserted.
    """

    def __init__(
        self,
        *,
        results: dict[str, Any] | None = None,
        fail_add: bool = False,
        fail_cognify: bool = False,
        fail_search: bool = False,
        fail_add_groups: Sequence[int] = (),
        fail_cognify_groups: Sequence[int] = (),
    ) -> None:
        self.results = results or {}
        self.fail_add = fail_add
        self.fail_cognify = fail_cognify
        self.fail_search = fail_search
        self.fail_add_groups = set(fail_add_groups)
        self.fail_cognify_groups = set(fail_cognify_groups)
        self.added: list[list[str]] = []
        self.add_datasets: list[str] = []
        self.cognified: list[list[str]] = []
        self.batch_sizes: list[int | None] = []
        self.add_attempts = 0
        self.cognify_attempts = 0

    async def add(self, paths: list[str], *, dataset_name: str) -> None:
        self.add_attempts += 1
        if self.fail_add or self.add_attempts in self.fail_add_groups:
            raise RuntimeError(f"add exploded on group {self.add_attempts}")
        self.added.append(paths)
        self.add_datasets.append(dataset_name)

    async def cognify(self, *, datasets: list[str], chunks_per_batch: int | None = None) -> None:
        self.cognify_attempts += 1
        if self.fail_cognify or self.cognify_attempts in self.fail_cognify_groups:
            raise RuntimeError(f"cognify exploded on group {self.cognify_attempts}")
        self.cognified.append(datasets)
        self.batch_sizes.append(chunks_per_batch)

    async def search(self, *, query_text: str, query_type: str, datasets: list[str]) -> Any:
        if self.fail_search:
            raise RuntimeError("search exploded")
        return self.results.get(query_type)


COGNEE_TRIPLETS = [
    [
        {"name": "Bob", "type": "Person", "description": "a friend"},
        {"relationship_name": "prefers"},
        {"name": "mechanical keyboard", "type": "Thing"},
    ],
    ["not", "a", "triplet-of-mappings"],
    {"not": "a triplet"},
]


def cognee_session(tmp_path: Path, api: FakeCognee, *, graph: Any = None) -> Any:
    session = cognee_adapter._CogneeSession(
        api,
        FakeSearchTypes,
        workdir=tmp_path,
        graph_engine=graph or (lambda: _resolved(FakeCogneeGraph())),
    )
    session.build([batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))])
    return session


def build_over_threads(tmp_path: Path, api: FakeCognee, count: int) -> BuildReport:
    """Build a corpus of ``count`` threads, i.e. ``count`` transcript documents."""
    session = cognee_adapter._CogneeSession(
        api,
        FakeSearchTypes,
        workdir=tmp_path,
        graph_engine=lambda: _resolved(FakeCogneeGraph()),
    )
    messages = [
        message(f"g{index}", thread_id=f"thread-{index}", thread_name=f"Thread {index}")
        for index in range(count)
    ]
    try:
        return session.build([batch(*messages)])
    finally:
        session.close()


async def _resolved(value: Any) -> Any:
    return value


class TestCogneeAdapter:
    def test_build_writes_transcript_files_then_cognifies(self, tmp_path: Path) -> None:
        api = FakeCognee()
        session = cognee_session(tmp_path, api)
        session.close()
        (written,) = api.added
        assert api.cognified == [[cognee_adapter.DATASET]]
        assert Path(written[0]).read_text(encoding="utf-8").startswith("Thread: Hardware Talk")
        assert Path(written[0]).parent == tmp_path / "documents"

    def test_cognify_bounds_its_extraction_fan_out(self, tmp_path: Path) -> None:
        """Extraction gathers over a whole batch with no concurrency cap.

        The batch size is therefore the queue depth on a single-slot
        llama.cpp, and cognee's default of 100 put the tail request's wait
        past LiteLLM's deadline (2026-07-28, the first full-profile run:
        76 min, zero entities).
        """
        api = FakeCognee()
        session = cognee_session(tmp_path, api)
        session.close()
        assert api.batch_sizes == [8]

    def test_documents_are_ingested_in_groups_of_the_shipped_size(self) -> None:
        """The blast-radius constant, asserted at its real boundary.

        250 documents at 100 per pass is three groups — the arithmetic the
        isolation depends on, checked without patching the constant away.
        """
        paths = [Path(f"doc-{index}.txt") for index in range(250)]
        groups = cognee_adapter._document_groups(paths)
        assert [len(group) for group in groups] == [100, 100, 50]
        assert [path for group in groups for path in group] == paths
        assert cognee_adapter._document_groups([]) == []

    def test_each_group_is_one_add_and_one_cognify_against_one_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """★ Grouped cognify: the failure blast radius is one group.

        A ``cognify`` is one cognee pipeline run, and a failed run is rolled
        back whole — so five documents ingested two at a time are three
        recoverable units instead of one all-or-nothing build (2026-07-29,
        the second full-profile attempt: one think-spiralled chunk zeroed
        3.1 hours of extraction).
        """
        monkeypatch.setattr(cognee_adapter, "_DOCS_PER_COGNIFY", 2)
        api = FakeCognee()
        report = build_over_threads(tmp_path, api, 5)
        assert [len(group) for group in api.added] == [2, 2, 1]
        # One dataset throughout: search has to span the whole corpus, and
        # cognee's incremental loading is what keeps the re-visits free.
        assert api.add_datasets == [cognee_adapter.DATASET] * 3
        assert api.cognified == [[cognee_adapter.DATASET]] * 3
        # The fan-out bound rides on every pass, not just the first.
        assert api.batch_sizes == [8, 8, 8]
        assert report.failures == []

    def test_a_failed_group_is_recorded_with_its_index_and_the_rest_still_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """★ The isolation guarantee: group 2 dies, group 3 still indexes."""
        monkeypatch.setattr(cognee_adapter, "_DOCS_PER_COGNIFY", 1)
        api = FakeCognee(fail_cognify_groups=[2])
        report = build_over_threads(tmp_path, api, 3)
        assert api.cognify_attempts == 3
        assert api.cognified == [[cognee_adapter.DATASET]] * 2  # groups 1 and 3
        (failure,) = report.failures
        assert failure.startswith("cognify[2/3]: ")
        assert "cognify exploded on group 2" in failure

    def test_a_failed_add_costs_only_its_own_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cognee_adapter, "_DOCS_PER_COGNIFY", 1)
        api = FakeCognee(fail_add_groups=[1])
        report = build_over_threads(tmp_path, api, 3)
        # The failed group is never cognified — nothing new to build from —
        # while the two behind it are added and cognified as usual.
        assert api.add_attempts == 3 and api.cognify_attempts == 2
        (failure,) = report.failures
        assert failure.startswith("add[1/3]: ")

    def test_transcript_file_names_survive_unsafe_thread_ids(self) -> None:
        assert cognee_adapter.transcript_filename(f"{THREAD}::2016-03-04") == (
            "iMessage_-_15125550101_2016-03-04.txt"
        )

    def test_a_failed_add_skips_cognify_and_is_recorded(self, tmp_path: Path) -> None:
        api = FakeCognee(fail_add=True)
        session = cognee_adapter._CogneeSession(
            api,
            FakeSearchTypes,
            workdir=tmp_path,
            graph_engine=lambda: _resolved(FakeCogneeGraph()),
        )
        report = session.build([batch(message("g1"))])
        session.close()
        assert api.cognified == []
        assert report.failures and "add exploded" in report.failures[0]

    def test_a_failed_cognify_is_recorded(self, tmp_path: Path) -> None:
        api = FakeCognee(fail_cognify=True)
        session = cognee_adapter._CogneeSession(
            api,
            FakeSearchTypes,
            workdir=tmp_path,
            graph_engine=lambda: _resolved(FakeCogneeGraph()),
        )
        report = session.build([batch(message("g1"))])
        session.close()
        assert report.failures and "cognify exploded" in report.failures[0]

    def test_an_empty_corpus_never_calls_the_engine(self, tmp_path: Path) -> None:
        api = FakeCognee()
        session = cognee_adapter._CogneeSession(
            api,
            FakeSearchTypes,
            workdir=tmp_path,
            graph_engine=lambda: _resolved(FakeCogneeGraph()),
        )
        assert session.build([]).messages_seen == 0
        session.close()
        assert api.added == [] and api.cognified == []

    def test_query_answers_from_completion_and_maps_triplet_evidence(self, tmp_path: Path) -> None:
        api = FakeCognee(
            results={
                FakeSearchTypes.GRAPH_COMPLETION: ["Bob prefers mechanical keyboards."],
                FakeSearchTypes.CHUNKS: COGNEE_TRIPLETS,
            }
        )
        session = cognee_session(tmp_path, api)
        answer = session.query("What does Bob think about computers?")
        session.close()
        assert answer.answer == "Bob prefers mechanical keyboards."
        assert answer.mode == cognee_adapter.PRIMARY_MODE
        assert [e.name for e in answer.evidence.entities] == ["Bob", "mechanical keyboard"]
        assert answer.evidence.relations[0].label == "prefers"
        assert answer.evidence.communities == []  # cognee has no community layer

    def test_provenance_resolves_through_the_written_file_name(self, tmp_path: Path) -> None:
        stem = Path(cognee_adapter.transcript_filename(f"{THREAD}::2016-03-04")).stem
        api = FakeCognee(
            results={
                FakeSearchTypes.GRAPH_COMPLETION: [f"see /work/documents/{stem}.txt"],
                FakeSearchTypes.CHUNKS: [],
            }
        )
        session = cognee_session(tmp_path, api)
        answer = session.query("q")
        session.close()
        assert answer.evidence.message_guids == ["g1", "g2"]

    def test_a_failed_search_answers_empty_rather_than_raising(self, tmp_path: Path) -> None:
        session = cognee_session(tmp_path, FakeCognee(fail_search=True))
        answer = session.query("q")
        session.close()
        assert answer.answer == ""
        assert answer.evidence.message_guids == []

    def test_an_unknown_search_type_is_skipped_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = cognee_session(tmp_path, FakeCognee())
        with caplog.at_level(logging.WARNING, logger="varagity.graph.engines.cognee"):
            answer = session.query("q", mode="TELEPATHY")
        session.close()
        assert answer.mode == "TELEPATHY"
        assert "unknown cognee search type" in caplog.text

    def test_stats_read_the_graph_engine(self, tmp_path: Path) -> None:
        session = cognee_session(tmp_path, FakeCognee())
        assert session.stats() == GraphStats(entities=4, relations=3, communities=None)
        session.close()

    def test_unavailable_stats_are_reported_as_unknown(self, tmp_path: Path) -> None:
        def boom() -> Any:
            raise RuntimeError("no graph engine")

        session = cognee_session(tmp_path, FakeCognee(), graph=boom)
        assert session.stats() == GraphStats()
        session.close()

    @pytest.mark.parametrize(
        ("results", "expected"),
        [
            (None, ""),
            ("plain string", "plain string"),
            ([" a ", "b"], "a\nb"),
            ([{"answer": 1}], "{'answer': 1}"),
            (42, "42"),
            # The per-dataset wrapper cognee 1.4 returns live (Phase 3 gate):
            # the answer is the search_result content, not the dict repr.
            (
                [{"dataset_id": "d-1", "dataset_name": "g", "search_result": ["He hated them."]}],
                "He hated them.",
            ),
        ],
    )
    def test_answers_flatten_whatever_shape_search_returned(
        self, results: Any, expected: str
    ) -> None:
        assert cognee_adapter.answer_from_results(results) == expected

    def test_dataset_wrapped_evidence_maps_like_plain_evidence(self) -> None:
        """The live per-dataset wrapper must not hide the payload inside it."""
        wrapped = [{"dataset_name": "varagity_graph", "search_result": COGNEE_TRIPLETS}]
        plain = cognee_adapter.evidence_from_search(COGNEE_TRIPLETS, None, {})
        unwrapped = cognee_adapter.evidence_from_search(wrapped, None, {})
        assert [e.name for e in unwrapped.entities] == [e.name for e in plain.entities]
        assert len(unwrapped.relations) == len(plain.relations) == 1

    @pytest.mark.parametrize(
        "insights",
        [None, "prose", [], [[1, 2, 3]], {"a": 1}, [[object(), object(), object()]]],
    )
    def test_unknown_evidence_shapes_degrade_instead_of_raising(self, insights: Any) -> None:
        evidence = cognee_adapter.evidence_from_search(insights, None, {})
        assert evidence.entities == [] and evidence.relations == []

    def test_pydantic_insight_models_are_mapped(self) -> None:
        node = SimpleNamespace(model_dump=lambda: {"name": "Jane"})
        edge = SimpleNamespace(model_dump=lambda: {"relationship_name": "likes"})
        other = SimpleNamespace(model_dump=lambda: {"name": "piano"})
        evidence = cognee_adapter.evidence_from_search([[node, edge, other]], None, {})
        assert [e.name for e in evidence.entities] == ["Jane", "piano"]
        assert evidence.relations[0].label == "likes"

    def test_the_environment_pins_point_cognee_at_the_local_endpoints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_env: Callable[..., None]
    ) -> None:
        settings_env(
            BASE_MODEL_API_URL="http://llamacpp:8080/v1",
            EMBEDDING_API_URL="http://infinity:8081/v1",
        )
        # Every pinned name goes through monkeypatch first: several of them
        # (EMBEDDING_MODEL, EMBEDDING_API_KEY) are varagity's own setting
        # names, so an unrestored write would leak into later tests' settings.
        for name in cognee_adapter._env_pins(tmp_path):
            monkeypatch.setenv(name, "stale")
        cognee_adapter._pin_env(tmp_path)
        assert os.environ["LLM_PROVIDER"] == "custom"
        # LiteLLM (the custom-provider route) rejects a bare model name.
        assert os.environ["LLM_MODEL"].startswith("openai/")
        # A pre-flight guardrail must not fail a whole build (Phase 3 gate).
        assert os.environ["COGNEE_SKIP_CONNECTION_TEST"] == "true"
        # Multi-tenant mode wraps results and hides the graph from the stats
        # context; the bake-off runs cognee single-user, like the repo.
        assert os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] == "false"
        # Session memory would fold earlier answers into later ones and
        # contaminate golden scoring.
        assert os.environ["CACHING"] == "false"
        # LiteLLM's own pin, not cognee's: its unset default resolves to a
        # 600 s completion deadline, which a queued call on a single-slot
        # server blows through (2026-07-28 full-profile run). Reaches the
        # search path too — same GenericAPIAdapter, same transport.
        assert os.environ["REQUEST_TIMEOUT"] == "3600"
        assert os.environ["EMBEDDING_PROVIDER"] == "openai_compatible"
        # the direct-SDK embedding path appends /v1 itself
        assert os.environ["EMBEDDING_ENDPOINT"] == "http://infinity:8081"
        assert os.environ["LLM_ENDPOINT"] == "http://llamacpp:8080/v1"


# --------------------------------------------------------------------------
# Graphiti
# --------------------------------------------------------------------------


class FakeDriver:
    def __init__(self, counts: Sequence[Any] = (7, 5, 2)) -> None:
        self.counts = list(counts)
        self.queries: list[str] = []

    async def execute_query(self, query: str) -> Any:
        self.queries.append(query)
        value = self.counts.pop(0)
        if isinstance(value, Exception):
            raise value
        return [{"value": value}]


@dataclass
class FakeGraphiti:
    """Stand-in for an initialized ``Graphiti`` instance."""

    search_results: Any = None
    fail_episodes: set[str] = field(default_factory=set)
    fail_search: bool = False
    fail_close: bool = False
    episodes: list[dict[str, Any]] = field(default_factory=list)
    communities_built: int = 0
    closed: bool = False
    driver: FakeDriver = field(default_factory=FakeDriver)

    async def add_episode(self, **kwargs: Any) -> Any:
        if kwargs["name"] in self.fail_episodes:
            raise RuntimeError("extraction exploded")
        self.episodes.append(kwargs)
        return SimpleNamespace(episode=SimpleNamespace(uuid=f"uuid-{kwargs['name']}"))

    async def build_communities(self) -> None:
        self.communities_built += 1

    async def search(self, question: str) -> Any:
        if self.fail_search:
            raise RuntimeError("search exploded")
        return self.search_results

    async def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("close exploded")
        self.closed = True


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


GRAPHITI_EDGES = [
    {
        "source_node_uuid": "n-bob",
        "target_node_uuid": "n-kb",
        "name": "PREFERS",
        "fact": "Bob prefers mechanical keyboards",
        "episodes": ["uuid-g1", "uuid-unknown"],
    },
    {"nothing": "usable"},
]


def graphiti_session(*, llm: Any = None, **kwargs: Any) -> Any:
    session = graphiti_adapter._GraphitiSession(
        FakeGraphiti(**kwargs), "message", llm=llm or ScriptedLLM()
    )
    session.build([batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))])
    return session


class TestGraphitiAdapter:
    def test_build_adds_one_episode_per_message_under_one_group(self) -> None:
        session = graphiti_session()
        session.close()
        engine = session._graphiti
        assert [episode["name"] for episode in engine.episodes] == ["g1", "g2"]
        assert {episode["group_id"] for episode in engine.episodes} == {graphiti_adapter.GROUP_ID}
        assert engine.episodes[0]["source"] == "message"
        # Communities are never built — the per-episode path is broken in
        # graphiti-core 0.29.2 (Phase 3 gate) and the end-of-build pass can
        # hang forever in an uncapped label-propagation loop (Phase 4 gate).
        assert "update_communities" not in engine.episodes[0]
        assert engine.communities_built == 0

    def test_a_second_build_skips_episodes_already_added(self) -> None:
        """Episode identity is message identity — an overlapping batch upserts."""
        session = graphiti_session()
        report = session.build(
            [batch(message("g1"), message("g3", when=START + timedelta(days=1)))]
        )
        session.close()
        assert [episode["name"] for episode in session._graphiti.episodes] == ["g1", "g2", "g3"]
        assert report.messages_seen == 2  # what the batch held, not what was new

    def test_a_failed_episode_is_recorded_and_the_run_continues(self) -> None:
        session = graphiti_adapter._GraphitiSession(
            FakeGraphiti(fail_episodes={"g1"}), "message", llm=ScriptedLLM()
        )
        report = session.build(
            [batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))]
        )
        session.close()
        assert [episode["name"] for episode in session._graphiti.episodes] == ["g2"]
        assert report.failures and "episode g1" in report.failures[0]

    def test_the_community_pass_is_skipped_and_the_skip_is_recorded(self) -> None:
        """★ Phase 4 gate: 0.29.2's uncapped label propagation can hang forever."""
        session = graphiti_adapter._GraphitiSession(FakeGraphiti(), "message", llm=ScriptedLLM())
        report = session.build([batch(message("g1"))])
        session.close()
        assert session._graphiti.communities_built == 0
        assert report.failures and "build_communities" in report.failures[0]
        assert "skipped" in report.failures[0]

    def test_an_empty_corpus_records_no_community_skip(self) -> None:
        session = graphiti_adapter._GraphitiSession(FakeGraphiti(), "message", llm=ScriptedLLM())
        report = session.build([])
        session.close()
        assert report.messages_seen == 0
        assert report.failures == []
        assert session._graphiti.communities_built == 0

    def test_query_synthesizes_an_answer_over_the_retrieved_facts(self) -> None:
        """★ Decision #12: Graphiti returns facts, so the answer is ours."""
        llm = ScriptedLLM()
        session = graphiti_session(llm=llm, search_results=GRAPHITI_EDGES)
        answer = session.query("What does Bob think about computers?")
        session.close()
        assert answer.answer == "Bob prefers mechanical keyboards."
        assert answer.mode == graphiti_adapter.PRIMARY_MODE
        assert answer.evidence.relations[0].description == "Bob prefers mechanical keyboards"
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "- Bob prefers mechanical keyboards" in prompt
        assert "What does Bob think about computers?" in prompt

    def test_provenance_resolves_episode_uuids_recorded_during_the_build(self) -> None:
        session = graphiti_session(search_results=GRAPHITI_EDGES)
        answer = session.query("q")
        session.close()
        assert answer.evidence.message_guids == ["g1"]  # the unknown uuid is dropped

    def test_a_combined_results_object_contributes_nodes_episodes_and_communities(self) -> None:
        results = SimpleNamespace(
            edges=GRAPHITI_EDGES,
            nodes=[{"name": "Bob", "labels": ["Entity", "Person"], "summary": "a friend"}],
            episodes=[{"name": "g2"}],
            communities=[{"uuid": "c1", "name": "Hardware", "summary": "talk about PCs"}],
        )
        session = graphiti_session(search_results=results)
        answer = session.query("q")
        session.close()
        assert [(e.name, e.type) for e in answer.evidence.entities] == [("Bob", "Person")]
        assert [c.title for c in answer.evidence.communities] == ["Hardware"]
        assert answer.evidence.message_guids == ["g2", "g1"]  # episodes first, then edges

    def test_a_failed_search_still_produces_a_scored_answer(self) -> None:
        session = graphiti_session(fail_search=True)
        answer = session.query("q")
        session.close()
        assert answer.answer == "The graph returned no facts for this question."
        assert answer.evidence.relations == []

    def test_a_failed_synthesis_answers_empty_rather_than_raising(self) -> None:
        session = graphiti_session(
            llm=ScriptedLLM(RuntimeError("model down")), search_results=GRAPHITI_EDGES
        )
        assert session.query("q").answer == ""
        session.close()

    def test_a_reasoning_stage_never_reaches_the_answer(self) -> None:
        """★ The same trap as condense/HyDE: generate() does not strip <think>."""
        session = graphiti_session(
            llm=ScriptedLLM("<think>weighing facts</think>Bob likes ARM."),
            search_results=GRAPHITI_EDGES,
        )
        assert session.query("q").answer == "Bob likes ARM."
        session.close()

    def test_stats_count_entities_relations_and_communities(self) -> None:
        session = graphiti_session()
        assert session.stats() == GraphStats(entities=7, relations=5, communities=2)
        session.close()

    def test_a_failed_stats_query_reports_unknown_not_zero(self) -> None:
        session = graphiti_session()
        session._graphiti.driver = FakeDriver([RuntimeError("no such label"), 1, 0])
        assert session.stats() == GraphStats(entities=None, relations=1, communities=0)
        session.close()

    def test_a_count_that_is_not_a_number_reports_unknown(self) -> None:
        """A boolean is an int subclass, and a string is not a count at all."""
        session = graphiti_session()
        session._graphiti.driver = FakeDriver([True, "seven", {"value": 3}])
        assert session.stats() == GraphStats(entities=None, relations=None, communities=3)
        session.close()

    @pytest.mark.parametrize(
        ("nodes", "communities"),
        [
            ([{"summary": "nameless"}], [{"uuid": "c1"}]),
            ([{"name": "Bob", "labels": "Entity"}], [{"summary": ""}]),
        ],
    )
    def test_records_missing_their_load_bearing_field_are_dropped(
        self, nodes: list[Any], communities: list[Any]
    ) -> None:
        results = SimpleNamespace(edges=[], nodes=nodes, episodes=[], communities=communities)
        evidence = graphiti_adapter.evidence_from_search(results, {})
        assert evidence.communities == []
        assert [entity.type for entity in evidence.entities] == (
            [] if len(nodes[0]) == 1 else [None]
        )

    def test_close_survives_a_failed_teardown(self) -> None:
        session = graphiti_adapter._GraphitiSession(
            FakeGraphiti(fail_close=True), "message", llm=ScriptedLLM()
        )
        session.close()

    @pytest.mark.parametrize("results", [None, "prose", [], [{"nothing": 1}], 17])
    def test_unknown_search_shapes_degrade_instead_of_raising(self, results: Any) -> None:
        evidence = graphiti_adapter.evidence_from_search(results, {})
        assert evidence.entities == [] and evidence.relations == []
        assert evidence.message_guids == []

    def test_pydantic_search_results_are_dumped_before_mapping(self) -> None:
        edge = SimpleNamespace(model_dump=lambda mode: {"fact": "Jane plays piano"})
        evidence = graphiti_adapter.evidence_from_search([edge], {})
        assert evidence.relations[0].description == "Jane plays piano"

    def test_facts_with_no_content_are_not_offered_to_the_synthesizer(self) -> None:
        evidence = graphiti_adapter.evidence_from_search([{"source_node_uuid": "n1"}], {})
        assert graphiti_adapter.facts_block(evidence) == ""

    def test_community_summaries_join_the_facts(self) -> None:
        evidence = GraphEvidence(
            communities=[
                graphiti_adapter.GraphCommunity(id="c1", title="Hardware", summary="PC talk")
            ]
        )
        assert graphiti_adapter.facts_block(evidence) == "- community Hardware: PC talk"

    def test_the_semaphore_pin_lands_before_the_engine_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEMAPHORE_LIMIT", "20")
        graphiti_adapter._pin_env()
        assert os.environ["SEMAPHORE_LIMIT"] == "1"


class TestSessionProtocol:
    def test_every_adapter_session_implements_the_whole_protocol(self, tmp_path: Path) -> None:
        """Structural conformance — the harness drives all three identically."""
        sessions: list[Any] = [
            lightrag_adapter._LightRAGSession(FakeRag(), FakeQueryParam),
            cognee_adapter._CogneeSession(
                FakeCognee(),
                FakeSearchTypes,
                workdir=tmp_path,
                graph_engine=lambda: _resolved(FakeCogneeGraph()),
            ),
            graphiti_adapter._GraphitiSession(FakeGraphiti(), "message", llm=ScriptedLLM()),
        ]
        try:
            for session in sessions:
                for method in GraphSession.__protocol_attrs__:
                    assert callable(getattr(session, method)), (type(session), method)
        finally:
            for session in sessions:
                session.close()
