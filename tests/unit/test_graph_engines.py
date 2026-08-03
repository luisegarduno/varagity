"""Unit tests for the graph-engine seam and the shipped LightRAG adapter.

The engine library is never imported here — that is the point. The registry
must be free (``import varagity.graph.engines`` pulls no engine), the pure
rendering must be exact (it is where most adapter correctness lives), the
evidence normalizers must map canned engine payloads and degrade rather than
raise on shapes they have never seen, and the session class must be drivable
by doubles so the only untested surface left is the lazy-import block inside
``session()``.
"""

import asyncio
import logging
import os
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from varagity.graph import answer as graph_answer
from varagity.graph.base import (
    GRAPH_ENGINE_REGISTRY,
    GraphEngine,
    GraphSession,
    get_graph_engine,
    register,
)
from varagity.graph.engines import lightrag as lightrag_adapter
from varagity.graph.manifest import (
    ManifestDoc,
    WorkdirManifest,
    load_manifest,
    load_summary,
    save_manifest,
)
from varagity.graph.records import BuildReport, GraphAnswer, GraphEvidence, GraphStats
from varagity.graph.render import (
    doc_guid_index,
    guids_in_payload,
    merge_batches,
    thread_transcripts,
)
from varagity.graph.sources.base import MessageBatch, SourceMessage, Tapback
from varagity.models.embeddings import format_query

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


class ScriptedLLM:
    """Records generate() calls; returns a scripted response or raises.

    Drives the ``+synthesis`` modes — ADR-017's retrieval-only design, where
    the answer is ours rather than the engine's.
    """

    def __init__(self, response: str | Exception = "Bob prefers mechanical keyboards.") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append({"messages": list(messages), **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    @property
    def prompt(self) -> str:
        """The prompt text of the first recorded call."""
        return str(self.calls[0]["messages"][0]["content"])


class TestRegistry:
    def test_the_shipped_engine_is_the_only_seat(self) -> None:
        """ADR-017 picked one; the losers went out with stage-2's start."""
        assert set(GRAPH_ENGINE_REGISTRY) == {"lightrag"}

    def test_get_graph_engine_returns_the_registered_instance(self) -> None:
        assert get_graph_engine("lightrag") is GRAPH_ENGINE_REGISTRY["lightrag"]

    def test_unknown_engine_raises_keyerror_listing_available(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_graph_engine("made_up")
        message_text = str(excinfo.value)
        assert "made_up" in message_text
        assert "lightrag" in message_text

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
        """★ The guard that keeps collection engine-free (stage-1 decision #8).

        Still load-bearing now that ``lightrag-hku`` is a main dependency and
        *is* installed: the adapter must import it inside ``session()``, or
        every unit run and every ``import varagity`` pays for the engine.
        This module has already imported the adapter; if it touched its
        library at module level, the library would be in sys.modules by now.
        """
        import varagity.graph.engines  # noqa: F401  (the import under test)

        assert "lightrag" not in sys.modules


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
    """Stand-in for an initialized ``LightRAG`` instance.

    ``calls`` records the *order* of the pipeline verbs, which is where the
    upsert's correctness lives: a changed document must be deleted before it
    is re-enqueued, or the engine's dedup silently keeps the stale one.
    """

    def __init__(
        self,
        *,
        answer: str = "Bob loves mechanical keyboards.",
        context: Any = None,
        query_data: Any = None,
        knowledge_graph: Any = None,
        statuses: Any = None,
        fail_insert: bool = False,
        fail_context: bool = False,
        fail_query_data: bool = False,
        fail_finalize: bool = False,
        fail_process: bool = False,
        fail_delete: Sequence[str] = (),
        fail_export: bool = False,
        fail_statuses: bool = False,
    ) -> None:
        self.answer = answer
        self.context = context
        self.query_data = query_data
        self.knowledge_graph = knowledge_graph
        self.statuses = {"processed": 2} if statuses is None else statuses
        self.fail_insert = fail_insert
        self.fail_context = fail_context
        self.fail_query_data = fail_query_data
        self.fail_finalize = fail_finalize
        self.fail_process = fail_process
        self.fail_delete = set(fail_delete)
        self.fail_export = fail_export
        self.fail_statuses = fail_statuses
        self.inserted: list[tuple[list[str], list[str]]] = []
        self.deleted: list[str] = []
        self.processed = 0
        self.exports: list[tuple[str, int, int]] = []
        self.calls: list[str] = []
        self.queries: list[FakeQueryParam] = []
        self.data_queries: list[FakeQueryParam] = []
        self.finalized = False
        self.chunk_entity_relation_graph = FakeGraphStore()

    async def apipeline_enqueue_documents(
        self, texts: list[str], *, ids: list[str], file_paths: list[str]
    ) -> str:
        self.calls.append(f"enqueue:{','.join(ids)}")
        if self.fail_insert:
            raise RuntimeError("enqueue exploded")
        self.inserted.append((ids, file_paths))
        return "track-1"

    async def apipeline_process_enqueue_documents(self) -> None:
        self.calls.append("process")
        if self.fail_process:
            raise RuntimeError("process exploded")
        self.processed += 1

    async def adelete_by_doc_id(self, doc_id: str) -> Any:
        self.calls.append(f"delete:{doc_id}")
        if doc_id in self.fail_delete:
            raise RuntimeError(f"delete exploded for {doc_id}")
        self.deleted.append(doc_id)
        return SimpleNamespace(status="success")

    async def get_processing_status(self) -> Any:
        if self.fail_statuses:
            raise RuntimeError("status store closed")
        return self.statuses

    async def get_knowledge_graph(self, label: str, *, max_depth: int, max_nodes: int) -> Any:
        self.exports.append((label, max_depth, max_nodes))
        if self.fail_export:
            raise RuntimeError("export exploded")
        return self.knowledge_graph

    async def aquery(self, question: str, *, param: FakeQueryParam) -> Any:
        self.queries.append(param)
        if param.only_need_context:
            if self.fail_context:
                raise RuntimeError("context exploded")
            return self.context
        return self.answer

    async def aquery_data(self, question: str, *, param: FakeQueryParam) -> Any:
        self.data_queries.append(param)
        if self.fail_query_data:
            raise RuntimeError("structured retrieval exploded")
        return self.query_data

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

# `aquery_data`'s envelope, keyed exactly as lightrag 1.5.4 documents it
# (lightrag/lightrag.py:2135-2200): sections under `data`, chunks carrying the
# `file_path` the adapter inserted them under.
LIGHTRAG_QUERY_DATA = {
    "status": "success",
    "message": "Query executed successfully",
    "data": {
        "entities": [
            {
                "entity_name": "Bob",
                "entity_type": "person",
                "description": "a friend",
                "file_path": f"{THREAD}::2016-03-04",
            },
            {"no_name": True},
        ],
        "relationships": [
            {
                "src_id": "Bob",
                "tgt_id": "mechanical keyboard",
                "keywords": "prefers",
                "description": "Bob prefers mechanical keyboards",
                "file_path": f"{THREAD}::2016-03-04",
            },
            {"src_id": "dangling"},
        ],
        "chunks": [
            {
                "content": (
                    "Thread: Hardware Talk (participants: Bob, Me)\n\n"
                    "[2016-03-04 18:22] Bob: I built the PC"
                ),
                "file_path": f"{THREAD}::2016-03-04",
                "chunk_id": "chunk-1",
            },
            {"file_path": f"{THREAD}::2016-03-04", "chunk_id": "chunk-2"},
        ],
        "references": [{"reference_id": "1", "file_path": f"{THREAD}::2016-03-04"}],
    },
    "metadata": {"query_mode": "hybrid"},
}


LIGHTRAG_KNOWLEDGE_GRAPH = SimpleNamespace(
    nodes=[
        SimpleNamespace(
            id="Bob",
            labels=["Bob"],
            properties={"entity_type": "person", "description": "a friend"},
        ),
        SimpleNamespace(id="mechanical keyboard", labels=["mechanical keyboard"], properties={}),
        SimpleNamespace(id="", labels=[], properties={}),  # nameless: dropped
    ],
    edges=[
        SimpleNamespace(
            id="Bob-mechanical keyboard",
            type="DIRECTED",
            source="Bob",
            target="mechanical keyboard",
            properties={"keywords": "prefers", "description": "Bob prefers them"},
        ),
        SimpleNamespace(id="dangling", type="DIRECTED", source="Bob", target=None, properties={}),
    ],
    is_truncated=True,
)


def new_session(tmp_path: Path, *, llm: Any = None, **kwargs: Any) -> Any:
    """An unbuilt session over a fresh workdir."""
    return lightrag_adapter._LightRAGSession(
        FakeRag(**kwargs), FakeQueryParam, workdir=tmp_path, llm=llm
    )


def lightrag_session(tmp_path: Path, *, llm: Any = None, **kwargs: Any) -> Any:
    """A session whose graph already holds one two-message transcript."""
    session = new_session(tmp_path, llm=llm, **kwargs)
    session.build([batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))])
    return session


class TestLightRAGSessionLoop:
    """The threading model that lets a query overtake a multi-day build."""

    def test_calls_run_on_the_sessions_own_thread_not_the_callers(self, tmp_path: Path) -> None:
        """★ Stage-2 decision #10: the loop is not driven by the caller."""
        session = new_session(tmp_path)
        seen: list[int] = []

        async def probe() -> int:
            seen.append(threading.get_ident())
            return 7

        try:
            assert session.run(probe()) == 7
            assert seen == [session._thread.ident]
            assert seen[0] != threading.get_ident()
        finally:
            session.close()

    def test_a_query_lands_while_a_build_is_still_in_flight(self, tmp_path: Path) -> None:
        """★ The point of the loop thread: the graph stays readable mid-backfill."""
        rag = FakeRag(context=LIGHTRAG_CONTEXT)
        released = threading.Event()
        entered = threading.Event()

        async def slow_process() -> None:
            entered.set()
            while not released.is_set():  # noqa: ASYNC110 (a test's hand-rolled gate)
                await asyncio.sleep(0.001)

        rag.apipeline_process_enqueue_documents = slow_process  # type: ignore[method-assign]
        session = lightrag_adapter._LightRAGSession(rag, FakeQueryParam, workdir=tmp_path)
        builder = threading.Thread(target=lambda: session.build([batch(message("g1"))]))
        builder.start()
        try:
            assert entered.wait(timeout=5)
            # The build thread is parked inside the engine; this call is made
            # from the main thread and must still be answered.
            assert session.query("q").answer == "Bob loves mechanical keyboards."
        finally:
            released.set()
            builder.join(timeout=5)
            session.close()

    def test_a_closed_session_refuses_further_work(self, tmp_path: Path) -> None:
        """The loop is gone, so a submitted call could never complete."""
        session = new_session(tmp_path)
        session.close()

        async def probe() -> int:
            return 1  # pragma: no cover - never scheduled

        # The coroutine is closed on the way out, so this raises RuntimeError
        # rather than leaving a "never awaited" warning behind it.
        with pytest.raises(RuntimeError, match="closed"):
            session.run(probe())

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """A service teardown may race the context manager's finally."""
        session = new_session(tmp_path)
        session.close()
        session.close()
        assert session._rag.finalized is True

    def test_close_cancels_the_engines_lingering_loop_tasks(self, tmp_path: Path) -> None:
        """★ ``finalize_storages`` never reaps LightRAG's own worker tasks.

        The library parks priority-queue rate limiters and a health check on
        the loop for the session's lifetime. Closing the loop under them
        spews "Task was destroyed but it is pending!" on every shutdown
        (reproduced in both Phase-2 live runs), so ``close`` must cancel
        whatever is still parked and let it unwind *before* the loop stops.
        """
        session = new_session(tmp_path)
        unwound = threading.Event()

        async def lingering_worker() -> None:
            """Sleep forever, the shape of lightrag's queue workers."""
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                unwound.set()
                raise

        parked: list[asyncio.Task[None]] = []

        async def park() -> None:
            """Start the worker on the session's loop, as the library does."""
            parked.append(asyncio.get_running_loop().create_task(lingering_worker()))

        session.run(park())
        session.close()
        # The worker unwound through CancelledError while the loop was still
        # running — not stranded mid-await by a closed loop.
        assert unwound.wait(timeout=5)
        assert parked[0].cancelled()
        assert session._loop.is_closed()

    def test_a_wedged_loop_thread_is_logged_not_waited_on_forever(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """★ An engine stuck mid-call must not wedge an API shutdown.

        The thread is a daemon, so overrunning the join is survivable; what
        must not happen is a silent hang. Staged with a stand-in thread
        rather than a real wall-clock wedge — the branch is about what
        ``close`` does when the join times out, not about how long it waits.
        """
        session = new_session(tmp_path)
        real_thread = session._thread
        session._thread = SimpleNamespace(  # type: ignore[assignment]
            join=lambda timeout=None: None, is_alive=lambda: True
        )
        with caplog.at_level(logging.WARNING, logger="varagity.graph.engines.lightrag"):
            session.close()
        assert "did not stop" in caplog.text
        # The real loop *was* asked to stop; tidy it up so nothing leaks.
        real_thread.join(timeout=5)
        session._loop.close()


class TestLightRAGBuild:
    def test_build_enqueues_transcripts_under_their_doc_keys_then_processes(
        self, tmp_path: Path
    ) -> None:
        rag = FakeRag()
        session = lightrag_adapter._LightRAGSession(rag, FakeQueryParam, workdir=tmp_path)
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
        assert rag.processed == 1
        assert isinstance(report, BuildReport)
        assert report.messages_seen == 3
        assert report.failures == []
        assert report.wall_clock_s >= 0.0

    def test_a_second_build_over_an_overlapping_batch_sees_the_union(self, tmp_path: Path) -> None:
        rag = FakeRag()
        session = lightrag_adapter._LightRAGSession(rag, FakeQueryParam, workdir=tmp_path)
        first = batch(message("g1"))
        second = batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))
        assert session.build([first, second]).messages_seen == 2
        session.close()

    def test_an_unchanged_corpus_costs_no_extraction_at_all(self, tmp_path: Path) -> None:
        """★ The manifest's first job: skip what is already indexed."""
        session = lightrag_session(tmp_path)
        session._rag.calls.clear()
        report = session.build(
            [batch(message("g1"), message("g2", when=START + timedelta(minutes=1)))]
        )
        session.close()
        assert session._rag.inserted == [
            ([f"{THREAD}::2016-03-04"], [f"{THREAD}::2016-03-04"])
        ]  # only the first build's
        # `process` still runs: it is what picks up anything a previous build
        # left pending or failed, which is the free half of resume.
        assert session._rag.calls == ["process"]
        assert report.failures == []

    def test_a_changed_transcript_is_deleted_before_it_is_re_enqueued(self, tmp_path: Path) -> None:
        """★ The trap: enqueue dedup drops a known doc_id, stale content and all."""
        session = lightrag_session(tmp_path)
        session._rag.calls.clear()
        grown = batch(
            message("g1"),
            message("g2", when=START + timedelta(minutes=1)),
            message("g3", text="and it boots", when=START + timedelta(minutes=2)),
        )
        session.build([grown])
        session.close()
        key = f"{THREAD}::2016-03-04"
        assert session._rag.calls == [f"delete:{key}", f"enqueue:{key}", "process"]
        assert session._rag.deleted == [key]

    def test_a_source_that_vanished_is_deleted_on_a_full_corpus_build(self, tmp_path: Path) -> None:
        session = lightrag_session(tmp_path)
        session._rag.calls.clear()
        session.build([batch(message("g9", thread_id="other", thread_name="Crew"))])
        session.close()
        assert session._rag.deleted == [f"{THREAD}::2016-03-04"]
        assert set(load_manifest(tmp_path).docs) == {"other::2016-03-04"}

    def test_a_bounded_build_never_deletes_what_it_did_not_render(self, tmp_path: Path) -> None:
        """★ Decision #9: a partial render may not speak for the whole archive."""
        session = lightrag_session(tmp_path)
        session._rag.calls.clear()
        session.build(
            [batch(message("g9", thread_id="other", thread_name="Crew"))], prune_removed=False
        )
        session.close()
        assert session._rag.deleted == []
        assert set(load_manifest(tmp_path).docs) == {f"{THREAD}::2016-03-04", "other::2016-03-04"}

    def test_the_manifest_records_content_guids_and_span(self, tmp_path: Path) -> None:
        lightrag_session(tmp_path).close()
        (key, doc) = next(iter(load_manifest(tmp_path).docs.items()))
        assert key == f"{THREAD}::2016-03-04"
        assert doc.message_guids == ["g1", "g2"]
        assert doc.thread_name == "Hardware Talk"
        assert doc.span == "2016-03-04"
        assert len(doc.content_sha256) == 64

    def test_the_summary_sidecar_is_refreshed_with_the_graph_size(self, tmp_path: Path) -> None:
        lightrag_session(tmp_path).close()
        summary = load_summary(tmp_path)
        assert summary is not None
        assert (summary.entities, summary.relations) == (3, 2)
        assert (summary.docs, summary.message_guids) == (1, 2)

    def test_a_failed_enqueue_is_recorded_not_raised(self, tmp_path: Path) -> None:
        session = new_session(tmp_path, fail_insert=True)
        report = session.build([batch(message("g1"))])
        session.close()
        assert report.failures and "enqueue exploded" in report.failures[0]

    def test_a_failed_process_is_recorded_not_raised(self, tmp_path: Path) -> None:
        session = new_session(tmp_path, fail_process=True)
        report = session.build([batch(message("g1"))])
        session.close()
        assert report.failures == [f"process: {RuntimeError('process exploded')!r}"]
        # The manifest still lands: the document was offered, and the engine's
        # own doc-status store is where the next pass retries it.
        assert set(load_manifest(tmp_path).docs) == {f"{THREAD}::2016-03-04"}

    def test_a_document_whose_delete_failed_stays_stale_in_the_manifest(
        self, tmp_path: Path
    ) -> None:
        """★ Otherwise the next build calls it unchanged and the staleness sticks."""
        key = f"{THREAD}::2016-03-04"
        session = lightrag_session(tmp_path)
        before = load_manifest(tmp_path).docs[key].content_sha256
        session._rag.fail_delete = {key}
        grown = batch(
            message("g1"),
            message("g2", when=START + timedelta(minutes=1)),
            message("g3", text="and it boots", when=START + timedelta(minutes=2)),
        )
        session.build([grown])
        session.close()
        assert load_manifest(tmp_path).docs[key].content_sha256 == before
        # …so a retry still sees it as changed and deletes it again.
        retry = new_session(tmp_path)
        retry._rag.calls.clear()
        retry.build([grown])
        retry.close()
        assert retry._rag.deleted == [key]

    def test_a_vanished_source_whose_delete_failed_is_retried_next_build(
        self, tmp_path: Path
    ) -> None:
        key = f"{THREAD}::2016-03-04"
        session = lightrag_session(tmp_path)
        session._rag.fail_delete = {key}
        session.build([batch(message("g9", thread_id="other", thread_name="Crew"))])
        session.close()
        assert set(load_manifest(tmp_path).docs) == {key, "other::2016-03-04"}

    def test_a_failed_delete_is_recorded_and_the_rest_still_index(self, tmp_path: Path) -> None:
        session = lightrag_session(tmp_path)
        session._rag.fail_delete = {f"{THREAD}::2016-03-04"}
        report = session.build(
            [
                batch(
                    message("g1"),
                    message("g2", when=START + timedelta(minutes=1)),
                    message("g3", text="and it boots", when=START + timedelta(minutes=2)),
                )
            ]
        )
        session.close()
        assert report.failures and "delete exploded" in report.failures[0]
        assert session._rag.inserted[-1][0] == [f"{THREAD}::2016-03-04"]

    def test_resume_only_processes_and_reports_the_indexed_corpus(self, tmp_path: Path) -> None:
        """★ A killed build finishes without re-rendering a thing."""
        session = lightrag_session(tmp_path)
        session._rag.calls.clear()
        report = session.resume()
        session.close()
        assert session._rag.calls == ["process"]
        assert report.messages_seen == 2  # what the manifest accounts for
        assert report.failures == []

    def test_resume_records_a_failed_pass(self, tmp_path: Path) -> None:
        session = lightrag_session(tmp_path)
        session._rag.fail_process = True
        report = session.resume()
        session.close()
        assert report.failures and "process exploded" in report.failures[0]

    def test_a_reopened_session_still_resolves_provenance(self, tmp_path: Path) -> None:
        """★ The manifest's second job: provenance outlives the process."""
        lightrag_session(tmp_path).close()
        reopened = lightrag_adapter._LightRAGSession(
            FakeRag(context=LIGHTRAG_CONTEXT), FakeQueryParam, workdir=tmp_path
        )
        answer = reopened.query("q")
        reopened.close()
        assert answer.evidence.message_guids == ["g1", "g2"]

    def test_an_unreadable_manifest_is_treated_as_an_empty_one(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "varagity_manifest.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="varagity.graph.manifest"):
            session = new_session(tmp_path)
        session.close()
        assert session._index == {}
        assert "unreadable" in caplog.text

    def test_a_foreign_schema_version_re_indexes_rather_than_mis_diffing(
        self, tmp_path: Path
    ) -> None:
        """Re-deriving costs a dedup pass; trusting the wrong fields costs the graph."""
        save_manifest(
            tmp_path,
            WorkdirManifest(
                version=99,
                docs={f"{THREAD}::2016-03-04": ManifestDoc(content_sha256="whatever")},
            ),
        )
        session = new_session(tmp_path)
        session.build([batch(message("g1"))])
        session.close()
        assert session._rag.inserted[0][0] == [f"{THREAD}::2016-03-04"]
        assert load_manifest(tmp_path).version == 1


class TestLightRAGProductionSurface:
    def test_document_statuses_pass_the_engines_counts_through(self, tmp_path: Path) -> None:
        session = new_session(tmp_path, statuses={"pending": 3, "processed": 7})
        assert session.document_statuses() == {"pending": 3, "processed": 7}
        session.close()

    @pytest.mark.parametrize("statuses", [None, "not a mapping", 17])
    def test_unreadable_statuses_are_no_news_not_zero_documents(
        self, tmp_path: Path, statuses: Any
    ) -> None:
        session = new_session(tmp_path, statuses=statuses, fail_statuses=statuses is None)
        assert session.document_statuses() == {}
        session.close()

    def test_export_flattens_nodes_edges_and_slice_local_degree(self, tmp_path: Path) -> None:
        session = new_session(tmp_path, knowledge_graph=LIGHTRAG_KNOWLEDGE_GRAPH)
        export = session.export(max_nodes=50)
        session.close()
        assert session._rag.exports == [("*", 3, 50)]
        assert [(node.id, node.entity_type, node.degree) for node in export.nodes] == [
            ("Bob", "person", 1),
            ("mechanical keyboard", None, 1),
        ]  # the nameless node is dropped
        (edge,) = export.edges  # the endpoint-less edge is dropped
        assert (edge.source, edge.target, edge.label) == ("Bob", "mechanical keyboard", "prefers")
        assert export.truncated is True

    def test_export_centres_on_a_named_entity(self, tmp_path: Path) -> None:
        session = new_session(tmp_path, knowledge_graph=LIGHTRAG_KNOWLEDGE_GRAPH)
        session.export("Bob", max_depth=1, max_nodes=10)
        session.close()
        assert session._rag.exports == [("Bob", 1, 10)]

    @pytest.mark.parametrize("graph", [None, "prose", SimpleNamespace(nodes=None, edges=None)])
    def test_an_unusable_export_payload_draws_nothing_rather_than_raising(self, graph: Any) -> None:
        export = lightrag_adapter.export_from_knowledge_graph(graph)
        assert (export.nodes, export.edges, export.truncated) == ([], [], False)

    def test_a_json_round_tripped_export_maps_identically(self, tmp_path: Path) -> None:
        """The engine returns models; a cached/serialized export returns dicts."""
        as_dicts = {
            "nodes": [{"id": "Bob", "properties": {"entity_type": "person"}}],
            "edges": [{"id": "e1", "source": "Bob", "target": "Ada", "properties": {}}],
            "is_truncated": False,
        }
        export = lightrag_adapter.export_from_knowledge_graph(as_dicts)
        assert [node.id for node in export.nodes] == ["Bob"]
        assert export.nodes[0].entity_type == "person"
        assert [edge.target for edge in export.edges] == ["Ada"]

    def test_a_failed_export_draws_nothing_rather_than_raising(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = new_session(tmp_path, fail_export=True)
        with caplog.at_level(logging.WARNING, logger="varagity.graph.engines.lightrag"):
            export = session.export()
        session.close()
        assert export.nodes == []
        assert "export failed" in caplog.text

    def test_deleting_documents_drops_them_from_the_graph_and_the_manifest(
        self, tmp_path: Path
    ) -> None:
        session = lightrag_session(tmp_path)
        key = f"{THREAD}::2016-03-04"
        assert session.delete_documents([key]) == 1
        session.close()
        assert session._rag.deleted == [key]
        assert load_manifest(tmp_path).docs == {}
        assert session._index == {}

    def test_a_failed_delete_is_not_counted_and_keeps_its_manifest_record(
        self, tmp_path: Path
    ) -> None:
        """A document still in the graph must still be in the manifest."""
        session = lightrag_session(tmp_path)
        key = f"{THREAD}::2016-03-04"
        session._rag.fail_delete = {key}
        assert session.delete_documents([key]) == 0
        session.close()
        assert set(load_manifest(tmp_path).docs) == {key}


class TestLightRAGAdapter:
    def test_query_answers_with_normalized_evidence_and_provenance(self, tmp_path: Path) -> None:
        session = lightrag_session(tmp_path, context=LIGHTRAG_CONTEXT)
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

    def test_an_explicit_mode_overrides_the_primary(self, tmp_path: Path) -> None:
        session = lightrag_session(tmp_path)
        assert session.query("q", mode="global").mode == "global"
        session.close()

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("hybrid", ("hybrid", False)),
            ("global", ("global", False)),
            ("hybrid+synthesis", ("hybrid", True)),
            ("global+synthesis", ("global", True)),
            ("synthesis", ("", True)),  # "" = the session's own primary
            ("+synthesis", ("", True)),
            ("  mix+synthesis  ", ("mix", True)),
        ],
    )
    def test_the_synthesis_suffix_splits_off_the_base_mode(
        self, mode: str, expected: tuple[str, bool]
    ) -> None:
        assert lightrag_adapter._split_mode(mode) == expected

    def test_a_synthesis_mode_retrieves_once_and_writes_the_answer_itself(
        self, tmp_path: Path
    ) -> None:
        """★ ADR-017's retrieval-only design: no engine answer call at all."""
        llm = ScriptedLLM()
        session = lightrag_session(tmp_path, query_data=LIGHTRAG_QUERY_DATA, llm=llm)
        answer = session.query("What does Bob think about computers?", mode="hybrid+synthesis")
        session.close()
        assert answer.answer == "Bob prefers mechanical keyboards."
        assert answer.mode == "hybrid+synthesis"  # the full string is what ran
        assert session._rag.queries == []  # aquery is never touched
        assert [param.mode for param in session._rag.data_queries] == ["hybrid"]
        assert [e.name for e in answer.evidence.entities] == ["Bob"]
        assert answer.evidence.message_guids == ["g1", "g2"]

    def test_the_synthesis_prompt_grounds_on_facts_and_transcript_excerpts(
        self, tmp_path: Path
    ) -> None:
        """★ Decision #6: measure the diet the shipped path will actually see."""
        llm = ScriptedLLM()
        session = lightrag_session(tmp_path, query_data=LIGHTRAG_QUERY_DATA, llm=llm)
        session.query("What does Bob think about computers?", mode="hybrid+synthesis")
        session.close()
        assert "- Bob prefers mechanical keyboards" in llm.prompt
        assert "[Hardware Talk (2016-03-04)]" in llm.prompt
        assert "I built the PC" in llm.prompt
        assert "What does Bob think about computers?" in llm.prompt

    def test_a_bare_synthesis_mode_retrieves_with_the_sessions_primary(
        self, tmp_path: Path
    ) -> None:
        session = lightrag_session(tmp_path, query_data=LIGHTRAG_QUERY_DATA, llm=ScriptedLLM())
        answer = session.query("q", mode="synthesis")
        session.close()
        assert [param.mode for param in session._rag.data_queries] == [
            lightrag_adapter.PRIMARY_MODE
        ]
        assert answer.mode == "synthesis"

    def test_an_unsuffixed_mode_keeps_the_engines_own_answer_pipeline(self, tmp_path: Path) -> None:
        """★ The bake-off numbers stay reproducible: the ADR-017 tables are this path."""
        session = lightrag_session(
            tmp_path, context=LIGHTRAG_CONTEXT, query_data=LIGHTRAG_QUERY_DATA
        )
        answer = session.query("q", mode="global")
        session.close()
        assert answer.answer == "Bob loves mechanical keyboards."
        assert session._rag.data_queries == []
        assert [param.mode for param in session._rag.queries] == ["global", "global"]

    def test_the_chat_client_is_built_once_and_only_when_synthesis_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An engine-composed session never opens a chat connection at all."""
        built: list[object] = []

        def fake_client() -> Any:
            built.append(object())
            return ScriptedLLM()

        monkeypatch.setattr(lightrag_adapter, "LLMClient", fake_client)
        session = lightrag_session(
            tmp_path, context=LIGHTRAG_CONTEXT, query_data=LIGHTRAG_QUERY_DATA
        )
        session.query("q")  # unsuffixed: the engine answers
        assert built == []
        session.query("q", mode="hybrid+synthesis")
        session.query("q", mode="hybrid+synthesis")
        session.close()
        assert len(built) == 1

    def test_a_failed_structured_retrieval_still_produces_a_scored_answer(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = ScriptedLLM()
        session = lightrag_session(tmp_path, fail_query_data=True, llm=llm)
        with caplog.at_level(logging.WARNING, logger="varagity.graph.engines.lightrag"):
            answer = session.query("q", mode="hybrid+synthesis")
        session.close()
        assert answer.answer == graph_answer.NO_EVIDENCE_ANSWER
        assert answer.evidence == GraphEvidence()
        assert llm.calls == []  # nothing to ground on, so no call is spent
        assert "structured retrieval failed" in caplog.text

    def test_a_reasoning_stage_never_reaches_the_answer(self, tmp_path: Path) -> None:
        session = lightrag_session(tmp_path, answer="<think>hmm</think>Bob likes ARM.")
        assert session.query("q").answer == "Bob likes ARM."
        session.close()

    def test_failed_context_retrieval_still_answers(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = lightrag_session(tmp_path, fail_context=True)
        with caplog.at_level(logging.WARNING, logger="varagity.graph.engines.lightrag"):
            answer = session.query("q")
        session.close()
        assert answer.answer
        assert answer.evidence == GraphEvidence()
        assert "context retrieval failed" in caplog.text

    def test_stats_come_from_the_summary_sidecar_a_build_refreshed(self, tmp_path: Path) -> None:
        session = lightrag_session(tmp_path)
        # Break the graph walk to prove the sidecar answered, not the graph.
        session._rag.chunk_entity_relation_graph = FakeGraphStore(fail=True)
        assert session.stats() == GraphStats(entities=3, relations=2, communities=None)
        session.close()

    def test_stats_fall_back_to_the_graph_when_there_is_no_sidecar(self, tmp_path: Path) -> None:
        """A workdir built before the sidecar existed still reports honestly."""
        session = new_session(tmp_path)
        assert session.stats() == GraphStats(entities=3, relations=2, communities=None)
        session.close()

    def test_unavailable_stats_are_reported_as_unknown_not_zero(self, tmp_path: Path) -> None:
        session = new_session(tmp_path)
        session._rag.chunk_entity_relation_graph = FakeGraphStore(fail=True)
        assert session.stats() == GraphStats(entities=None, relations=None, communities=None)
        session.close()

    def test_close_finalizes_storages_and_survives_a_failed_teardown(self, tmp_path: Path) -> None:
        rag = FakeRag()
        lightrag_adapter._LightRAGSession(rag, FakeQueryParam, workdir=tmp_path).close()
        assert rag.finalized is True
        lightrag_adapter._LightRAGSession(
            FakeRag(fail_finalize=True), FakeQueryParam, workdir=tmp_path
        ).close()

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

    def test_structured_retrieval_maps_entities_relations_and_excerpts(self) -> None:
        """★ The production evidence path: structured end to end, no re-parsing."""
        index = {f"{THREAD}::2016-03-04": ["g1", "g2"]}
        evidence, excerpts = lightrag_adapter.retrieval_from_query_data(LIGHTRAG_QUERY_DATA, index)
        assert [e.name for e in evidence.entities] == ["Bob"]
        assert evidence.relations[0].target == "mechanical keyboard"
        assert evidence.communities == []  # LightRAG has no community layer
        assert evidence.message_guids == ["g1", "g2"]
        (excerpt,) = excerpts  # the second chunk carries no text
        assert excerpt.doc_key == f"{THREAD}::2016-03-04"
        assert excerpt.span == "2016-03-04"
        assert excerpt.message_guids == ["g1", "g2"]
        assert "I built the PC" in excerpt.text

    def test_an_excerpts_thread_label_is_read_off_the_transcript_header(self) -> None:
        """The doc_key only carries the thread *id*; the text carries its name."""
        _, (excerpt,) = lightrag_adapter.retrieval_from_query_data(LIGHTRAG_QUERY_DATA, {})
        assert excerpt.thread_name == "Hardware Talk"

    def test_a_headerless_excerpt_falls_back_to_its_thread_id(self) -> None:
        payload = {
            "data": {
                "chunks": [{"content": "[18:25] Me: nice", "file_path": f"{THREAD}::2016-03-04"}]
            }
        }
        _, (excerpt,) = lightrag_adapter.retrieval_from_query_data(payload, {})
        assert excerpt.thread_name == THREAD
        assert excerpt.span == "2016-03-04"

    def test_a_chunk_without_text_cannot_ground_anything(self) -> None:
        payload = {"data": {"chunks": [{"file_path": f"{THREAD}::2016-03-04"}]}}
        _, excerpts = lightrag_adapter.retrieval_from_query_data(payload, {})
        assert excerpts == []

    def test_sections_are_read_at_the_top_level_when_there_is_no_envelope(self) -> None:
        evidence, _ = lightrag_adapter.retrieval_from_query_data(
            {"entities": [{"entity_name": "Jane"}]}, {}
        )
        assert [e.name for e in evidence.entities] == ["Jane"]

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "not json at all",
            "[]",
            {"data": {}},
            {"data": "nope"},
            {"status": "failure", "message": "Query returned no results", "data": {}},
            {"data": {"entities": "not a list", "chunks": [1, 2]}},
        ],
    )
    def test_unknown_structured_shapes_degrade_instead_of_raising(self, payload: Any) -> None:
        evidence, excerpts = lightrag_adapter.retrieval_from_query_data(payload, {})
        assert evidence.entities == []
        assert evidence.relations == []
        assert evidence.message_guids == []
        assert excerpts == []

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
        """The default: unprefixed, exactly as the ADR-017 bake-off ran."""
        client = self.client("unused")
        embed = lightrag_adapter.make_embedding_func(client, model="e5")
        assert self.run(embed(["a passage"])) == [[0.5, 0.25]]
        assert client.embeddings.calls[0]["input"] == ["a passage"]

    def test_the_prefix_setting_off_ignores_the_side_lightrag_declared(self) -> None:
        client = self.client("unused")
        embed = lightrag_adapter.make_embedding_func(client, model="e5")
        self.run(embed(["what did bob say"], context="query"))
        assert client.embeddings.calls[0]["input"] == ["what did bob say"]

    def test_queries_are_e5_instruction_wrapped_when_the_prefix_is_on(self) -> None:
        """★ The asymmetric seam: only the query side, only when asked."""
        client = self.client("unused")
        embed = lightrag_adapter.make_embedding_func(client, model="e5", query_prefix=True)
        assert self.run(embed(["what did bob say"], context="query")) == [[0.5, 0.25]]
        assert client.embeddings.calls[0]["input"] == [format_query("what did bob say")]

    @pytest.mark.parametrize("context", [None, "document"])
    def test_passages_are_never_prefixed_even_with_the_setting_on(
        self, context: str | None
    ) -> None:
        """e5 requires documents unprefixed — which is why no re-embed is needed."""
        client = self.client("unused")
        embed = lightrag_adapter.make_embedding_func(client, model="e5", query_prefix=True)
        self.run(embed(["a passage"], context=context))
        assert client.embeddings.calls[0]["input"] == ["a passage"]


class TestSessionProtocol:
    def test_the_adapter_session_implements_the_whole_protocol(self, tmp_path: Path) -> None:
        """Structural conformance — every caller drives the seam, not the adapter.

        The protocol widened at stage 2 (resume/export/delete/statuses), so
        this is what tells a *re-entering* engine what it owes: the harness,
        the build runner, the query path, and the graph view all reach an
        adapter only through these names.
        """
        session = new_session(tmp_path)
        try:
            for method in GraphSession.__protocol_attrs__:
                assert callable(getattr(session, method)), method
        finally:
            session.close()

    def test_the_protocol_names_the_whole_production_surface(self) -> None:
        assert GraphSession.__protocol_attrs__ == {
            "build",
            "resume",
            "query",
            "stats",
            "document_statuses",
            "export",
            "delete_documents",
        }
