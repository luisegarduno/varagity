"""Unit tests for the Prefect flow adapters (spec §9/§10).

Flows execute for real under ``prefect_test_harness`` — an ephemeral API
with a temporary database — with every service seam stubbed (no Docker, no
GPU). The harness is module-scoped: it takes several seconds to boot and
every test here shares it. Stage *logic* is covered by ``test_loader.py``
and friends; these tests cover the flow layer — stage→task wiring, tracked
task runs, retry configuration, and the §10.1 state dict.
"""

import dataclasses
import importlib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from prefect.cache_policies import NO_CACHE
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter
from prefect.testing.utilities import prefect_test_harness
from prometheus_client import REGISTRY

from tests.unit.test_backup_graph import build_workdir
from tests.unit.test_backup_rag import FakeRebuildBM25, FakeRestoreStore, rezip
from tests.unit.test_backup_rag import FakeStore as FakeCorpusStore
from tests.unit.test_backup_rag import chunk as backup_chunk
from tests.unit.test_backup_rag import document as backup_document
from tests.unit.test_graph_service import FAKE_FORMAT, FakeEngine
from tests.unit.test_loader import FakeBM25, FakeEmbeddings, FakeStore
from varagity.chat import PreparedQuery, Turn
from varagity.graph.base import GRAPH_ENGINE_REGISTRY
from varagity.graph.records import BuildReport
from varagity.graph.service import GraphService
from varagity.pipeline import ingest_flow, query_flow
from varagity.pipeline.ingest_flow import (
    chunk_document_task,
    contextualize_chunks_task,
    discover_documents_task,
    embed_chunks_task,
    parse_document_task,
    store_chunks_task,
)
from varagity.pipeline.query_flow import (
    condense_query_task,
    embed_query_task,
    generate_answer_task,
    retrieve_task,
)
from varagity.stores.records import RetrievedChunk

# The submodule itself — the package attribute of the same name is rebound
# to the Flow object by the package's re-export, so plain `import … as`
# (which resolves via the parent attribute) would grab the Flow instead.
ingest_flow_module = importlib.import_module("varagity.pipeline.ingest_flow")

# The spec §9 stage names, exactly as registered on the task wrappers.
INGEST_STAGE_NAMES = {
    "discover_documents",
    "parse_document",
    "chunk_document",
    "contextualize_chunks",
    "embed_chunks",
    "store_chunks",
}


@pytest.fixture(scope="module", autouse=True)
def prefect_harness() -> Iterator[None]:
    """Run every test in this module against an ephemeral Prefect API."""
    with prefect_test_harness():
        yield


@pytest.fixture
def pinned_settings(settings_env: Callable[..., None]) -> None:
    """Hermetic pipeline settings (identity path, no machine .env leakage)."""
    settings_env(
        ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        CHUNKING_STRATEGY="recursive_character",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        CONTEXTUALIZE="false",
        EMBEDDING_MODEL="test-model",
        TOP_K=10,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "aurora.md").write_text(
        "# Aurora\n\nThe Aurora station sits at 2,847 meters and is powered by a "
        "thorium micro-reactor nicknamed Lantern producing 4.2 megawatts at peak."
    )
    (root / "tidal.txt").write_text(
        "The Corvo Tidal Grid consists of 87 underwater turbines in three arrays "
        "named Alba, Bruma, and Cinza, delivering 310 megawatt-hours per day."
    )
    return root


def _task_run_names(flow_run_id: UUID) -> list[str]:
    """Read the base task names tracked for a flow run from the test API."""
    with get_client(sync_client=True) as client:
        runs = client.read_task_runs(flow_run_filter=FlowRunFilter(id={"any_": [flow_run_id]}))
    # Task run names are "<task name>-<slug>".
    return [run.name.rsplit("-", 1)[0] for run in runs]


class TestIngestFlow:
    def test_every_stage_is_a_tracked_task_run(self, pinned_settings: None, corpus: Path) -> None:
        """★ Each §9 stage of each file → a task run."""
        store, bm25 = FakeStore(), FakeBM25()
        state = ingest_flow(
            str(corpus),
            store=store,
            bm25=bm25,
            embeddings=FakeEmbeddings(),
            verbose=0,
            return_state=True,
        )
        assert state.is_completed()
        summary = state.result()

        # Loader semantics are untouched by the flow shell.
        assert summary.discovered == 2
        assert summary.ingested == 2
        assert summary.failed == 0
        assert summary.chunks == len(store.records) > 0
        assert [r.chunk_id for r in bm25.indexed] == [r.chunk_id for r in store.records]
        assert all(record.context is None for record in store.records)  # identity path

        # Every stage ran as a tracked task run: discovery once, the five
        # per-file stages once per file.
        names = _task_run_names(state.state_details.flow_run_id)
        assert set(names) == INGEST_STAGE_NAMES
        assert names.count("discover_documents") == 1
        for per_file in INGEST_STAGE_NAMES - {"discover_documents"}:
            assert names.count(per_file) == 2, f"expected one {per_file} task run per file"

    def test_failing_store_stage_fails_that_file_only(
        self, pinned_settings: None, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stage failure is contained per file, and retries actually re-run.

        The store stage's retry delays are zeroed via ``with_options`` (the
        real backoff values are asserted separately) so the unit suite stays
        fast; retry *behavior* — the task re-runs after a failure and the
        flow continues — is exercised for real.
        """

        class FlakyBM25(FakeBM25):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            def index_chunks(self, records: list) -> int:  # type: ignore[override]
                self.attempts += 1
                if self.attempts <= 2:  # fail the first file's write twice…
                    raise RuntimeError("elasticsearch restarting")
                return super().index_chunks(records)  # …retries then succeed

        fast_stages = dataclasses.replace(
            ingest_flow_module.TASK_STAGES,
            store=store_chunks_task.with_options(retry_delay_seconds=0),
        )
        monkeypatch.setattr(ingest_flow_module, "TASK_STAGES", fast_stages)

        store, bm25 = FakeStore(), FlakyBM25()
        summary = ingest_flow(
            str(corpus), store=store, bm25=bm25, embeddings=FakeEmbeddings(), verbose=0
        )

        # retries=2 → three attempts for the first file: fail, fail, succeed;
        # the second file's single attempt makes four.
        assert bm25.attempts == 4
        assert summary.ingested == 2
        assert summary.failed == 0

    def test_exhausted_retries_fail_loudly_but_not_the_run(
        self, pinned_settings: None, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hard store failure is counted per file; the corpus run continues."""

        class ExplodingBM25(FakeBM25):
            def index_chunks(self, records: list) -> int:  # type: ignore[override]
                raise RuntimeError("elasticsearch is gone")

        fast_stages = dataclasses.replace(
            ingest_flow_module.TASK_STAGES,
            store=store_chunks_task.with_options(retry_delay_seconds=0),
        )
        monkeypatch.setattr(ingest_flow_module, "TASK_STAGES", fast_stages)

        store = FakeStore()
        summary = ingest_flow(
            str(corpus), store=store, bm25=ExplodingBM25(), embeddings=FakeEmbeddings(), verbose=0
        )
        assert summary.failed == 2
        assert summary.ingested == 0
        assert store.documents == {}  # no idempotency marker → re-attempted next run

    def test_model_and_store_tasks_carry_retry_config(self) -> None:
        """retries=2 + exponential backoff on model/store stages only."""
        for retrying in (contextualize_chunks_task, embed_chunks_task, store_chunks_task):
            assert retrying.retries == 2, retrying.name
            assert retrying.retry_delay_seconds == [2, 4], retrying.name  # exponential
        for local in (discover_documents_task, parse_document_task, chunk_document_task):
            assert not local.retries, local.name


class FakeRetriever:
    """Duck-typed retriever recording how the flow drives the seam."""

    def __init__(self, chunks: list[RetrievedChunk], vector: list[float] | None) -> None:
        self.chunks = chunks
        self.vector = vector
        self.encode_calls: list[str] = []
        self.retrieve_calls: list[dict] = []

    def encode_query(self, query: str, verbose: int | None = None) -> list[float] | None:
        self.encode_calls.append(query)
        return self.vector

    def retrieve(
        self,
        query: str,
        k: int,
        verbose: int | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        self.retrieve_calls.append({"query": query, "k": k, "query_vector": query_vector})
        return self.chunks


class ScriptedLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: int | None = None,
    ) -> str:
        self.prompts.append(messages[0]["content"])
        return self.response


def _chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="docaaa000000000a::0",
        doc_id="docaaa000000000a",
        original_index=0,
        content=content,
        context=None,
        metadata={"source": "/docs/corpus/a.md", "file_name": "a.md"},
        score=0.91,
    )


class TestQueryFlow:
    def test_state_dict_and_stage_wiring(self, pinned_settings: None) -> None:
        """Embed → retrieve → generate: vector reuse, hook order, §10.1 state."""
        vector = [0.25, 0.75]
        retriever = FakeRetriever([_chunk("Lantern produces 4.2 megawatts.")], vector)
        llm = ScriptedLLM("<think>…</think>Lantern powers Aurora. [SOURCE]: a.md")
        events: list[str] = []

        state = query_flow(
            "What powers Aurora?",
            retriever=retriever,
            llm=llm,
            verbose=0,
            on_retrieved=lambda chunks: events.append(f"hook:{len(chunks)}"),
        )

        # The query was encoded exactly once (its own stage) and the vector
        # was handed to retrieval instead of being re-encoded there.
        assert retriever.encode_calls == ["What powers Aurora?"]
        assert retriever.retrieve_calls == [
            {"query": "What powers Aurora?", "k": 10, "query_vector": vector}
        ]
        # The hook fired before generation (spec §10.1 step 4 → 5).
        assert events == ["hook:1"]
        assert len(llm.prompts) == 1
        assert "using ONLY the CONTEXT" in llm.prompts[0]
        assert "Lantern produces 4.2 megawatts." in llm.prompts[0]

        assert state["query"] == "What powers Aurora?"
        assert state["query_vector"] == vector  # the flow records it
        assert state["retrieved"] == retriever.chunks
        assert state["formatted_context"] in llm.prompts[0]
        assert state["answer"] == "Lantern powers Aurora. [SOURCE]: a.md"  # think-stripped
        # The default (simple) engine states the identity split (spec_v3 §4.2).
        assert state["prepared"].search_query == "What powers Aurora?"
        assert state["prepared"].original_query == "What powers Aurora?"
        assert state["prepared"].condensed is False

    def test_bm25_style_retriever_yields_no_query_vector(self, pinned_settings: None) -> None:
        retriever = FakeRetriever([_chunk("Pelican-9 hauls 40 tons.")], vector=None)
        state = query_flow(
            "Pelican-9 capacity?", retriever=retriever, llm=ScriptedLLM("40 tons."), verbose=0
        )
        assert state["query_vector"] is None
        assert retriever.retrieve_calls[0]["query_vector"] is None
        assert state["answer"] == "40 tons."

    def test_each_query_stage_is_a_tracked_task_run(self, pinned_settings: None) -> None:
        retriever = FakeRetriever([_chunk("Lantern.")], vector=[0.5])
        state = query_flow(
            "q?", retriever=retriever, llm=ScriptedLLM("A."), verbose=0, return_state=True
        )
        assert state.is_completed()
        names = _task_run_names(state.state_details.flow_run_id)
        # condense is always in the graph, simple engine included (v3 #14).
        assert sorted(names) == ["condense_query", "embed_query", "generate_answer", "retrieve"]

    def test_engine_seam_splits_search_and_answer_queries(self, pinned_settings: None) -> None:
        """★ spec_v3 §4.2: the condensed string retrieves, the original answers."""

        class RecordingEngine:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def prepare(
                self, query: str, *, history: Sequence[Turn], llm: object, verbose: int
            ) -> PreparedQuery:
                self.calls.append({"query": query, "history": tuple(history), "llm": llm})
                return PreparedQuery(
                    search_query="standalone kelp corridor length",
                    original_query=query,
                    condensed=True,
                    condense_latency_s=0.01,
                )

        vector = [0.5, 0.5]
        retriever = FakeRetriever([_chunk("The corridor is 1.8 km.")], vector)
        llm = ScriptedLLM("1.8 km.")
        engine = RecordingEngine()
        history = (Turn("user", "Tell me about the corridor"), Turn("assistant", "It exists."))

        state = query_flow(
            "how long is it?",
            history=history,
            engine=engine,
            retriever=retriever,
            llm=llm,
            verbose=0,
        )

        # The engine saw the turn, its history, and the flow's LLM.
        assert engine.calls == [{"query": "how long is it?", "history": history, "llm": llm}]
        # The condensed string drove BOTH retrieval arms (embed + retrieve)…
        assert retriever.encode_calls == ["standalone kelp corridor length"]
        assert retriever.retrieve_calls[0]["query"] == "standalone kelp corridor length"
        # …while the answer prompt got the user's words, verbatim.
        assert "QUESTION: how long is it?" in llm.prompts[0]
        assert "standalone kelp corridor length" not in llm.prompts[0]
        assert state["query"] == "how long is it?"
        assert state["prepared"].condensed is True
        assert state["prepared"].search_query == "standalone kelp corridor length"

    def test_interactive_tasks_carry_no_retries(self) -> None:
        """The query path leaves retrying to the clients' tenacity layer."""
        for interactive in (
            condense_query_task,
            embed_query_task,
            retrieve_task,
            generate_answer_task,
        ):
            assert not interactive.retries, interactive.name

    def test_condense_task_disables_result_caching(self) -> None:
        """NO_CACHE, like every pipeline task (live inputs, unhashable args)."""
        assert condense_query_task.cache_policy is NO_CACHE

    def test_invalid_verbose_raises(self, pinned_settings: None) -> None:
        with pytest.raises(ValueError, match="verbose"):
            query_flow("q?", retriever=FakeRetriever([], None), llm=ScriptedLLM("A."), verbose=9)


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Read one sample from the process-wide registry, defaulting to 0."""
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else value


class TestFlowMetrics:
    """The flows are Prometheus probe points (spec_v2 §6.2).

    Injected fakes aren't registry members, so the method label is
    ``custom`` — the low-cardinality fallback — which conveniently keeps
    these deltas isolated from any registry-method observations.
    """

    def test_stubbed_query_increments_the_catalog(self, pinned_settings: None) -> None:
        retriever = FakeRetriever([_chunk("Lantern produces 4.2 megawatts.")], vector=[0.5])

        def stage(s: str) -> dict[str, str]:
            return {"stage": s, "method": "custom"}

        ok = {"method": "custom", "outcome": "ok"}
        rank1 = {"method": "custom", "rank": "1"}
        before_stages = {
            s: _sample("varagity_query_latency_seconds_count", stage(s))
            for s in ("condense", "embed", "retrieve", "generate")
        }
        before_ok = _sample("varagity_query_total", ok)
        before_score = _sample("varagity_retrieval_score_count", rank1)

        query_flow("What powers Aurora?", retriever=retriever, llm=ScriptedLLM("A."), verbose=0)

        for s, before in before_stages.items():
            assert _sample("varagity_query_latency_seconds_count", stage(s)) == before + 1, (
                f"stage {s} not observed"
            )
        assert _sample("varagity_query_total", ok) == before_ok + 1
        assert _sample("varagity_retrieval_score_count", rank1) == before_score + 1

    def test_failing_query_counts_an_error_outcome(self, pinned_settings: None) -> None:
        class ExplodingRetriever(FakeRetriever):
            def retrieve(self, *args: object, **kwargs: object) -> list[RetrievedChunk]:
                raise RuntimeError("stores are gone")

        labels = {"method": "custom", "outcome": "error"}
        before = _sample("varagity_query_total", labels)

        with pytest.raises(RuntimeError, match="stores are gone"):
            query_flow(
                "q?",
                retriever=ExplodingRetriever([], vector=None),
                llm=ScriptedLLM("A."),
                verbose=0,
            )

        assert _sample("varagity_query_total", labels) == before + 1

    def test_stubbed_ingest_increments_the_catalog(
        self, pinned_settings: None, corpus: Path, settings_env: Callable[..., None]
    ) -> None:
        """Contextualized ingest: doc/chunk counters + the blurb histogram."""
        settings_env(CONTEXTUALIZE="true")
        md = {"file_type": "md", "extraction": "text"}
        txt = {"file_type": "txt", "extraction": "text"}
        strategy = {"chunking_strategy": "recursive_character"}
        before_md = _sample("varagity_ingest_docs_total", md)
        before_txt = _sample("varagity_ingest_docs_total", txt)
        before_chunks = _sample("varagity_ingest_chunks_total", strategy)
        before_ctx = _sample("varagity_contextualize_latency_seconds_count")

        summary = ingest_flow(
            str(corpus),
            store=FakeStore(),
            bm25=FakeBM25(),
            embeddings=FakeEmbeddings(),
            llm=ScriptedLLM("situating blurb"),
            verbose=0,
        )

        assert summary.ingested == 2
        assert _sample("varagity_ingest_docs_total", md) == before_md + 1
        assert _sample("varagity_ingest_docs_total", txt) == before_txt + 1
        assert _sample("varagity_ingest_chunks_total", strategy) == before_chunks + summary.chunks
        # One blurb-latency observation per contextualized document.
        assert _sample("varagity_contextualize_latency_seconds_count") == before_ctx + 2

    def test_identity_path_ingest_skips_the_contextualize_histogram(
        self, pinned_settings: None, corpus: Path
    ) -> None:
        before_ctx = _sample("varagity_contextualize_latency_seconds_count")
        ingest_flow(
            str(corpus), store=FakeStore(), bm25=FakeBM25(), embeddings=FakeEmbeddings(), verbose=0
        )
        assert _sample("varagity_contextualize_latency_seconds_count") == before_ctx


class TestEvalFlows:
    """The eval flows delegate to the harness with the tracked ingest."""

    def test_eval_flow_passes_the_tracked_ingest_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same package-attribute shadowing as ingest_flow_module above.
        eval_flow_module = importlib.import_module("varagity.pipeline.eval_flow")
        from varagity.pipeline import eval_flow

        captured: dict[str, object] = {}

        def fake_run_matrix(**kwargs: object) -> dict[str, str]:
            captured.update(kwargs)
            return {"kind": "retrieval_matrix"}

        monkeypatch.setattr(eval_flow_module, "run_matrix", fake_run_matrix)
        result = eval_flow(verbose=0)

        assert result == {"kind": "retrieval_matrix"}
        assert captured["ingest"] is ingest_flow  # eval ingests are tracked subflows
        assert captured["verbose"] == 0

    def test_graph_eval_flow_forwards_its_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The graph bake-off has no ingest seam — engines self-store."""
        eval_flow_module = importlib.import_module("varagity.pipeline.eval_flow")
        from varagity.pipeline import graph_eval_flow

        captured: dict[str, object] = {}

        def fake_run_graph_eval(**kwargs: object) -> dict[str, str]:
            captured.update(kwargs)
            return {"kind": "graph_eval"}

        monkeypatch.setattr(eval_flow_module, "run_graph_eval", fake_run_graph_eval)
        result = graph_eval_flow(
            profile="full",
            engines=["lightrag"],
            mode="global",
            skip_build=True,
            message_target=1000,
            verbose=2,
        )

        assert result == {"kind": "graph_eval"}
        assert captured == {
            "profile": "full",
            "engines": ["lightrag"],
            "mode": "global",
            "skip_build": True,
            "message_target": 1000,
            "verbose": 2,
        }

    def test_ocr_benchmark_flow_passes_the_tracked_ingest_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eval_flow_module = importlib.import_module("varagity.pipeline.eval_flow")
        from varagity.pipeline import ocr_benchmark_flow

        captured: dict[str, object] = {}

        def fake_benchmark(**kwargs: object) -> dict[str, str]:
            captured.update(kwargs)
            return {"kind": "ocr_benchmark"}

        monkeypatch.setattr(eval_flow_module, "run_ocr_benchmark", fake_benchmark)
        result = ocr_benchmark_flow(verbose=1)

        assert result == {"kind": "ocr_benchmark"}
        assert captured["ingest"] is ingest_flow
        assert captured["verbose"] == 1


class TestGraphBuildFlow:
    """The graph build's tracking shell — the service does the work."""

    class FakeService:
        """Records what the flow handed the graph service."""

        def __init__(self) -> None:
            self.calls: list[tuple[int, bool, int]] = []

        def build(
            self,
            batches: Sequence[object],
            *,
            verbose: int = 0,
            prune_removed: bool = True,
        ) -> BuildReport:
            self.calls.append((len(batches), prune_removed, verbose))
            return BuildReport(messages_seen=7, wall_clock_s=2.5, failures=["one doc refused"])

    def test_the_flow_forwards_to_the_service_and_is_a_tracked_run(self) -> None:
        from varagity.pipeline import GraphBatches, graph_build_flow

        service = self.FakeService()
        report = graph_build_flow(
            service, GraphBatches([object(), object()]), prune_removed=False, verbose=2
        )

        assert report.messages_seen == 7
        assert service.calls == [(2, False, 2)]
        assert graph_build_flow.name == "graph-build"

    def test_an_archive_never_serializes_into_the_flow_run_record(self) -> None:
        """★ Prefect's server caps flow-run parameters at 512 KB.

        Bare batches 422 a real backfill at flow-run creation — after
        ``reingest`` has already wiped the graph.
        """
        import json
        from datetime import UTC, datetime, timedelta

        from varagity.graph.sources.base import MessageBatch, SourceMessage
        from varagity.pipeline import GraphBatches, graph_build_flow

        first = datetime(2026, 1, 1, tzinfo=UTC)
        messages = [
            SourceMessage(
                guid=f"guid-{index}",
                thread_id="any;-;+15550000000",
                thread_name="+15550000000",
                sender_handle="+15550000000",
                sender_name="Bob",
                is_from_me=False,
                timestamp=first + timedelta(minutes=index),
                text="x" * 200,
            )
            for index in range(2000)
        ]
        boxed = GraphBatches(
            [MessageBatch(doc_id="doc-sms", relative_path="sms.db", messages=messages)]
        )
        # The payload itself is comfortably over the server's cap …
        assert len(boxed.batches[0].model_dump_json().encode()) > 524_288
        serialized = graph_build_flow.serialize_parameters(
            {"service": object(), "batches": boxed, "prune_removed": False, "verbose": 0}
        )
        # … but the recorded parameters carry only the opaque placeholder.
        assert serialized["batches"] == "<GraphBatches>"
        assert len(json.dumps(serialized).encode()) < 4_096

    def test_the_flow_carries_no_retries(self) -> None:
        """★ A failed backfill must surface, not silently re-run hours of extraction."""
        from varagity.pipeline import graph_build_flow

        assert graph_build_flow.retries in (0, None)


class TestRagExportFlow:
    """The RAG export's tracking shell — the backup package does the work (ADR-019)."""

    @pytest.fixture
    def docs_root(self, tmp_path: Path, settings_env: Callable[..., None]) -> Path:
        root = tmp_path / "docs"
        root.mkdir()
        settings_env(DOCS_PATH=str(root), EMBEDDING_MODEL="test-model", EMBEDDING_DIM=2)
        return root

    @staticmethod
    def corpus(docs_root: Path) -> "FakeCorpusStore":
        return FakeCorpusStore(
            [backup_document("doc-a", str(docs_root / "a.md"), 1)],
            [backup_chunk("doc-a", 0, 0, [0.5, 0.25])],
        )

    def test_rag_export_is_a_tracked_flow_run(self, docs_root: Path) -> None:
        from varagity.backup.bundle import read_manifest
        from varagity.pipeline import rag_export_flow

        frames: list[str] = []
        state = rag_export_flow(
            docs_root / "varagity-exports",
            ingest_settings={"CHUNK_SIZE": "400"},
            on_progress=lambda stage, current, total: frames.append(stage),
            store=self.corpus(docs_root),
            stale=False,
            return_state=True,
        )
        assert state.is_completed()
        result = state.result()
        assert rag_export_flow.name == "rag-export"
        assert result.counts == {"documents": 1, "chunks": 1}
        assert read_manifest(result.bundle.path).ingest_settings == {"CHUNK_SIZE": "400"}
        assert frames[0] == "snapshot" and frames[-1] == "finalize"
        with get_client(sync_client=True) as client:
            run = client.read_flow_run(state.state_details.flow_run_id)
        assert run.state is not None and run.state.is_completed()

    def test_the_flow_reads_the_stale_flag_and_opens_its_own_store(
        self, docs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unseeded, the flow asks app_settings for the flag and connects itself."""
        from varagity.backup.bundle import read_manifest
        from varagity.pipeline import rag_export_flow

        backup_flow_module = importlib.import_module("varagity.pipeline.backup_flow")
        corpus = self.corpus(docs_root)
        opened: list[str] = []

        class FlaggedSettings:
            def __enter__(self) -> "FlaggedSettings":
                opened.append("settings")
                return self

            def __exit__(self, *exc: object) -> None:
                opened.append("settings closed")

            def is_corpus_stale(self) -> bool:
                return True

        class OwnedStore:
            def __enter__(self) -> FakeCorpusStore:
                opened.append("store")
                return corpus

            def __exit__(self, *exc: object) -> None:
                opened.append("store closed")

        monkeypatch.setattr(backup_flow_module, "AppSettingsStore", FlaggedSettings)
        monkeypatch.setattr(backup_flow_module, "ContextualVectorDB", OwnedStore)
        result = rag_export_flow(docs_root / "varagity-exports", ingest_settings={})
        assert read_manifest(result.bundle.path).stale is True
        assert opened == ["settings", "settings closed", "store", "store closed"]

    def test_the_flow_carries_no_retries(self) -> None:
        """A failed export leaves no bundle; the user presses Export again."""
        from varagity.pipeline import rag_export_flow

        assert rag_export_flow.retries in (0, None)


class FakeFlagStore:
    """Records the corpus-stale writes a flow makes."""

    def __init__(self) -> None:
        self.writes: list[bool] = []

    def set_corpus_stale(self, stale: bool) -> None:
        self.writes.append(stale)


class TestRagRestoreFlow:
    """The RAG restore's tracking shell — the backup package does the work (ADR-019)."""

    SETTINGS = {"CHUNK_SIZE": "400"}

    @pytest.fixture
    def docs_root(self, tmp_path: Path, settings_env: Callable[..., None]) -> Path:
        root = tmp_path / "docs"
        root.mkdir()
        settings_env(DOCS_PATH=str(root), EMBEDDING_MODEL="test-model", EMBEDDING_DIM=2)
        return root

    def bundle(self, docs_root: Path) -> Path:
        from varagity.backup.rag import export_rag

        store = FakeCorpusStore(
            [backup_document("doc-a", str(docs_root / "a.md"), 1)],
            [backup_chunk("doc-a", 0, 0, [0.5, 0.25])],
        )
        return export_rag(
            docs_root / "varagity-exports", store, self.SETTINGS, stale=False
        ).bundle.path

    def test_rag_restore_is_a_tracked_flow_run_that_flags_stale_on_completion(
        self, docs_root: Path
    ) -> None:
        from varagity.pipeline import rag_restore_flow

        flags = FakeFlagStore()
        store = FakeRestoreStore()
        frames: list[str] = []
        state = rag_restore_flow(
            self.bundle(docs_root),
            ingest_settings=lambda: {"CHUNK_SIZE": "800"},  # the stack moved on
            on_progress=lambda stage, current, total: frames.append(stage),
            store=store,
            bm25=FakeRebuildBM25(),
            settings_store=flags,
            return_state=True,
        )
        assert state.is_completed()
        result = state.result()
        assert rag_restore_flow.name == "rag-restore"
        assert result.counts == {"documents": 1, "chunks": 1, "bm25": 1}
        assert result.stale is True
        assert flags.writes == [True]  # derived: the bundle's CHUNK_SIZE differs
        assert frames[0] == "validate" and frames[-1] == "bm25"
        assert [row.source for row in store.documents] == [str((docs_root / "a.md").resolve())]
        with get_client(sync_client=True) as client:
            run = client.read_flow_run(state.state_details.flow_run_id)
        assert run.state is not None and run.state.is_completed()

    def test_matching_settings_clear_the_flag(self, docs_root: Path) -> None:
        from varagity.pipeline import rag_restore_flow

        flags = FakeFlagStore()
        result = rag_restore_flow(
            self.bundle(docs_root),
            ingest_settings=lambda: dict(self.SETTINGS),
            store=FakeRestoreStore(),
            bm25=FakeRebuildBM25(),
            settings_store=flags,
        )
        assert result.stale is False
        assert flags.writes == [False]

    def test_a_failed_restore_writes_no_flag(self, docs_root: Path) -> None:
        """★ Only a completed restore may say what the corpus now is."""
        import zipfile

        from varagity.backup.bundle import BundleDamaged
        from varagity.pipeline import rag_restore_flow

        bundle = self.bundle(docs_root)
        with zipfile.ZipFile(bundle) as archive:
            chunks = archive.read("chunks.jsonl")
        damaged = rezip(
            bundle, {"chunks.jsonl": chunks.replace(b"chunk 0", b"chunk 9")}, reseal=False
        )
        flags = FakeFlagStore()
        with pytest.raises(BundleDamaged):
            rag_restore_flow(
                damaged,
                ingest_settings=lambda: dict(self.SETTINGS),
                store=FakeRestoreStore(),
                bm25=FakeRebuildBM25(),
                settings_store=flags,
            )
        assert flags.writes == []

    def test_the_flow_opens_its_own_stores_when_none_are_given(
        self, docs_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from varagity.pipeline import rag_restore_flow

        backup_flow_module = importlib.import_module("varagity.pipeline.backup_flow")
        opened: list[str] = []
        flags = FakeFlagStore()

        def owned(name: str, value: object) -> Any:
            class _Owned:
                def __enter__(self) -> object:
                    opened.append(name)
                    return value

                def __exit__(self, *exc: object) -> None:
                    opened.append(f"{name} closed")

            return lambda: _Owned()

        monkeypatch.setattr(
            backup_flow_module, "ContextualVectorDB", owned("store", FakeRestoreStore())
        )
        monkeypatch.setattr(
            backup_flow_module, "ElasticsearchBM25", owned("bm25", FakeRebuildBM25())
        )
        monkeypatch.setattr(backup_flow_module, "AppSettingsStore", owned("settings", flags))
        rag_restore_flow(self.bundle(docs_root), ingest_settings=lambda: dict(self.SETTINGS))
        assert opened == [
            "store",
            "bm25",
            "bm25 closed",
            "store closed",
            "settings",
            "settings closed",
        ]
        assert flags.writes == [False]

    def test_the_flow_carries_no_retries(self) -> None:
        """A failed restore rolled back (or says what it left); the user presses Restore again."""
        from varagity.pipeline import rag_restore_flow

        assert rag_restore_flow.retries in (0, None)


class TestGraphExportFlow:
    """The graph export's tracking shell — the backup package does the work (ADR-019)."""

    @pytest.fixture
    def service(
        self, tmp_path: Path, settings_env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> GraphService:
        settings_env(
            EMBEDDING_MODEL="test-model",
            EMBEDDING_DIM=2,
            GRAPH_DOCS_PATH=str(tmp_path / "graph-docs"),
            GRAPH_STORAGE_PATH=str(tmp_path / "graph-data"),
        )
        monkeypatch.setitem(GRAPH_ENGINE_REGISTRY, "_flow_probe", FakeEngine())
        return GraphService(
            "_flow_probe", workdir=build_workdir(tmp_path / "graph-data" / "lightrag")
        )

    def test_graph_export_is_a_tracked_flow_run(
        self, service: GraphService, tmp_path: Path
    ) -> None:
        from varagity.backup.bundle import read_manifest
        from varagity.pipeline import graph_export_flow

        frames: list[str] = []
        state = graph_export_flow(
            tmp_path / "graph-docs" / "varagity-exports",
            on_progress=lambda stage, current, total: frames.append(stage),
            service=service,
            stale=True,
            return_state=True,
        )
        assert state.is_completed()
        result = state.result()
        assert graph_export_flow.name == "graph-export"
        manifest = read_manifest(result.bundle.path)
        assert (manifest.corpus, manifest.stale, manifest.engine) == ("graph", True, FAKE_FORMAT)
        assert result.counts["thread_days"] == 2
        assert frames[0] == "snapshot" and "pack" in frames and frames[-1] == "finalize"
        assert service.building is False  # the write lock went back
        with get_client(sync_client=True) as client:
            run = client.read_flow_run(state.state_details.flow_run_id)
        assert run.state is not None and run.state.is_completed()

    def test_the_flow_reads_the_stale_flag_and_finds_the_process_service(
        self, service: GraphService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unseeded, the flow asks app_settings for the graph flag and uses the singleton."""
        from varagity.backup.bundle import read_manifest
        from varagity.pipeline import graph_export_flow

        backup_flow_module = importlib.import_module("varagity.pipeline.backup_flow")
        opened: list[str] = []

        class FlaggedSettings:
            def __enter__(self) -> "FlaggedSettings":
                opened.append("settings")
                return self

            def __exit__(self, *exc: object) -> None:
                opened.append("settings closed")

            def is_graph_stale(self) -> bool:
                return True

        monkeypatch.setattr(backup_flow_module, "AppSettingsStore", FlaggedSettings)
        monkeypatch.setattr(backup_flow_module, "get_graph_service", lambda: service)
        result = graph_export_flow(tmp_path / "graph-docs" / "varagity-exports")
        assert read_manifest(result.bundle.path).stale is True
        assert opened == ["settings", "settings closed"]

    def test_the_flow_carries_no_retries(self) -> None:
        """A failed export leaves no bundle; the user presses Export again."""
        from varagity.pipeline import graph_export_flow

        assert graph_export_flow.retries in (0, None)
