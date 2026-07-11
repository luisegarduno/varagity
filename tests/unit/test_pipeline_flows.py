"""Unit tests for the Prefect flow adapters (spec §9/§10, Phase 8).

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
from uuid import UUID

import pytest
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter
from prefect.testing.utilities import prefect_test_harness

from tests.unit.test_loader import FakeBM25, FakeEmbeddings, FakeStore
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
        """★ The phase's DoD row: each §9 stage of each file → a task run."""
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
            ingest_flow_module._TASK_STAGES,
            store=store_chunks_task.with_options(retry_delay_seconds=0),
        )
        monkeypatch.setattr(ingest_flow_module, "_TASK_STAGES", fast_stages)

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
            ingest_flow_module._TASK_STAGES,
            store=store_chunks_task.with_options(retry_delay_seconds=0),
        )
        monkeypatch.setattr(ingest_flow_module, "_TASK_STAGES", fast_stages)

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
        assert state["query_vector"] == vector  # filled since Phase 8
        assert state["retrieved"] == retriever.chunks
        assert state["formatted_context"] in llm.prompts[0]
        assert state["answer"] == "Lantern powers Aurora. [SOURCE]: a.md"  # think-stripped

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
        assert sorted(names) == ["embed_query", "generate_answer", "retrieve"]

    def test_interactive_tasks_carry_no_retries(self) -> None:
        """The query path leaves retrying to the clients' tenacity layer."""
        for interactive in (embed_query_task, retrieve_task, generate_answer_task):
            assert not interactive.retries, interactive.name

    def test_invalid_verbose_raises(self, pinned_settings: None) -> None:
        with pytest.raises(ValueError, match="verbose"):
            query_flow("q?", retriever=FakeRetriever([], None), llm=ScriptedLLM("A."), verbose=9)
