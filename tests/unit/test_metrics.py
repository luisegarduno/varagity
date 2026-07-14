"""Unit tests for the Prometheus catalog, its helpers, and GET /metrics.

The collectors live in ``prometheus_client``'s process-wide default
registry and accumulate across tests, so every assertion here is
delta-based (read → act → read) rather than absolute. Flow-level
increments on stubbed query/ingest runs are covered in
``test_pipeline_flows.py`` (they need the Prefect harness); this module
covers the helpers, the reranked probe point, the endpoint, and the
dependency gauge.
"""

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY

import varagity.api.deps as deps
from tests.unit.test_reranked import FakeBase, FakeRerank, _traced_chunk
from varagity.api.main import create_app
from varagity.api.schemas import ServiceHealth
from varagity.models.rerank import RerankResult
from varagity.observability import metrics
from varagity.retrieval.reranked import RerankedRetriever
from varagity.stores.records import RetrievedChunk


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Read one sample from the default registry, defaulting to 0."""
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else value


def _reranked_chunk(i: int, score: float, *, fused_rank: int, delta: int) -> RetrievedChunk:
    """A traced chunk whose trace carries a rerank delta."""
    chunk = _traced_chunk(i, score, fused_rank=fused_rank)
    assert chunk.trace is not None
    trace = chunk.trace.model_copy(update={"rerank_delta": delta})
    return chunk.model_copy(update={"trace": trace})


@pytest.fixture
def rerank_settings(settings_env: Callable[..., None]) -> Callable[..., None]:
    """Pin the rerank knobs (enabled, small pool) for the probe-point tests."""

    def _pin(**overrides: object) -> None:
        values: dict[str, object] = {
            "RERANK_ENABLED": "true",
            "RERANK_BASE_METHOD": "hybrid",
            "RERANK_CANDIDATES": 5,
            "RERANK_TOP_N": 3,
            "TOP_K": 10,
        }
        values.update(overrides)
        settings_env(**values)

    return _pin


class TestRecordingHelpers:
    def test_observe_query_stage_records_with_labels(self) -> None:
        labels = {"stage": "embed", "method": "hybrid"}
        before_count = sample("varagity_query_latency_seconds_count", labels)
        before_sum = sample("varagity_query_latency_seconds_sum", labels)

        metrics.observe_query_stage("embed", "hybrid", 0.25)

        assert sample("varagity_query_latency_seconds_count", labels) == before_count + 1
        assert sample("varagity_query_latency_seconds_sum", labels) == pytest.approx(
            before_sum + 0.25
        )

    def test_observe_retrieval_records_scores_by_rank_and_rerank_deltas(self) -> None:
        chunks: list[RetrievedChunk] = [
            _reranked_chunk(0, 0.9, fused_rank=1, delta=3),
            _traced_chunk(1, 0.7, fused_rank=2),  # trace without a rerank delta
        ]
        rank1 = {"method": "reranked", "rank": "1"}
        rank2 = {"method": "reranked", "rank": "2"}
        before_r1 = sample("varagity_retrieval_score_count", rank1)
        before_r2 = sample("varagity_retrieval_score_count", rank2)
        before_delta_count = sample("varagity_rerank_delta_count")
        before_le4 = sample("varagity_rerank_delta_bucket", {"le": "4.0"})
        before_le2 = sample("varagity_rerank_delta_bucket", {"le": "2.0"})

        metrics.observe_retrieval("reranked", chunks)

        assert sample("varagity_retrieval_score_count", rank1) == before_r1 + 1
        assert sample("varagity_retrieval_score_count", rank2) == before_r2 + 1
        # Only the chunk whose trace carries a delta feeds the movement
        # histogram. The +3 observation lands in the le=4 bucket but not
        # le=2 (asserted via buckets: a negative-bucket histogram exposes
        # no `_sum` sample — Prometheus semantics).
        assert sample("varagity_rerank_delta_count") == before_delta_count + 1
        assert sample("varagity_rerank_delta_bucket", {"le": "4.0"}) == before_le4 + 1
        assert sample("varagity_rerank_delta_bucket", {"le": "2.0"}) == before_le2

    def test_observe_retrieval_of_traceless_chunks_records_no_delta(self) -> None:
        chunk = _traced_chunk(0, 0.5, fused_rank=1).model_copy(update={"trace": None})
        before = sample("varagity_rerank_delta_count")
        metrics.observe_retrieval("semantic", [chunk])
        assert sample("varagity_rerank_delta_count") == before

    def test_count_query_labels_method_and_outcome(self) -> None:
        labels = {"method": "bm25", "outcome": "error"}
        before = sample("varagity_query_total", labels)
        metrics.count_query("bm25", "error")
        assert sample("varagity_query_total", labels) == before + 1

    def test_ingest_counters_label_provenance_and_strategy(self) -> None:
        doc_labels = {"file_type": "pdf", "extraction": "ocr_fallback"}
        chunk_labels = {"chunking_strategy": "markdown_aware"}
        before_docs = sample("varagity_ingest_docs_total", doc_labels)
        before_chunks = sample("varagity_ingest_chunks_total", chunk_labels)

        metrics.count_ingested_document("pdf", "ocr_fallback")
        metrics.count_ingested_chunks("markdown_aware", 7)

        assert sample("varagity_ingest_docs_total", doc_labels) == before_docs + 1
        assert sample("varagity_ingest_chunks_total", chunk_labels) == before_chunks + 7

    def test_observe_contextualize_feeds_the_histogram(self) -> None:
        before_count = sample("varagity_contextualize_latency_seconds_count")
        before_sum = sample("varagity_contextualize_latency_seconds_sum")
        metrics.observe_contextualize(42.0)
        assert sample("varagity_contextualize_latency_seconds_count") == before_count + 1
        assert sample("varagity_contextualize_latency_seconds_sum") == pytest.approx(
            before_sum + 42.0
        )

    def test_count_llm_tokens_by_direction_and_none_safety(self) -> None:
        prompt = {"direction": "prompt"}
        completion = {"direction": "completion"}
        before_prompt = sample("varagity_llm_tokens_total", prompt)
        before_completion = sample("varagity_llm_tokens_total", completion)

        metrics.count_llm_tokens(120, 34)
        metrics.count_llm_tokens(None, None)  # unreported usage records nothing

        assert sample("varagity_llm_tokens_total", prompt) == before_prompt + 120
        assert sample("varagity_llm_tokens_total", completion) == before_completion + 34

    def test_set_dependency_up_flips_the_gauge(self) -> None:
        metrics.set_dependency_up("elasticsearch", True)
        assert sample("varagity_dependency_up", {"service": "elasticsearch"}) == 1.0
        metrics.set_dependency_up("elasticsearch", False)
        assert sample("varagity_dependency_up", {"service": "elasticsearch"}) == 0.0


class TestRerankedProbePoint:
    def test_rerank_stage_latency_is_observed(self, rerank_settings: Callable[..., None]) -> None:
        """The rerank sub-stage records under stage="rerank", method="reranked"."""
        rerank_settings()
        candidates = [_traced_chunk(i, 1.0 - i / 10, fused_rank=i + 1) for i in range(5)]
        scored = [RerankResult(index=i, relevance_score=0.9 - i / 10) for i in range(5)]
        retriever = RerankedRetriever(base=FakeBase(candidates), rerank=FakeRerank(scored))
        labels = {"stage": "rerank", "method": "reranked"}
        before = sample("varagity_query_latency_seconds_count", labels)

        retriever.retrieve("q", k=5, verbose=0)

        assert sample("varagity_query_latency_seconds_count", labels) == before + 1

    def test_kill_switch_records_no_rerank_stage(
        self, rerank_settings: Callable[..., None]
    ) -> None:
        rerank_settings(RERANK_ENABLED="false")
        candidates = [_traced_chunk(i, 1.0 - i / 10, fused_rank=i + 1) for i in range(5)]
        retriever = RerankedRetriever(base=FakeBase(candidates), rerank=FakeRerank([]))
        labels = {"stage": "rerank", "method": "reranked"}
        before = sample("varagity_query_latency_seconds_count", labels)

        retriever.retrieve("q", k=5, verbose=0)

        assert sample("varagity_query_latency_seconds_count", labels) == before


async def get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.get(path)


class TestMetricsEndpoint:
    async def test_exposes_the_catalog_in_prometheus_text_format(self) -> None:
        """Every spec_v2 §6.2 family is present on a fresh scrape.

        ``prometheus_client`` canonically strips the ``_total`` suffix from
        counter *family* names (their samples keep it), so the counters are
        asserted via their TYPE lines.
        """
        response = await get(create_app(), "/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        text = response.text
        for header in (
            "# TYPE varagity_query_latency_seconds histogram",
            "# TYPE varagity_retrieval_score histogram",
            "# TYPE varagity_rerank_delta histogram",
            "# TYPE varagity_query_total counter",
            "# TYPE varagity_ingest_docs_total counter",
            "# TYPE varagity_ingest_chunks_total counter",
            "# TYPE varagity_contextualize_latency_seconds histogram",
            "# TYPE varagity_llm_tokens_total counter",
            "# TYPE varagity_dependency_up gauge",
        ):
            assert header in text, f"missing catalog entry: {header}"

    async def test_serves_recorded_samples_with_labels(self) -> None:
        metrics.count_query("hybrid", "ok")
        metrics.set_dependency_up("postgres", True)
        text = (await get(create_app(), "/metrics")).text
        assert 'varagity_query_total{method="hybrid",outcome="ok"}' in text
        assert 'varagity_dependency_up{service="postgres"} 1.0' in text

    async def test_metrics_route_appears_in_openapi(self) -> None:
        schema = (await get(create_app(), "/openapi.json")).json()
        assert "/metrics" in schema["paths"]

    async def test_disabled_gate_turns_the_endpoint_into_404(
        self, settings_env: Callable[..., None]
    ) -> None:
        settings_env(METRICS_ENABLED="false")
        response = await get(create_app(), "/metrics")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


class TestDependencyGauge:
    async def test_check_services_refreshes_the_gauge(
        self, settings_env: Callable[..., None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings_env(  # deterministic probe URLs so the fake can tell services apart
            BASE_MODEL_API_URL="http://llamacpp.test/v1",
            EMBEDDING_API_URL="http://infinity.test/v1",
            ELASTICSEARCH_URL="http://es.test:9200",
            PREFECT_API_URL="http://prefect.test/api",
        )

        async def probe(client: Any, url: str, *, headers: Any = None) -> ServiceHealth:
            return ServiceHealth(ok="es.test" not in url)

        monkeypatch.setattr(deps, "_probe_http", probe)
        monkeypatch.setattr(deps, "_probe_postgres", lambda: ServiceHealth(ok=True))

        statuses = await deps.check_services(
            ("llamacpp", "infinity", "postgres", "elasticsearch", "prefect")
        )

        assert statuses["elasticsearch"].ok is False
        assert sample("varagity_dependency_up", {"service": "elasticsearch"}) == 0.0
        for service in ("llamacpp", "infinity", "postgres", "prefect"):
            assert sample("varagity_dependency_up", {"service": service}) == 1.0
