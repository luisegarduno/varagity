"""Prometheus instrumentation: the spec_v2 §6.2 metric catalog + recording helpers.

The catalog registers in ``prometheus_client``'s default process-wide
registry and is exposed by the API's ``GET /metrics`` (spec_v2 §6.1 source
1 — direct in-app instrumentation, the primary metrics source; plan
decision #10). Probe points call the ``observe_*``/``count_*``/``set_*``
helpers below instead of touching the metric objects, so the label
vocabulary stays in one place:

* the Prefect flows (:mod:`varagity.pipeline.query_flow`,
  :mod:`varagity.pipeline.ingest_flow`) — per-stage query latency,
  retrieval scores + rerank deltas, query/ingest counters, LLM tokens;
* the reranked retriever (:mod:`varagity.retrieval.reranked`) — the
  ``rerank`` stage's latency;
* the API health probes (:mod:`varagity.api.deps`) — per-dependency
  reachability.

Metrics accumulate per process: the API (a single uvicorn worker — plan
decision #11, no multiprocess registry) is what Prometheus scrapes. A CLI
run records into its own short-lived process instead — harmless, and it
keeps the instrumented code path identical for both front-ends.

Recording is unconditional; ``settings.METRICS_ENABLED`` gates only the
``/metrics`` endpoint (the collectors are cheap in-memory counters).
"""

from collections.abc import Sequence

from prometheus_client import Counter, Gauge, Histogram

from varagity.stores.records import RetrievedChunk

# Stage latencies span ~5 ms (query embedding) to minutes (generation on
# a reasoning model), so the buckets stretch far past the client defaults.
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
    10.0, 20.0, 30.0, 60.0, 120.0,
)  # fmt: skip

# Semantic/fused/rerank scores live in [0, 1]; raw BM25 scores are
# unbounded (Elasticsearch practical range ≲ 20), hence the sparse tail.
_SCORE_BUCKETS = (
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    2.0, 5.0, 10.0, 20.0,
)  # fmt: skip

# Rank movement (pre-rerank − post-rerank) is signed and bounded by the
# candidate pool (±RERANK_CANDIDATES, default 40). Negative buckets mean
# the client exposes no `_sum` sample (Prometheus semantics) — chart this
# with bucket quantiles, never sum/count averages.
_RERANK_DELTA_BUCKETS = (-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

# Contextualization is observed per document (the task boundary): ≈12 s per
# chunk on the reference GPU × tens of chunks reaches into the hundreds of
# seconds. Per-chunk cost is derived in PromQL against the chunk counter.
_CONTEXTUALIZE_BUCKETS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0)

QUERY_LATENCY_SECONDS = Histogram(
    "varagity_query_latency_seconds",
    "Query latency per pipeline stage — the per-stage numbers the evidence footer shows.",
    labelnames=("stage", "method"),
    buckets=_LATENCY_BUCKETS,
)

RETRIEVAL_SCORE = Histogram(
    "varagity_retrieval_score",
    "Final score of each retrieved chunk, by retrieval method and result rank.",
    labelnames=("method", "rank"),
    buckets=_SCORE_BUCKETS,
)

RERANK_DELTA = Histogram(
    "varagity_rerank_delta",
    "Rank movement per reranked chunk (pre-rerank − post-rerank; + moved up).",
    buckets=_RERANK_DELTA_BUCKETS,
)

QUERY_TOTAL = Counter(
    "varagity_query_total",
    "Completed query flows, by retrieval method and outcome (ok/aborted/error).",
    labelnames=("method", "outcome"),
)

INGEST_DOCS_TOTAL = Counter(
    "varagity_ingest_docs_total",
    "Documents stored by ingestion, by file type and extraction provenance.",
    labelnames=("file_type", "extraction"),
)

INGEST_CHUNKS_TOTAL = Counter(
    "varagity_ingest_chunks_total",
    "Chunks stored by ingestion, by chunking strategy.",
    labelnames=("chunking_strategy",),
)

CONTEXTUALIZE_LATENCY_SECONDS = Histogram(
    "varagity_contextualize_latency_seconds",
    "Per-document situating-blurb generation latency (the spec §9.4 throughput cost).",
    buckets=_CONTEXTUALIZE_BUCKETS,
)

LLM_TOKENS_TOTAL = Counter(
    "varagity_llm_tokens_total",
    "Server-reported LLM tokens consumed by streamed answers, by direction.",
    labelnames=("direction",),
)

DEPENDENCY_UP = Gauge(
    "varagity_dependency_up",
    "Reachability of each backing service at its last health probe (1 up, 0 down).",
    labelnames=("service",),
)


def observe_query_stage(stage: str, method: str, seconds: float) -> None:
    """Record one query stage's wall-clock latency.

    Args:
        stage: Pipeline stage name (``embed``/``retrieve``/``rerank``/
            ``generate``). The ``retrieve`` observation of the ``reranked``
            method *includes* its ``rerank`` sub-stage, which is also
            recorded separately by the reranked retriever.
        method: The retrieval method's registry name (low-cardinality
            label; injected non-registry implementations record as
            ``custom``).
        seconds: The stage's duration.
    """
    QUERY_LATENCY_SECONDS.labels(stage=stage, method=method).observe(seconds)


def observe_retrieval(method: str, chunks: Sequence[RetrievedChunk]) -> None:
    """Record the retrieved chunks' final scores and any rerank movement.

    Args:
        method: The retrieval method's registry name.
        chunks: The query's final retrieved chunks, best first. Each score
            lands in the ``rank``-labelled histogram; chunks whose trace
            carries a ``rerank_delta`` also feed the rerank-movement
            histogram.
    """
    for rank, chunk in enumerate(chunks, start=1):
        RETRIEVAL_SCORE.labels(method=method, rank=str(rank)).observe(chunk.score)
        if chunk.trace is not None and chunk.trace.rerank_delta is not None:
            RERANK_DELTA.observe(chunk.trace.rerank_delta)


def count_query(method: str, outcome: str) -> None:
    """Count one completed query flow.

    Args:
        method: The retrieval method's registry name.
        outcome: ``ok``, ``aborted`` (client disconnected mid-stream), or
            ``error`` (the flow raised).
    """
    QUERY_TOTAL.labels(method=method, outcome=outcome).inc()


def count_ingested_document(file_type: str, extraction: str) -> None:
    """Count one document stored by ingestion.

    Args:
        file_type: The document's file type (``pdf``/``md``/``docx``/…).
        extraction: Extraction provenance (``text`` or ``ocr_fallback``) —
            the OCR-fallback-rate numerator.
    """
    INGEST_DOCS_TOTAL.labels(file_type=file_type, extraction=extraction).inc()


def count_ingested_chunks(chunking_strategy: str, n: int) -> None:
    """Count chunks stored by ingestion.

    Args:
        chunking_strategy: Registry name of the strategy that produced them.
        n: How many chunks the document stored.
    """
    INGEST_CHUNKS_TOTAL.labels(chunking_strategy=chunking_strategy).inc(n)


def observe_contextualize(seconds: float) -> None:
    """Record one document's situating-blurb generation latency.

    Args:
        seconds: Wall-clock duration of the document's contextualize stage.
    """
    CONTEXTUALIZE_LATENCY_SECONDS.observe(seconds)


def count_llm_tokens(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """Count server-reported token usage for one streamed answer.

    Args:
        prompt_tokens: Prompt-side tokens, or ``None`` when the server
            reported no usage (nothing is recorded for that direction).
        completion_tokens: Completion-side tokens, same convention.
    """
    if prompt_tokens is not None:
        LLM_TOKENS_TOTAL.labels(direction="prompt").inc(prompt_tokens)
    if completion_tokens is not None:
        LLM_TOKENS_TOTAL.labels(direction="completion").inc(completion_tokens)


def set_dependency_up(service: str, up: bool) -> None:
    """Record one backing service's probe outcome.

    Refreshed by every health probe — in compose, the ``api`` container's
    own healthcheck hits ``GET /api/health`` every 15 s, so the gauge stays
    current without a dedicated poller.

    Args:
        service: The service name (``llamacpp``/``infinity``/``postgres``/
            ``elasticsearch``/``prefect``).
        up: Whether the probe succeeded.
    """
    DEPENDENCY_UP.labels(service=service).set(1.0 if up else 0.0)
