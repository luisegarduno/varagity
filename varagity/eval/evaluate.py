"""Retrieval evaluation: recall@k / pass@k over the 4-config matrix (spec §16).

Quantifies the retrieval-quality ladder Phases 4→5→6 climbed, on the golden
set over ``tests/fixtures/corpus``:

1. ``semantic_noncontextual`` — the Phase 4 baseline,
2. ``semantic_contextual`` — contextual embeddings (≈35% tier),
3. ``bm25_contextual`` — contextual BM25,
4. ``hybrid_contextual`` — rank fusion (≈49% tier, the v1 default).

Measurement runs against **ephemeral testcontainers stores** (plan decision
#4) so the live corpus is never touched, while embeddings and the
contextualizing LLM are the real GPU services from the running compose
stack (they are stateless). Two ingests cover all four configs: ingest A
(``CONTEXTUALIZE=false``) backs config 1; ingest B (``CONTEXTUALIZE=true``,
``reingest`` over the same stores) backs configs 2–4 against the same
index.

Metric semantics: :func:`recall_at_k` is the Anthropic cookbook's
evaluation number (there called *Pass@n*) — the per-query fraction of
golden chunks present in the top-k, averaged over queries.
:func:`pass_at_k` is the stricter complement reported alongside it: the
fraction of queries whose golden chunks are **all** in the top-k.

The eval pipeline settings are pinned (:data:`PINNED_EVAL_SETTINGS`) rather
than inherited from ``.env``: golden ``chunk_index`` values assume those
chunk boundaries, and pinning keeps runs comparable across machines. Pins
are applied by exporting environment variables and clearing the settings
cache — the same mechanism the test suite uses — because deep pipeline
code resolves ``get_settings()`` internally; the eval harness is a
single-threaded CLI path, where this is safe.
"""

import json
import logging
import os
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.eval.containers import EphemeralStores, ephemeral_stores
from varagity.eval.datasets import ResolvedGoldenEntry, load_golden, resolve_golden
from varagity.ingest.loader import IngestSummary, ingest_corpus
from varagity.models.registry import get_model
from varagity.retrieval.base import Retriever
from varagity.retrieval.bm25 import BM25Retriever
from varagity.retrieval.hybrid import HybridRetriever
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.records import RetrievedChunk

logger = logging.getLogger(__name__)

# Retrieval depths reported by every evaluation (spec §16).
K_VALUES: tuple[int, ...] = (5, 10, 20)

# Defaults for the repo-root-relative eval inputs/outputs (run from the
# repo root, as every CLI command is).
EVAL_CORPUS = Path("tests/fixtures/corpus")
GOLDEN_PATH = Path("data/eval/golden_qa.jsonl")
RESULTS_DIR = Path("data/eval/results")

# The pipeline settings every eval run pins (see the module docstring).
# CONTEXTUALIZE is deliberately absent — it is the variable under test —
# as are the model endpoints, which name the live GPU services.
PINNED_EVAL_SETTINGS: dict[str, str] = {
    "ALLOWED_EXTENSIONS": ".pdf,.txt,.md",
    "CHUNKING_STRATEGY": "recursive_character",
    "CHUNK_SIZE": "400",
    "CHUNK_OVERLAP": "50",
    "PDF_OCR_FALLBACK": "true",
    "PDF_OCR_MIN_CHARS": "50",
    "PDF_OCR_TEXTLESS_PAGE_RATIO": "0.2",
    "PDF_OCR_FORCE_FULL_PAGE": "false",
    "OCR_LANGUAGES": "en",
    "SEMANTIC_WEIGHT": "0.8",
    "BM25_WEIGHT": "0.2",
}

# An ingest-compatible callable: `ingest_corpus` or the Prefect-tracked
# `ingest_flow` (same keyword surface), injected by the pipeline layer.
IngestCallable = Callable[..., IngestSummary]


def recall_at_k(golden: Collection[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of golden chunk ids present in the top-k retrieved.

    The Anthropic cookbook's evaluation metric (there reported as
    *Pass@n*): one query's score; callers average over queries.

    Args:
        golden: The query's golden chunk ids (non-empty).
        retrieved: Retrieved chunk ids, best first.
        k: Retrieval depth to evaluate at.

    Returns:
        ``|golden ∩ retrieved[:k]| / |golden|``.

    Raises:
        ValueError: If ``golden`` is empty or ``k`` is not positive.
    """
    if not golden:
        raise ValueError("recall_at_k needs at least one golden chunk id")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    top = set(retrieved[:k])
    return sum(1 for chunk_id in golden if chunk_id in top) / len(golden)


def pass_at_k(golden: Collection[str], retrieved: Sequence[str], k: int) -> float:
    """Whether **all** golden chunk ids are in the top-k (0.0 or 1.0).

    The strict complement of :func:`recall_at_k`: a query passes only if
    nothing a perfect retriever would return is missing. Callers average
    over queries into a pass rate.

    Args:
        golden: The query's golden chunk ids (non-empty).
        retrieved: Retrieved chunk ids, best first.
        k: Retrieval depth to evaluate at.

    Returns:
        ``1.0`` if every golden id is in ``retrieved[:k]``, else ``0.0``.

    Raises:
        ValueError: If ``golden`` is empty or ``k`` is not positive.
    """
    return 1.0 if recall_at_k(golden, retrieved, k) == 1.0 else 0.0


@contextmanager
def pinned_eval_settings(**overrides: str) -> Iterator[None]:
    """Apply :data:`PINNED_EVAL_SETTINGS` (plus overrides) for a block.

    Exports the pins as environment variables and clears the settings
    cache, restoring both on exit — the eval-harness counterpart of the
    test suite's ``settings_env`` fixture (see the module docstring for
    why an environment override rather than parameter injection).

    Args:
        **overrides: Additional setting pins for this block (e.g.
            ``CONTEXTUALIZE="true"``, ``OCR_ENGINE="tesseract"``); they
            take precedence over the standing pins.

    Yields:
        Nothing; the pinned settings are active inside the block.
    """
    pins = {**PINNED_EVAL_SETTINGS, **overrides}
    saved = {name: os.environ.get(name) for name in pins}
    os.environ.update(pins)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_settings.cache_clear()


def measure_retriever(
    retriever: Retriever,
    entries: Sequence[ResolvedGoldenEntry],
    *,
    k_values: Sequence[int] = K_VALUES,
) -> tuple[dict[str, dict[str, float]], list[dict[str, int | None]]]:
    """Score one retriever over the golden entries at every depth.

    Each query retrieves once at ``max(k_values)``; shallower depths are
    prefix cuts of that single ranked list (retrieval order is
    deterministic given the stores' contents).

    Args:
        retriever: The retrieval method under measurement.
        entries: Resolved golden entries.
        k_values: Depths to report.

    Returns:
        A ``(summary, golden_ranks)`` pair. ``summary`` holds the
        ``{"recall": {"5": …}, "pass": {"5": …}}`` averages over queries;
        ``golden_ranks`` holds, per entry in order, each golden
        ``chunk_id``'s 1-based position in the ranked list (``None`` if
        absent) — the raw material for debugging a bad score.

    Raises:
        ValueError: If ``entries`` or ``k_values`` is empty.
    """
    if not entries:
        raise ValueError("measure_retriever needs at least one golden entry")
    if not k_values:
        raise ValueError("measure_retriever needs at least one k value")
    max_k = max(k_values)
    recall_sums = dict.fromkeys(k_values, 0.0)
    pass_sums = dict.fromkeys(k_values, 0.0)
    golden_ranks: list[dict[str, int | None]] = []
    for entry in entries:
        chunks: list[RetrievedChunk] = retriever.retrieve(entry.query, k=max_k, verbose=0)
        retrieved_ids = [chunk.chunk_id for chunk in chunks]
        for k in k_values:
            recall_sums[k] += recall_at_k(entry.chunk_ids, retrieved_ids, k)
            pass_sums[k] += pass_at_k(entry.chunk_ids, retrieved_ids, k)
        positions = {chunk_id: rank for rank, chunk_id in enumerate(retrieved_ids, start=1)}
        golden_ranks.append({chunk_id: positions.get(chunk_id) for chunk_id in entry.chunk_ids})
    summary = {
        "recall": {str(k): recall_sums[k] / len(entries) for k in k_values},
        "pass": {str(k): pass_sums[k] / len(entries) for k in k_values},
    }
    return summary, golden_ranks


def validate_golden_against_store(
    entries: Sequence[ResolvedGoldenEntry],
    stores: EphemeralStores,
    *,
    strict: bool = True,
) -> list[str]:
    """Check that every golden ref exists in the freshly ingested store.

    A golden ``chunk_index`` beyond a document's actual chunk count means
    the golden set has drifted from the chunking pipeline (or, in the OCR
    benchmark, that an engine's text produced different chunk boundaries)
    — a score computed against it would silently under-report.

    Args:
        entries: Resolved golden entries.
        stores: The ingested ephemeral stores.
        strict: Raise on unresolvable refs (the retrieval matrix) instead
            of warning and returning them (the OCR benchmark, where
            per-engine boundary drift is a known, reported caveat).

    Returns:
        The unresolvable ``chunk_id``s (always empty when ``strict``).

    Raises:
        ValueError: If ``strict`` and any ref is unresolvable.
    """
    n_chunks: dict[str, int | None] = {}
    unresolvable: list[str] = []
    for entry in entries:
        for ref, chunk_id in zip(entry.relevant, entry.chunk_ids, strict=True):
            doc_id = chunk_id.split("::", 1)[0]
            if doc_id not in n_chunks:
                n_chunks[doc_id] = stores.store.document_n_chunks(doc_id)
            count = n_chunks[doc_id]
            if count is None or ref.chunk_index >= count:
                unresolvable.append(chunk_id)
    if unresolvable:
        message = (
            f"{len(unresolvable)} golden ref(s) don't exist in the ingested store "
            f"(chunk boundaries drifted?): {unresolvable}"
        )
        if strict:
            raise ValueError(message)
        logger.warning("%s — counted as guaranteed misses", message)
    return unresolvable


def write_results(kind: str, payload: dict[str, Any], *, results_dir: Path = RESULTS_DIR) -> Path:
    """Persist one evaluation run's results as timestamped JSON (spec §16).

    Args:
        kind: Short result-file tag (``"matrix"`` or ``"ocr"``).
        payload: The full results document.
        results_dir: Directory for result files (created if missing).

    Returns:
        The path written.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{stamp}-{kind}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s results to %s", kind, path)
    return path


def run_matrix(
    *,
    corpus_root: Path = EVAL_CORPUS,
    golden_path: Path = GOLDEN_PATH,
    results_dir: Path = RESULTS_DIR,
    ingest: IngestCallable = ingest_corpus,
    verbose: int | None = None,
) -> dict[str, Any]:
    """Run the 4-configuration retrieval matrix and persist its results.

    Two ingests into ephemeral stores cover all four configs (see the
    module docstring). Embeddings and the contextualizing LLM resolve from
    settings — the live GPU services.

    Args:
        corpus_root: The eval corpus (defaults to the fixtures corpus the
            golden set is authored over).
        golden_path: The golden dataset file.
        results_dir: Directory for the timestamped results JSON.
        ingest: Ingest callable (``ingest_corpus``, or the Prefect
            ``ingest_flow`` when invoked through the eval flow so both
            ingests are tracked subflows).
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The results document (also written to ``results_dir``), with a
        ``"results_path"`` key naming the file.

    Raises:
        ValueError: If ``verbose`` is invalid, or the golden set doesn't
            match the ingested chunk boundaries.
        FileNotFoundError: If the golden set or a referenced corpus file
            is missing.
    """
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    entries = resolve_golden(load_golden(golden_path), corpus_root)

    # Resolve the live model clients up front: a misconfigured endpoint
    # should fail before containers spin, not after an ingest.
    embeddings = get_model("embedding")
    llm = get_model("default")

    configs: dict[str, dict[str, dict[str, float]]] = {}
    ranks_by_config: dict[str, list[dict[str, int | None]]] = {}
    ingest_seconds: dict[str, float] = {}

    with ephemeral_stores() as stores:
        retrievers: dict[str, Retriever] = {
            "semantic": SemanticRetriever(store=stores.store, embeddings=embeddings),
            "bm25": BM25Retriever(bm25=stores.bm25, store=stores.store),
            "hybrid": HybridRetriever(store=stores.store, bm25=stores.bm25, embeddings=embeddings),
        }

        # Ingest A — non-contextual baseline → config 1.
        with pinned_eval_settings(CONTEXTUALIZE="false"):
            logger.info("ingest A (non-contextual baseline) into ephemeral stores")
            started = time.monotonic()
            summary_a = ingest(
                str(corpus_root),
                store=stores.store,
                bm25=stores.bm25,
                embeddings=embeddings,
                verbose=verbose,
            )
            ingest_seconds["noncontextual"] = round(time.monotonic() - started, 2)
            if summary_a.failed:
                raise RuntimeError(f"ingest A failed for {summary_a.failed} file(s)")
            validate_golden_against_store(entries, stores, strict=True)
            configs["semantic_noncontextual"], ranks_by_config["semantic_noncontextual"] = (
                measure_retriever(retrievers["semantic"], entries)
            )

        # Ingest B — contextual (LLM blurbs) → configs 2–4 on one index.
        with pinned_eval_settings(CONTEXTUALIZE="true"):
            logger.info("ingest B (contextual) into the same ephemeral stores (reingest)")
            started = time.monotonic()
            summary_b = ingest(
                str(corpus_root),
                store=stores.store,
                bm25=stores.bm25,
                embeddings=embeddings,
                llm=llm,
                reingest=True,
                verbose=verbose,
            )
            ingest_seconds["contextual"] = round(time.monotonic() - started, 2)
            if summary_b.failed:
                raise RuntimeError(f"ingest B failed for {summary_b.failed} file(s)")
            for config, retriever_name in (
                ("semantic_contextual", "semantic"),
                ("bm25_contextual", "bm25"),
                ("hybrid_contextual", "hybrid"),
            ):
                configs[config], ranks_by_config[config] = measure_retriever(
                    retrievers[retriever_name], entries
                )

    settings = get_settings()
    results: dict[str, Any] = {
        "kind": "retrieval_matrix",
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": str(corpus_root),
        "golden_path": str(golden_path),
        "n_queries": len(entries),
        "k_values": list(K_VALUES),
        "embedding_model": settings.EMBEDDING_MODEL,
        "pinned_settings": dict(PINNED_EVAL_SETTINGS),
        "chunks_ingested": summary_b.chunks,
        "ingest_seconds": ingest_seconds,
        "configs": configs,
        "per_query": [
            {
                "query": entry.query,
                "golden": entry.chunk_ids,
                "golden_ranks": {
                    config: ranks_by_config[config][index] for config in ranks_by_config
                },
            }
            for index, entry in enumerate(entries)
        ],
    }
    results["results_path"] = str(write_results("matrix", results, results_dir=results_dir))
    return results
