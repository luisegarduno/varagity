"""Retrieval evaluation: recall@k / pass@k over the 7-config matrix (spec §16).

Quantifies the retrieval-quality ladder, on the golden set over
``tests/fixtures/corpus``:

1. ``semantic_noncontextual`` — the vanilla-RAG v1 baseline,
2. ``semantic_contextual`` — contextual embeddings (≈35% tier),
3. ``bm25_contextual`` — contextual BM25,
4. ``hybrid_contextual`` — rank fusion (≈49% tier, the default),
5. ``hybrid_rerank_contextual`` — + cross-encoder re-ranking (≈67% tier,
   spec_v2 §5.5),
6. ``hyde_contextual`` — HyDE over the same hybrid base (ADR-016): the
   dense arm searches with an LLM-written hypothetical passage,
7. ``hyde_rerank_contextual`` — the HyDE+rerank pairing (``reranked``
   composing ``hyde``): HyDE steers the candidate pool, the cross-encoder
   judges the real query.

The **chunker sweep** (spec_v2 §7.4) follows the matrix: every
registered chunking strategy gets its own contextual ingest over the same
ephemeral stores and is measured across the contextual retrieval configs.
Because the golden set's ``chunk_index`` refs are authored against the
pinned default boundaries, the sweep re-resolves each ref **by content**:
a ref's ``fact`` snippet is located in the strategy-true chunks
(:func:`resolve_golden_by_fact`), and a ref counts as retrieved when *any*
chunk containing its fact is in the top-k — index-anchored scoring under
foreign boundaries would measure nothing but the boundary mismatch.

Measurement runs against **ephemeral testcontainers stores** (plan decision
#4) so the live corpus is never touched, while embeddings, the
contextualizing LLM, and the reranker are the real GPU services from the
running compose stack (they are stateless). Two ingests cover all seven
configs: ingest A (``CONTEXTUALIZE=false``) backs config 1; ingest B
(``CONTEXTUALIZE=true``, ``reingest`` over the same stores) backs configs
2–7 against the same index — and doubles as the sweep's
``recursive_character`` row. Each remaining strategy adds one reingest.
The two HyDE configs additionally pay one live-LLM passage generation per
query each (independent generations — sampling noise between configs 6 and
7 is part of what the pairing comparison measures).

Metric semantics: :func:`recall_at_k` is the Anthropic cookbook's
evaluation number (there called *Pass@n*) — the per-query fraction of
golden chunks present in the top-k, averaged over queries.
:func:`pass_at_k` is the stricter complement reported alongside it: the
fraction of queries whose golden chunks are **all** in the top-k. The
sweep's :func:`recall_at_k_any` / :func:`pass_at_k_any` are the same two
numbers with each golden ref generalized from one chunk id to an
acceptable **set** (fact-containing chunks).

The eval pipeline settings are pinned (:data:`PINNED_EVAL_SETTINGS`) rather
than inherited from ``.env``: golden ``chunk_index`` values assume those
chunk boundaries, and pinning keeps runs comparable across machines. Pins
are applied by exporting environment variables and clearing the settings
cache — the same mechanism the test suite uses — because deep pipeline
code resolves ``get_settings()`` internally; the eval harness is a
single-threaded CLI path, where this is safe.

The **chat-engine eval** (spec_v3 §4.9) is the multi-turn
counterpart: every registered chat engine runs the hand-authored
conversation fixtures (``tests/fixtures/conversations/``), and each turn's
retrieval is scored fact-anchored exactly like the sweep — the engine's
``search_query`` (the condensed rewrite, under ``condense_context``)
drives retrieval, and a turn's ref is satisfied when any fact-carrying
chunk lands in the top-k. Assistant replies are scripted in the fixtures,
so both engines see byte-identical history: the comparison isolates the
engine, never its answers. Depths are :data:`CHAT_K_VALUES`, shallower
than the matrix's: the eval corpus ingests ~16 chunks, so recall@20 is
1.0 by definition and k=5 is the production context cut
(``RERANK_TOP_N``).
"""

import json
import logging
import os
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from varagity.chat import CHAT_ENGINE_REGISTRY
from varagity.chat.base import ChatEngine, Turn
from varagity.chunking import CHUNKER_REGISTRY
from varagity.config import get_settings
from varagity.debug.show import check_verbose
from varagity.eval.containers import EphemeralStores, ephemeral_stores
from varagity.eval.datasets import (
    CONVERSATION_KINDS,
    ConversationFixture,
    GoldenEntry,
    ResolvedGoldenEntry,
    load_conversations,
    load_golden,
    resolve_golden,
)
from varagity.ingest.loader import IngestSummary, ingest_corpus
from varagity.models.llm import LLMClient
from varagity.models.registry import get_model
from varagity.retrieval.base import Retriever
from varagity.retrieval.bm25 import BM25Retriever
from varagity.retrieval.hybrid import HybridRetriever
from varagity.retrieval.hyde import HydeRetriever
from varagity.retrieval.reranked import RerankedRetriever
from varagity.retrieval.semantic import SemanticRetriever
from varagity.stores.records import RetrievedChunk
from varagity.stores.vector_store import ContextualVectorDB

logger = logging.getLogger(__name__)

# Retrieval depths reported by every evaluation (spec §16).
K_VALUES: tuple[int, ...] = (5, 10, 20)

# Chat-eval retrieval depths (spec_v3 §4.9). Deliberately shallower than
# K_VALUES: the pinned eval-corpus ingest yields ~16 chunks, so recall@20
# retrieves the whole store and discriminates nothing, while k=5 is the
# production context cut (RERANK_TOP_N=5 — what the answer actually sees).
CHAT_K_VALUES: tuple[int, ...] = (1, 3, 5)

# Defaults for the repo-root-relative eval inputs/outputs (run from the
# repo root, as every CLI command is).
EVAL_CORPUS = Path("tests/fixtures/corpus")
GOLDEN_PATH = Path("data/eval/golden_qa.jsonl")
CONVERSATIONS_DIR = Path("tests/fixtures/conversations")
RESULTS_DIR = Path("data/eval/results")

# The pipeline settings every eval run pins (see the module docstring).
# CONTEXTUALIZE is deliberately absent — it is the variable under test —
# as are the model endpoints, which name the live GPU services. The rerank
# pins (spec_v2 §5.5) hold RERANK_TOP_N at max(K_VALUES) so the reranked
# config fills the deepest reported cut, and TOP_K alongside it so the
# combination stays valid whatever RETRIEVAL_METHOD the host env selects.
# The HyDE pins are the PINNED_CHAT_SETTINGS lesson over again: a host env
# with HYDE_ENABLED=false would silently measure configs 6–7 as plain
# hybrid(+rerank).
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
    "TOP_K": "20",
    "RERANK_ENABLED": "true",
    "RERANK_BASE_METHOD": "hybrid",
    "RERANK_CANDIDATES": "40",
    "RERANK_TOP_N": "20",
    "HYDE_ENABLED": "true",
    "HYDE_BASE_METHOD": "hybrid",
    "HYDE_MAX_TOKENS": "1024",
    "HYDE_MAX_CHARS": "2000",
}

# An ingest-compatible callable: `ingest_corpus` or the Prefect-tracked
# `ingest_flow` (same keyword surface), injected by the pipeline layer.
IngestCallable = Callable[..., IngestSummary]

# The retrieval configs each swept chunker is measured across (the matrix's
# contextual ladder; the CONTEXTUALIZE dimension is orthogonal to chunking
# and already measured by configs 1–2, so the sweep doesn't re-pay a
# non-contextual ingest per strategy).
SWEEP_RETRIEVAL_CONFIGS: tuple[str, ...] = ("semantic", "bm25", "hybrid", "reranked")

# The retrieval configs each chat engine is measured across: the production
# method and its base. Each turn condenses once; the same search_query is
# scored under both, so a rerank-masks/amplifies effect stays visible.
CHAT_EVAL_RETRIEVAL_CONFIGS: tuple[str, ...] = ("hybrid", "reranked")

# Condense pins for the chat eval, alongside PINNED_EVAL_SETTINGS: the
# engines read these via get_settings() inside prepare(), and a host env
# with the kill switch off (or different bounds) would silently measure
# `condense_context` as `simple`.
PINNED_CHAT_SETTINGS: dict[str, str] = {
    "CONDENSE_ENABLED": "true",
    "CONDENSE_HISTORY_TURNS": "6",
    "CONDENSE_MAX_TOKENS": "512",
    "CONDENSE_MAX_CHARS": "512",
}


class FactRef(BaseModel):
    """One golden ref resolved to the chunks that satisfy it (sweep form).

    Attributes:
        label: Human-readable identity in reports — the ref's ``fact``
            (or its index-anchored ``chunk_id`` for a fact-less ref).
        chunk_ids: Every chunk id whose content contains the fact under the
            currently ingested boundaries (overlap can spread a fact over
            several); retrieval of **any** of them satisfies the ref. Empty
            when the fact matched nothing — a guaranteed miss.
    """

    label: str
    chunk_ids: list[str]


class FactResolvedEntry(BaseModel):
    """A golden entry re-resolved by content for one chunking strategy.

    Attributes:
        query: The evaluation question.
        refs: One :class:`FactRef` per golden ref, in ref order.
    """

    query: str
    refs: list[FactRef]


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


def recall_at_k_any(
    acceptable: Sequence[Collection[str]], retrieved: Sequence[str], k: int
) -> float:
    """Fraction of golden refs with **any** acceptable chunk in the top-k.

    :func:`recall_at_k` generalized for the chunker sweep: a ref is
    satisfied by whichever strategy-true chunk carries its fact. An empty
    acceptable set (the fact matched no chunk) can never be satisfied.

    Args:
        acceptable: Per golden ref, the chunk ids that satisfy it.
        retrieved: Retrieved chunk ids, best first.
        k: Retrieval depth to evaluate at.

    Returns:
        ``|satisfied refs| / |refs|``.

    Raises:
        ValueError: If ``acceptable`` is empty or ``k`` is not positive.
    """
    if not acceptable:
        raise ValueError("recall_at_k_any needs at least one golden ref")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    top = set(retrieved[:k])
    return sum(1 for ids in acceptable if top.intersection(ids)) / len(acceptable)


def pass_at_k_any(acceptable: Sequence[Collection[str]], retrieved: Sequence[str], k: int) -> float:
    """Whether **every** golden ref is satisfied in the top-k (0.0 or 1.0).

    Args:
        acceptable: Per golden ref, the chunk ids that satisfy it.
        retrieved: Retrieved chunk ids, best first.
        k: Retrieval depth to evaluate at.

    Returns:
        ``1.0`` if every ref has an acceptable id in ``retrieved[:k]``,
        else ``0.0``.

    Raises:
        ValueError: If ``acceptable`` is empty or ``k`` is not positive.
    """
    return 1.0 if recall_at_k_any(acceptable, retrieved, k) == 1.0 else 0.0


def resolve_golden_by_fact(
    entries: Sequence[ResolvedGoldenEntry], store: ContextualVectorDB
) -> list[FactResolvedEntry]:
    """Re-resolve golden refs by content against the ingested boundaries.

    For each ref with a ``fact``, scans its document's stored chunks
    (case-insensitively — OCR extraction may case-shift) and collects every
    chunk containing the fact. A ref without a fact falls back to its
    index-anchored ``chunk_id`` (only honest under the pinned default
    boundaries — the golden set carries facts precisely so the sweep never
    relies on this). A fact that matches no chunk is logged and left with
    an empty acceptable set: a guaranteed miss, mirroring the OCR
    benchmark's drift posture rather than silently inflating scores.

    Args:
        entries: The index-resolved golden entries (their ``relevant`` refs
            carry the facts; their ``chunk_ids`` carry the doc ids).
        store: The vector store holding the current strategy's ingest.

    Returns:
        One fact-resolved entry per input entry, in order.
    """
    chunks_by_doc: dict[str, list[RetrievedChunk]] = {}
    resolved: list[FactResolvedEntry] = []
    for entry in entries:
        refs: list[FactRef] = []
        for ref, chunk_id in zip(entry.relevant, entry.chunk_ids, strict=True):
            if ref.fact is None:
                refs.append(FactRef(label=chunk_id, chunk_ids=[chunk_id]))
                continue
            doc_id = chunk_id.split("::", 1)[0]
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = store.document_chunks(doc_id)
            needle = ref.fact.lower()
            matches = [
                chunk.chunk_id for chunk in chunks_by_doc[doc_id] if needle in chunk.content.lower()
            ]
            if not matches:
                logger.warning(
                    "golden fact %r not found in any ingested chunk of %s "
                    "(%d chunks) — counted as a guaranteed miss",
                    ref.fact,
                    ref.rel_source,
                    len(chunks_by_doc[doc_id]),
                )
            refs.append(FactRef(label=ref.fact, chunk_ids=matches))
        resolved.append(FactResolvedEntry(query=entry.query, refs=refs))
    return resolved


def measure_retriever_facts(
    retriever: Retriever,
    entries: Sequence[FactResolvedEntry],
    *,
    k_values: Sequence[int] = K_VALUES,
) -> tuple[dict[str, dict[str, float]], list[dict[str, int | None]]]:
    """Score one retriever over fact-resolved entries at every depth.

    The sweep counterpart of :func:`measure_retriever` — identical shape,
    with each ref satisfied by *any* of its acceptable chunk ids.

    Args:
        retriever: The retrieval method under measurement.
        entries: Fact-resolved golden entries.
        k_values: Depths to report.

    Returns:
        A ``(summary, golden_ranks)`` pair; ``golden_ranks`` holds, per
        entry in order, each ref label's best 1-based rank among its
        acceptable ids (``None`` if none was retrieved).

    Raises:
        ValueError: If ``entries`` or ``k_values`` is empty.
    """
    if not entries:
        raise ValueError("measure_retriever_facts needs at least one golden entry")
    if not k_values:
        raise ValueError("measure_retriever_facts needs at least one k value")
    max_k = max(k_values)
    recall_sums = dict.fromkeys(k_values, 0.0)
    pass_sums = dict.fromkeys(k_values, 0.0)
    golden_ranks: list[dict[str, int | None]] = []
    for entry in entries:
        chunks: list[RetrievedChunk] = retriever.retrieve(entry.query, k=max_k, verbose=0)
        retrieved_ids = [chunk.chunk_id for chunk in chunks]
        acceptable = [ref.chunk_ids for ref in entry.refs]
        for k in k_values:
            recall_sums[k] += recall_at_k_any(acceptable, retrieved_ids, k)
            pass_sums[k] += pass_at_k_any(acceptable, retrieved_ids, k)
        positions = {chunk_id: rank for rank, chunk_id in enumerate(retrieved_ids, start=1)}
        golden_ranks.append(
            {
                ref.label: min(
                    (positions[chunk_id] for chunk_id in ref.chunk_ids if chunk_id in positions),
                    default=None,
                )
                for ref in entry.refs
            }
        )
    summary = {
        "recall": {str(k): recall_sums[k] / len(entries) for k in k_values},
        "pass": {str(k): pass_sums[k] / len(entries) for k in k_values},
    }
    return summary, golden_ranks


def _mean_turn_scores(
    records: Sequence[dict[str, Any]], method: str, k_values: Sequence[int]
) -> dict[str, Any]:
    """Average per-turn recall/pass scores over a slice of turn records.

    Args:
        records: The turn records to average (non-empty).
        method: The retrieval config whose scores to read.
        k_values: Depths to report.

    Returns:
        ``{"n_turns": …, "recall": {"5": …}, "pass": {"5": …}}`` averages.
    """
    return {
        "n_turns": len(records),
        "recall": {
            str(k): sum(r["methods"][method]["recall"][str(k)] for r in records) / len(records)
            for k in k_values
        },
        "pass": {
            str(k): sum(r["methods"][method]["pass"][str(k)] for r in records) / len(records)
            for k in k_values
        },
    }


def measure_chat_engine(
    engine: ChatEngine,
    conversations: Sequence[ConversationFixture],
    fact_entries: Sequence[FactResolvedEntry],
    retrievers: Mapping[str, Retriever],
    *,
    llm: LLMClient | None,
    k_values: Sequence[int] = CHAT_K_VALUES,
) -> dict[str, Any]:
    """Run every conversation through one chat engine and score each turn.

    Per turn: the engine prepares the search query against the scripted
    history (one condense LLM call per follow-up under
    ``condense_context``; none under ``simple``), then that **one**
    ``search_query`` is scored under every retrieval config by swapping it
    into the turn's fact-resolved entry and reusing
    :func:`measure_retriever_facts`. History threads the fixture's
    scripted replies, so every engine sees the identical conversation —
    the comparison isolates the engine. Engine rendering is silenced
    (``verbose=0``): the harness reports rewrites itself, and per-turn
    console output across engines × turns would swamp it.

    Args:
        engine: The chat engine under measurement.
        conversations: The fixtures, in report order.
        fact_entries: One fact-resolved entry per turn, flattened in
            conversation order (the fixtures' turns, depth-first).
        retrievers: Retrieval configs to score under, keyed by name.
        llm: Chat client for engines that condense (the live service);
            ``None`` lets them resolve one via the model registry.
        k_values: Depths to report.

    Returns:
        The engine's results: ``"summary"`` (per retrieval config: the
        ``"all"`` / ``"follow_up"`` / ``"by_kind"`` slice averages —
        follow-ups are the discriminating slice, first turns being
        identity splits under every engine), ``"condense"`` (call count
        and latency stats), and ``"conversations"`` (per-turn detail:
        the search query used, per-config scores, and each ref's best
        golden rank).

    Raises:
        ValueError: If ``conversations``, ``retrievers`` or ``k_values``
            is empty, or ``fact_entries`` doesn't match the fixtures'
            total turn count.
    """
    if not conversations:
        raise ValueError("measure_chat_engine needs at least one conversation")
    if not retrievers:
        raise ValueError("measure_chat_engine needs at least one retriever")
    if not k_values:
        raise ValueError("measure_chat_engine needs at least one k value")
    n_turns = sum(len(conversation.turns) for conversation in conversations)
    if len(fact_entries) != n_turns:
        raise ValueError(
            f"got {len(fact_entries)} fact-resolved entries for {n_turns} conversation turns"
        )

    flat_index = 0
    conversation_results: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for conversation in conversations:
        history: list[Turn] = []
        turn_records: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(conversation.turns):
            entry = fact_entries[flat_index]
            flat_index += 1
            prepared = engine.prepare(turn.query, history=history, llm=llm, verbose=0)
            methods: dict[str, Any] = {}
            for name, retriever in retrievers.items():
                scored = entry.model_copy(update={"query": prepared.search_query})
                summary, ranks = measure_retriever_facts(retriever, [scored], k_values=k_values)
                methods[name] = {
                    "recall": summary["recall"],
                    "pass": summary["pass"],
                    "golden_ranks": ranks[0],
                }
            record: dict[str, Any] = {
                "turn": turn_index,
                "kind": turn.kind,
                "query": turn.query,
                "search_query": prepared.search_query,
                "condensed": prepared.condensed,
                "condense_latency_s": (
                    round(prepared.condense_latency_s, 3)
                    if prepared.condense_latency_s is not None
                    else None
                ),
                "methods": methods,
            }
            turn_records.append(record)
            all_records.append(record)
            if turn_index < len(conversation.turns) - 1:
                history.append(Turn(role="user", content=turn.query))
                # Validation guarantees a scripted reply on non-final turns.
                history.append(Turn(role="assistant", content=turn.assistant or ""))
        conversation_results.append({"name": conversation.name, "turns": turn_records})

    follow_ups = [record for record in all_records if record["turn"] > 0]
    summary_by_method: dict[str, Any] = {}
    for name in retrievers:
        by_kind = {
            kind: _mean_turn_scores(selected, name, k_values)
            for kind in CONVERSATION_KINDS
            if (selected := [r for r in follow_ups if r["kind"] == kind])
        }
        summary_by_method[name] = {
            "all": _mean_turn_scores(all_records, name, k_values),
            "follow_up": _mean_turn_scores(follow_ups, name, k_values),
            "by_kind": by_kind,
        }
    latencies = [
        record["condense_latency_s"]
        for record in all_records
        if record["condense_latency_s"] is not None
    ]
    condense = {
        "calls": len(latencies),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
    }
    return {
        "summary": summary_by_method,
        "condense": condense,
        "conversations": conversation_results,
    }


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
    """Run the 7-config retrieval matrix + the chunker sweep; persist results.

    Two ingests into ephemeral stores cover all seven configs (see the
    module docstring); the chunker sweep then reingests once per remaining
    registered strategy (``recursive_character`` reuses ingest B) and
    measures each across :data:`SWEEP_RETRIEVAL_CONFIGS` with fact-anchored
    golden resolution — deliberately *without* the HyDE configs: the
    hypothetical passage depends only on the query, never on chunk
    boundaries, so a per-strategy re-measure would re-pay every LLM
    generation to measure the same transformation. Embeddings, the
    contextualizing LLM (which also writes the HyDE passages), and the
    reranker resolve from settings — the live GPU services (the ``semantic``
    strategy additionally embeds at ingest time through the same service).

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
        RuntimeError: If either eval ingest reports failed files (a partial
            index would silently under-report every score).
    """
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    entries = resolve_golden(load_golden(golden_path), corpus_root)

    # Resolve the live model clients up front: a misconfigured endpoint
    # should fail before containers spin, not after an ingest.
    embeddings = get_model("embedding")
    llm = get_model("default")
    rerank = get_model("rerank")

    configs: dict[str, dict[str, dict[str, float]]] = {}
    ranks_by_config: dict[str, list[dict[str, int | None]]] = {}
    ingest_seconds: dict[str, float] = {}

    with ephemeral_stores() as stores:
        hybrid = HybridRetriever(store=stores.store, bm25=stores.bm25, embeddings=embeddings)
        hyde = HydeRetriever(base=hybrid, llm=llm, embeddings=embeddings)
        retrievers: dict[str, Retriever] = {
            "semantic": SemanticRetriever(store=stores.store, embeddings=embeddings),
            "bm25": BM25Retriever(bm25=stores.bm25, store=stores.store),
            "hybrid": hybrid,
            # Config 5 composes the same store-wired hybrid; the reranker
            # is the live infinity container's /rerank (spec_v2 §5.5).
            "reranked": RerankedRetriever(base=hybrid, rerank=rerank),
            # Configs 6–7 (ADR-016): hyde steers the same hybrid's dense
            # arm with a live-LLM hypothetical passage; the pairing stacks
            # rerank OUTSIDE hyde, so the cross-encoder judges the real
            # query while HyDE shapes the candidate pool.
            "hyde": hyde,
            "hyde_reranked": RerankedRetriever(base=hyde, rerank=rerank),
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

        # Ingest B — contextual (LLM blurbs) → configs 2–7 on one index.
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
                ("hybrid_rerank_contextual", "reranked"),
                ("hyde_contextual", "hyde"),
                ("hyde_rerank_contextual", "hyde_reranked"),
            ):
                configs[config], ranks_by_config[config] = measure_retriever(
                    retrievers[retriever_name], entries
                )

        # ── Chunker sweep (spec_v2 §7.4): one contextual ingest per
        # strategy, measured across the contextual retrieval configs with
        # fact-anchored golden resolution. recursive_character goes first,
        # reusing ingest B's index before any reingest replaces it.
        sweep: dict[str, dict[str, Any]] = {}
        strategies = ["recursive_character"] + sorted(
            name for name in CHUNKER_REGISTRY if name != "recursive_character"
        )
        for strategy in strategies:
            with pinned_eval_settings(CONTEXTUALIZE="true", CHUNKING_STRATEGY=strategy):
                if strategy == "recursive_character":
                    chunks_ingested, seconds = summary_b.chunks, ingest_seconds["contextual"]
                else:
                    logger.info("chunker sweep: reingesting with %r", strategy)
                    started = time.monotonic()
                    summary = ingest(
                        str(corpus_root),
                        store=stores.store,
                        bm25=stores.bm25,
                        embeddings=embeddings,
                        llm=llm,
                        reingest=True,
                        verbose=verbose,
                    )
                    seconds = round(time.monotonic() - started, 2)
                    if summary.failed:
                        raise RuntimeError(
                            f"chunker-sweep ingest ({strategy}) failed for {summary.failed} file(s)"
                        )
                    chunks_ingested = summary.chunks
                fact_entries = resolve_golden_by_fact(entries, stores.store)
                unresolved = [
                    ref.label for entry in fact_entries for ref in entry.refs if not ref.chunk_ids
                ]
                methods: dict[str, dict[str, dict[str, float]]] = {}
                method_ranks: dict[str, list[dict[str, int | None]]] = {}
                for name in SWEEP_RETRIEVAL_CONFIGS:
                    methods[name], method_ranks[name] = measure_retriever_facts(
                        retrievers[name], fact_entries
                    )
                sweep[strategy] = {
                    "chunks": chunks_ingested,
                    "ingest_seconds": seconds,
                    "unresolved_facts": unresolved,
                    "configs": methods,
                    "golden_ranks": method_ranks,
                }

    settings = get_settings()
    results: dict[str, Any] = {
        "kind": "retrieval_matrix",
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": str(corpus_root),
        "golden_path": str(golden_path),
        "n_queries": len(entries),
        "k_values": list(K_VALUES),
        "embedding_model": settings.EMBEDDING_MODEL,
        "rerank_model": settings.RERANK_MODEL,
        "pinned_settings": dict(PINNED_EVAL_SETTINGS),
        "chunks_ingested": summary_b.chunks,
        "ingest_seconds": ingest_seconds,
        "configs": configs,
        "sweep_retrieval_configs": list(SWEEP_RETRIEVAL_CONFIGS),
        "chunker_sweep": sweep,
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


def run_chat_eval(
    *,
    corpus_root: Path = EVAL_CORPUS,
    conversations_dir: Path = CONVERSATIONS_DIR,
    results_dir: Path = RESULTS_DIR,
    ingest: IngestCallable = ingest_corpus,
    verbose: int | None = None,
) -> dict[str, Any]:
    """Run the multi-turn chat-engine eval; persist results (spec_v3 §4.9).

    One contextual ingest into ephemeral stores backs the whole run; every
    registered chat engine (enumerated from the registry, like the chunker
    sweep) then plays each conversation fixture, and every turn is scored
    fact-anchored under :data:`CHAT_EVAL_RETRIEVAL_CONFIGS`. The condense
    settings are pinned (:data:`PINNED_CHAT_SETTINGS`) so a host env with
    the kill switch off can't silently measure ``condense_context`` as
    ``simple``. This harness decides the shipped ``CHAT_ENGINE`` default —
    and "the default stays ``simple``" is a valid result, not a failure.

    Args:
        corpus_root: The eval corpus (defaults to the fixtures corpus the
            conversation refs are authored over).
        conversations_dir: The conversation fixtures directory.
        results_dir: Directory for the timestamped results JSON.
        ingest: Ingest callable (``ingest_corpus``, or the Prefect
            ``ingest_flow`` when invoked through the eval flow).
        verbose: Console verbosity (0–2); defaults to
            ``settings.DEFAULT_VERBOSE``.

    Returns:
        The results document (also written to ``results_dir``), with a
        ``"results_path"`` key naming the file.

    Raises:
        ValueError: If ``verbose`` is invalid or a fixture fails
            validation.
        FileNotFoundError: If the fixtures directory or a referenced
            corpus file is missing.
        RuntimeError: If the eval ingest reports failed files (a partial
            index would silently under-report every score).
    """
    verbose = check_verbose(get_settings().DEFAULT_VERBOSE if verbose is None else verbose)
    conversations = load_conversations(conversations_dir)
    flat_golden = [
        GoldenEntry(query=turn.query, relevant=turn.relevant)
        for conversation in conversations
        for turn in conversation.turns
    ]
    resolved = resolve_golden(flat_golden, corpus_root)

    # Resolve the live model clients up front: a misconfigured endpoint
    # should fail before containers spin, not after an ingest.
    embeddings = get_model("embedding")
    llm = get_model("default")
    rerank = get_model("rerank")

    with ephemeral_stores() as stores:
        hybrid = HybridRetriever(store=stores.store, bm25=stores.bm25, embeddings=embeddings)
        # Keyed to match CHAT_EVAL_RETRIEVAL_CONFIGS (defined together above).
        retrievers: dict[str, Retriever] = {
            "hybrid": hybrid,
            "reranked": RerankedRetriever(base=hybrid, rerank=rerank),
        }
        with pinned_eval_settings(CONTEXTUALIZE="true", **PINNED_CHAT_SETTINGS):
            logger.info("chat eval: contextual ingest into ephemeral stores")
            started = time.monotonic()
            summary = ingest(
                str(corpus_root),
                store=stores.store,
                bm25=stores.bm25,
                embeddings=embeddings,
                llm=llm,
                verbose=verbose,
            )
            ingest_seconds = round(time.monotonic() - started, 2)
            if summary.failed:
                raise RuntimeError(f"chat-eval ingest failed for {summary.failed} file(s)")
            fact_entries = resolve_golden_by_fact(resolved, stores.store)
            unresolved = [
                ref.label for entry in fact_entries for ref in entry.refs if not ref.chunk_ids
            ]
            engines = {
                name: measure_chat_engine(
                    CHAT_ENGINE_REGISTRY[name],
                    conversations,
                    fact_entries,
                    retrievers,
                    llm=llm,
                    k_values=CHAT_K_VALUES,
                )
                for name in sorted(CHAT_ENGINE_REGISTRY)
            }

    settings = get_settings()
    results: dict[str, Any] = {
        "kind": "chat_eval",
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": str(corpus_root),
        "conversations_dir": str(conversations_dir),
        "n_conversations": len(conversations),
        "n_turns": len(flat_golden),
        "n_follow_up_turns": sum(len(c.turns) - 1 for c in conversations),
        "k_values": list(CHAT_K_VALUES),
        "retrieval_configs": list(CHAT_EVAL_RETRIEVAL_CONFIGS),
        "embedding_model": settings.EMBEDDING_MODEL,
        "rerank_model": settings.RERANK_MODEL,
        "pinned_settings": {
            **PINNED_EVAL_SETTINGS,
            "CONTEXTUALIZE": "true",
            **PINNED_CHAT_SETTINGS,
        },
        "chunks_ingested": summary.chunks,
        "ingest_seconds": ingest_seconds,
        "unresolved_facts": unresolved,
        "engines": engines,
    }
    results["results_path"] = str(write_results("chat", results, results_dir=results_dir))
    return results
