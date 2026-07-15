# ADR-006: Reranking wired into the query path (the ≈67% tier)

**Status:** Accepted (2026-07-14)

## Context

v1 shipped at the Anthropic ladder's ≈49% tier (contextual embeddings +
contextual BM25 + hybrid fusion) with reranking *staged*: the cross-encoder
(`bge-reranker-v2-m3`) was already served by the infinity container
([ADR-002](ADR-002-infinity-over-fastembed.md)) and `.env.example` carried
`RERANK_ENABLED=false`. v2 Phase 1 wires it in — sequenced first in the v2
build because it also produces the per-chunk rank provenance the evidence
panel ([ADR-005 §3](ADR-005-web-stack-and-api.md)) is designed to display.

## Decision

A **composing `reranked` retriever** (`varagity/retrieval/reranked.py`, a
registry file), not a hard-coded flow stage and not a fork of fusion:

1. Resolve the base retriever from `RERANK_BASE_METHOD` (default `hybrid`;
   validated ∈ {semantic, bm25, hybrid} — no recursion) and over-fetch a
   pool of `max(RERANK_CANDIDATES, k)` candidates (default **40**, the
   Anthropic 150→20 pattern scaled to this corpus).
2. Cross-encode each candidate's **`content`** against the query via
   infinity's `POST /v1/rerank` (a dedicated `httpx` `RerankClient` — the
   endpoint is not an OpenAI-SDK method; same `tenacity` posture as the
   other clients).
3. Keep the top `min(k, RERANK_TOP_N)` (default **5**), recording
   `rerank_score`, `rerank_delta = pre-rank − final rank`, and `final_rank`
   onto each chunk's `RetrievalTrace` — composing the per-arm ranks and
   fused score the base retriever already filled in.

`RERANK_ENABLED` is retained as a **kill switch orthogonal to method
selection**: with `RETRIEVAL_METHOD=reranked` and the switch off, the
retriever degrades to its base method's ranking and logs the degradation —
the GUI toggle and the eval baseline both work without renaming the method.

## Measured results

Eval matrix config #5 (`hybrid_rerank_contextual`), 20 golden queries, 16
chunks, eval pins `RERANK_CANDIDATES=40` / `RERANK_TOP_N=20` (pinned wide so
recall@10/20 stay meaningful; the production default keeps 5):

- Run `20260712T020600Z-matrix.json`: all five configs — including
  `hybrid_rerank_contextual` — at **1.000 recall/pass for k ∈ {5, 10, 20}**.
  `reranked ≥ hybrid` holds as equality: the fixtures corpus saturates, as
  [ADR-004](ADR-004-ocr-engine-choice.md) already documented for OCR. The
  honest deliverable (spec_v2 §5.5) is that the harness proves the wiring
  end-to-end; the *discriminative* signal still waits on the deferred
  cookbook-corpus swap.
- Run `20260712T194137Z-matrix.json` (chunker sweep, foreign chunk
  boundaries): `reranked` recall@5 = **1.000 under every strategy**, and it
  repaired the sweep's only hybrid slip — `docling_hybrid` hybrid 0.975 →
  reranked 1.000 at k=5. Small, but the right direction everywhere.
- Latency ≈ **0.66 s per 40-document pool** on the RTX 5060, stable over
  three runs (v2 plan Phase 1 verification) — timed as its own `rerank`
  sub-stage metric ([ADR-007](ADR-007-observability-stack.md)).

## Rationale

- **Compose, don't fork.** The retriever calls `get_retriever(base)`, so
  fusion logic exists once and hybrid improvements flow through; a
  per-method fork would duplicate fusion *and* the trace plumbing. Selection
  stays the registry promise: `RETRIEVAL_METHOD=reranked`, zero caller edits.
- **Rerank the original `content`,** not `contextualized_content` —
  cross-encoders score the actual passage; the situating blurb already did
  its job at the embedding/BM25 stage.
- **No new container.** A separate reranker service was rejected: the 8 GB
  RTX 5060 already hosts the model next to e5 in infinity, whose multi-model
  serving came free (ADR-002) — a second GPU service would fight the same
  card for VRAM and add an operational unit for zero capability.
- **The trace is the product** as much as the ranking: `rerank_delta` feeds
  the CLI `-v 2` badges, the web evidence panel, the `message_sources`
  snapshots, and the `varagity_rerank_delta` histogram — one data model,
  four consumers.

## Consequences

- Config validation enforces the composition's invariants:
  `0 < RERANK_TOP_N ≤ RERANK_CANDIDATES`, and with
  `RETRIEVAL_METHOD=reranked`, `RERANK_TOP_N ≤ TOP_K ≤ RERANK_CANDIDATES`.
- Operational constraints ride the embedding container (see the
  [runbook](../runbook.md#the-reranker-rides-the-embedding-container)):
  torch has no `sm_120` kernels for the 5060, so infinity runs
  `INFINITY_ENGINE=optimum` with pre-exported ONNX and the
  `INFINITY_BATCH_SIZE='32;4'` per-model cap; the optimum engine ignores
  `INFINITY_DEVICE_ID`, so GPU pinning happens via compose `device_ids`.
- Only a served **cross-encoder** is structurally valid at `/rerank`:
  configuring a bi-encoder (e5, jina) as `RERANK_MODEL` fails the request,
  and the client surfaces it as a clear `ValueError` rather than a retry
  loop. Transient failures retry (`tenacity`); query-path Prefect tasks
  stay deliberately retry-free, unchanged.
- Re-run `uv run --group eval main.py eval` after the discriminative corpus
  lands — that is where `reranked > hybrid` gets to show a strict
  inequality (and where `RERANK_CANDIDATES=40` earns re-tuning).
