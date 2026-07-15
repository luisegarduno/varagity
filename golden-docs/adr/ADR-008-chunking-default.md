# ADR-008: `recursive_character` stays the chunking default

**Status:** Accepted (decided by the Phase 6 chunker sweep, 2026-07-12; owner-confirmed)

## Context

The chunker registry carried one strategy since v1; v2 Phase 6 added four
(`token_based`, `markdown_aware`, `semantic`, `docling_hybrid`), each a
registry file, with the shipped default to be **benchmark-decided, not
preference-decided** (v2 plan decision #14 — the same rule that produced
[ADR-004](ADR-004-ocr-engine-choice.md)).

The sweep had to be honest about a subtle trap first: the golden refs'
`chunk_index` values are authored against the pinned default's boundaries,
so under any other strategy an index-anchored ref points at arbitrary text.
Every golden ref therefore carries a verbatim **`fact` snippet**, and the
sweep re-resolves refs by scanning each strategy's *actual* chunks for the
fact (case-insensitive; unmatched facts count as guaranteed misses). All 20
queries' facts resolved under all five strategies (`unresolved_facts: []`).

## Benchmark results

Run `20260712T194137Z-matrix.json`: five strategies × four retrieval
methods (contextualized ingest per strategy), 20 golden queries, recall@5:

| Strategy | Chunks | Ingest (s) | semantic | bm25 | hybrid | reranked |
|---|---|---|---|---|---|---|
| **recursive_character** (default) | 16 | 196.11 | 0.950 | 1.000 | 1.000 | 1.000 |
| token_based | 6 | 86.14 | 0.950 | 1.000 | 1.000 | 1.000 |
| markdown_aware | 17 | 183.20 | **1.000** | 1.000 | 1.000 | 1.000 |
| semantic | 13 | 164.14 | 1.000 | 0.975 | 1.000 | 1.000 |
| docling_hybrid | 12 | 141.18 | 0.950 | 1.000 | 0.975 | 1.000 |

- `hybrid` and `reranked` sit at 1.000 under (almost) every strategy — the
  tiny corpus saturates, exactly as it did for the retrieval matrix and the
  OCR benchmark (ADR-004). `reranked` is 1.000 under **all five**.
- Ingest cost tracks **chunk count**, not strategy sophistication: the
  ≈12 s/chunk contextualization blurbs dominate (6 chunks → 86 s, 16 →
  196 s); the `semantic` chunker's own embedding calls are negligible next
  to them.
- `markdown_aware` is the only strategy at 1.000 across **all four**
  methods at k=5 — i.e. unaided by rerank, including the weakest
  (semantic-only) arm — and additionally carries `heading_path` breadcrumb
  provenance.

## Decision

**The default stays `recursive_character`** (400 chars / 50 overlap).
**`markdown_aware` is recorded as the strongest candidate** — first to
re-test when the deferred discriminative corpus lands.

A sub-decision rode the sweep (plan decision #12): token-denominated
strategies count tokens with **tiktoken `cl100k_base`, not the exact e5 HF
tokenizer**. Measured on the fixture chunks, the true e5 (XLM-RoBERTa)
count runs **mean +10.6% / max +22.9%** above tiktoken — a 400-tiktoken
budget is ≈442 real e5 tokens (worst case 492), inside the 512 ceiling with
thin headroom. The exact counter buys that headroom back at the price of a
`transformers` hot-path dependency and a first-run tokenizer download; the
≥480-token warning is the guard that actually matters. The
`TokenBasedStrategy(length_function=…)` seam accepts an exact counter with
zero registry changes if budgets ever grow.

## Rationale

- **The data cannot justify a change.** With hybrid/reranked saturated at
  1.000 under every strategy, swapping a load-bearing default would be
  preference wearing a benchmark costume. `recursive_character` also keeps
  v1's eval history comparable.
- **The recommendation is still recorded**: `markdown_aware`'s clean sweep
  at k=5 plus heading provenance make it the presumptive winner *if* a
  discriminative corpus separates the strategies — that re-test is the
  standing follow-up, same as ADR-004's fixture-growth trigger.
- **Throughput is a chunk-count story**, so no strategy is disqualified on
  ingest cost; `token_based`'s 6-chunk/86 s run is an artifact of bigger
  chunks (400 *tokens*), not a faster chunker.

## Consequences

- `CHUNKING_STRATEGY=recursive_character` stays in `config.py` /
  `.env.example`, now benchmark-decided; all five strategies are one config
  flip (or one GUI setting) away, and changing one flags the corpus stale —
  content hashes don't change, so only `ingest --reingest` applies it.
- **`CHUNK_SIZE`/`CHUNK_OVERLAP` units are per-strategy** (documented in
  `config.py` and each module): **characters** for `recursive_character` /
  `markdown_aware`; **tokens** for `token_based` / `docling_hybrid` /
  `semantic` (`semantic` splits on embedding-similarity boundaries — 95th
  percentile cosine-distance outliers — with the token budget as a re-split
  ceiling; `docling_hybrid` ignores overlap and merges peers instead).
- The sweep is re-runnable (`uv run --group eval main.py eval`) and rides
  the fact-anchored refs, so future strategies join with one registry file
  plus golden `fact` coverage — no re-authoring of indices.
