# ADR-016: HyDE as a composing retriever (default stays `hybrid`)

**Status:** Accepted (2026-07-23)

## Context

The retrieval ladder tops out at `reranked` (≈67% tier,
[ADR-006](ADR-006-reranking-wired.md)). HyDE — Hypothetical Document
Embeddings (Gao et al. 2022, *Precise Zero-Shot Dense Retrieval without
Relevance Labels*) — attacks the other end of the pipeline: instead of
re-ordering what dense retrieval found, it changes **what the dense arm
searches with**. An LLM writes a short hypothetical answer passage for the
query; that passage, embedded with the *document* encoder, lands in the
corpus's own vector space, where its nearest neighbors are the chunks that
*look like* the answer. Query↔passage asymmetry (short question vs.
paragraph-shaped chunk) is exactly where e5's instruct-mode bridging can
fall short, and exactly what HyDE sidesteps. The method is added **to be
evaluated** — the interesting question is whether it stacks with
cross-encoder reranking — not to move the shipped default.

## Decision

A **composing `hyde` retriever** (`varagity/retrieval/hyde.py`, a registry
file — the `reranked` shape, ADR-006), not a flow stage and not a fork of
fusion:

1. `encode_query()` makes one non-streaming LLM call
   (`HYDE_MODEL_TYPE`, default the chat model; `HYDE_MAX_TOKENS=1024` —
   the `CONDENSE_MAX_TOKENS` lesson with margin: measured live on this
   stack's reasoning model, a 512 cap starved ~1/3 of generations into
   the empty-passage fallback because the hidden reasoning plus a 3–5
   sentence passage doesn't fit; 1024 generated 3/3) and
   post-processes exactly like the condense stage: `clean_response()`
   (`<think>` stripping — an unstripped block embedded as the probe
   silently destroys retrieval), a `PASSAGE:` label-echo strip, then
   empty/overlong guards (`HYDE_MAX_CHARS=2000`).
2. The passage is embedded in e5 **passage mode** (`embed_passages` — the
   paper's document-encoder choice; the corpus was embedded the same way),
   *not* query mode.
3. `retrieve()` hands the base retriever (`HYDE_BASE_METHOD`, default
   `hybrid`; validated ∈ {semantic, hybrid}) the **original query text**
   plus that vector through the existing `query_vector` seam —
   **dense-arm-only substitution**: a `hybrid` base's BM25 arm keeps exact
   keyword recall (error codes, identifiers), traces pass through
   untouched, and the answer prompt never sees the hypothetical (the
   spec_v3 §4.2 invariant extended: generated text may steer *search*,
   never *answers*).

**Stacking with rerank goes one way.** `RERANK_BASE_METHOD` gains `hyde`,
so the pairing is `RETRIEVAL_METHOD=reranked` + `RERANK_BASE_METHOD=hyde`:
HyDE shapes the over-fetched candidate pool, then the cross-encoder scores
candidates against the **user's real query**. The reverse nesting
(`HYDE_BASE_METHOD=reranked`) is config-rejected — it would cross-encode
against the hypothetical, judging relevance to a guess. `bm25` is likewise
rejected as a HyDE base (it never consumes a query vector — the LLM call
would buy nothing), as is recursion.

**Failure is a fallback, not an error** (the spec_v3 §4.6 posture): a
raised LLM call (after client `tenacity` retries), an empty cleaned
passage, or an overlong one all degrade to the base method's raw-query
retrieval at `WARNING`. `HYDE_ENABLED=false` is the kill switch,
orthogonal to method selection — the `RERANK_ENABLED` shape, checked
inside the retriever.

Surfaces follow the registry conventions with zero bespoke wiring: the
settings drawer and quick-toggles pick `hyde` up from the registry-derived
choices, `GET /api/config` lists it, per-request `ChatOverrides` accept
it, the passage-generation sub-stage is timed as `stage="hyde"` (inside
the flow's `embed` observation, mirroring `rerank` inside `retrieve`), and
verbose level 2 renders the passage (`v_hyde`).

## Measured results

Eval matrix configs 6–7 (`hyde_contextual`, `hyde_rerank_contextual`), 20
golden queries over the fixtures corpus, eval pins `HYDE_ENABLED=true`,
`HYDE_BASE_METHOD=hybrid`, `HYDE_MAX_TOKENS=1024`, `HYDE_MAX_CHARS=2000`
(each HyDE config pays one live-LLM generation per query; configs 6 and 7
generate independently, so sampling noise between them is part of what the
pairing comparison measures):

- Run `20260723T142318Z-matrix.json`: all seven configs — both HyDE forms
  included — at **1.000 recall/pass for k ∈ {5, 10, 20}**, with 40 live
  passage generations behind configs 6–7. `hyde ≥ hybrid` and
  `hyde+rerank ≥ reranked` hold as equalities: the fixtures corpus
  saturates the ladder (the ADR-006/ADR-004 caveat over again), so the
  harness proves the wiring end-to-end; the *discriminative* HyDE verdict
  waits on the deferred cookbook-corpus swap, same as reranking's.
- 4 of those 40 generations (10%) came back empty even at the 1024 cap —
  the reasoning model's sampling tail — and every one degraded to
  raw-query retrieval per design, costing nothing on this corpus. The
  fallback rate is a number to re-check whenever the served model
  changes.
- Passage generation costs a mean **~12.5 s per query** on this stack
  (llama.cpp on the 2080 Ti, `HYDE_MAX_TOKENS=1024`, measured over 8
  varied queries at implementation; 2048 bought no reliability, only
  longer generations) — the latency any HyDE promotion has to earn back.
- The chunker sweep deliberately excludes the HyDE configs: the
  hypothetical passage depends only on the query, never on chunk
  boundaries, so a per-strategy re-measure would re-pay every LLM
  generation to measure the same transformation.

## Consequences

- The default is **unchanged** (`RETRIEVAL_METHOD=hybrid`) — promotion of
  `hyde` (in either form) is a future benchmark-decided ADR, the ADR-011
  pattern; the eval matrix now carries the two configs that decision
  needs.
- Every `hyde` question pays an LLM round-trip *before* retrieval, on the
  same llama.cpp server that generates answers — latency and contention,
  visible as the `hyde` stage histogram. `condense_context` + `hyde`
  stacks two pre-retrieval LLM calls; both engines' rewrites feed HyDE
  coherently (condense resolves references, HyDE hypothesizes), but the
  latency compounds.
- The passage is never persisted and never reaches the evidence panel:
  per-chunk traces are untouched, so a HyDE-retrieved answer explains its
  *ranking* but not the probe that produced the pool. If evaluation
  promotes HyDE, surfacing the passage (the `condensed_query` treatment —
  SSE retrieval event + a message column) is the follow-up.
- `RERANK_BASE_METHOD`'s vocabulary is no longer static: settings UIs and
  tests read it from the config validators / catalog, which both grew in
  lockstep here (the tuple↔registry regression tests cover the pair).
