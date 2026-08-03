# ADR-017: The GraphRAG engine — LightRAG (bake-off-decided)

**Status:** Accepted (2026-08-03) — drafted from final measurements
2026-07-31, owner-accepted per the
[ADR-011](ADR-011-chat-engine-condense.md) precedent (*"the numbers said
X"* is the record). The acceptance gate in Decision #2 **ran the same day
and passed** — see the amendment below; the engine choice stands.

## Context

GraphRAG (spec_graphrag §8) is a **peer** retrieval corpus beside
chunk-RAG: an iMessage `chat.db` becomes a knowledge graph with
per-message provenance, so the system can answer the three question
shapes the chunk pipeline structurally cannot — *aggregation* ("what does
Bob think about computers", scattered over years and threads),
*verification* ("did Jane ever say that, and when"), and *relation*
("who told me it was Bob's birthday"). The spec froze every
engine-shaped parameter behind this ADR.

Stage 1 built the engine-independent half and then measured: the
message-source registry family + the iMessage parser (`varagity/graph/`),
a deterministic synthetic corpus with golden QA, the `GraphEngine`
protocol/registry with three adapters, and the `eval graph` harness —
which survives as the permanent regression guard the way `eval chat` did
for ADR-011.

**The three seats** (survey + R1 closure, 2026-07-24): **LightRAG**
1.5.4 and **cognee** 1.4.0 (the two stack-fit finalists) plus
**graphiti-core** 0.29.2 (whose survey blockers — no communities, no
embedded storage — both dissolved under source inspection). Excluded:
MS GraphRAG (delete is `not_planned`, parquet/LanceDB only),
nano-graphrag / fast-graphrag (dormant), HippoRAG 2 / Semantica
(screen-passed only), and the Karpathy wiki pattern (a build, not an
adoption — it stays the named fallback if all three fail small-model
extraction).

Decision criteria, in the spec's own order: offline fit → small-model
tolerance → incremental insert/delete → provenance granularity →
query-mode fit for Q1–Q3 → storage fit (existing Postgres > files in a
named volume > a new service) → graph export for the view → indexing
cost on a single-slot llama.cpp → license/maintenance → fork-and-own
viability.

## The bake-off

**Corpus** — synthetic, deterministic (`seed 13`), built by
`varagity/eval/graph_fixtures.py` into a real SQLite `chat.db`: **10,001
messages, 50 threads, 2013-01-02 → 2024-12-26**, of which 226 are
hand-scripted (they carry every golden anchor, both Apple epoch eras, 7
tapbacks, and 2 `attributedBody`-only rows so the typedstream decoder
runs inside the eval too) and 9,775 are filler that a whole-word
blocklist keeps clear of every golden term. Parsing runs through the
product path (`find_message_source` → `batch_for_path`), not a test
shortcut.

**Scoring** — 17 golden questions (5 aggregation / 5 verification / 7
relation), house style, **no LLM judge**: `fact_recall` is
AND-of-OR-groups matched case-insensitively against the answer text;
`provenance_recall` is `|surfaced guids ∩ required guids| / |required|`.
Question `kind` is never shown to an engine.

**Stack** — host mode against the live services: llama.cpp on the 2080 Ti
serving one slot (`Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf`, a
*reasoning* model), infinity serving `multilingual-e5-large-instruct`
(1024-dim). Engines self-store in per-engine working directories; no
compose changes, no testcontainers.

Two asymmetries are baked into the comparison and must be read with the
tables:

1. **LightRAG and cognee were scored on their own answer pipelines.**
   Graphiti has none — its search returns edge facts — so its answers are
   *our* synthesis (one grounded `LLMClient.generate` + `clean_response`
   over the retrieved facts). The `fact_recall` column therefore measures
   "engine retrieval + engine prose" twice and "engine retrieval + our
   prose" once.
2. **Graphiti ran a capped slice.** At its measured **40.0 s/message** an
   uncapped 10,001-message build is **≈111 h ≈ 4.6 days**, so its seat is
   `--profile full --message-target 500` — corpus `full-mt500`: all 226
   scripted messages + 274 filler across 45 threads, so every golden
   anchor still resolves. A first capped attempt at `--message-target
   1000` died externally at 663/1000 episodes. **Graphiti's row is not
   measured over the same corpus as the other two** — its retrieval
   haystack is 20× smaller.

## Results

Results documents (under gitignored `data/`, so the numbers below are
transcribed): `20260729T133236Z-graph.json` (LightRAG, full),
`20260731T171518Z-graph.json` (cognee, full, attempt 3),
`20260730T194220Z-graph.json` (Graphiti, `full-mt500`).

### Headline

| Engine | Corpus | Index wall-clock | s/msg | Entities / relations / communities | Fact recall | Provenance recall | Mean query | Build failures | Workdir |
|---|---|---|---|---|---|---|---|---|---|
| **LightRAG 1.5.4** | 10,001 | 71,692 s (**19.9 h**) | 7.17 | **347 / 841** / n-a | **0.078** | **0.515** | 44.2 s | **0** | 28 MB |
| cognee 1.4.0 | 10,001 | 51,087 s (**14.2 h**) | **5.11** | 233 / 649 / n-a | **0.382** | 0.132 | **1,553 s** (25.9 min) | 2 (contained) | 47 MB |
| graphiti-core 0.29.2 | **500** (capped) | 20,021 s (5.6 h) | **40.04** | 106 / 94 / **0** | 0.373 | 0.490 | **10.4 s** | 1 (deliberate) | 2.0 MB |

`n-a` means *the engine cannot report this*, not zero: neither
document-shaped engine has a community tier to count. Graphiti's `0` is
real — see the community failure below.

### By question kind (fact recall / provenance recall)

| Engine | Aggregation (Q1) | Verification (Q2) | Relation (Q3) |
|---|---|---|---|
| LightRAG | 0.200 / **0.750** | 0.067 / 0.400 | 0.000 / 0.429 |
| cognee | 0.133 / 0.050 | **0.667** / 0.000 | 0.357 / 0.286 |
| graphiti | 0.200 / 0.267 | 0.267 / 0.400 | **0.571** / **0.714** |

Mean latency per kind tells the operability story the averages hide —
cognee: **5,052 s** (aggregation) / 163 s / 47 s; LightRAG: 49.0 / 53.7 /
33.9 s; graphiti: 13.7 / 8.5 / 9.4 s. A single cognee aggregation
question peaked at **6,912 s (1 h 55 m)**.

### Provenance is a granularity question, not just a recall number

Both document-shaped engines cite *transcripts*, so a "cited message" is
every message in a matched thread-day. Recall alone flatters them;
recall against the fraction of the corpus they cited does not:

| Engine | Mean guids surfaced / query | Share of corpus cited | Provenance recall | Lift over citing at random |
|---|---|---|---|---|
| LightRAG | 1,722 | 17.2% | 0.515 | **3.0×** |
| cognee | 1,464 | 14.6% | 0.132 | **0.9×** (at chance) |
| graphiti | **9.3** | **1.9%** | 0.490 | **26.4×** |

Graphiti's per-message episodes are the real thing: nine cited messages,
half the required anchors. cognee's document-grain mapping (CHUNKS →
`document_name`) is indistinguishable from citing a seventh of the
archive at random once the corpus is big. And **only Graphiti returned
structured relations per answer** (10 per query, every run); LightRAG's
context pass and cognee's CHUNKS pass surfaced message ids only, never
question-scoped entities or relations — the evidence payload spec_graphrag
§4.2 asks for exists today in exactly one of the three.

### Scale is where the ranking changes

The 226-message smoke run (`20260727T064146Z-graph.json`, all three
engines, 4 h 51 m) ranked them differently — which is precisely why the
full profile was mandatory:

| Engine | Smoke fact / prov (226 msgs) | Full fact / prov | Incremental (+10 msgs, smoke) |
|---|---|---|---|
| LightRAG | 0.373 / 0.765 | 0.078 / 0.515 | **140 s** |
| cognee | **0.569 / 1.000** | 0.382 / 0.132 | 394 s |
| graphiti | 0.324 / 0.441 | 0.373 / 0.490 | 294 s |

cognee's perfect smoke provenance was an artifact of corpus size (at 226
messages one transcript *is* a large share of everything) and collapsed
to below-chance at 10,001. All three incrementals confirm criterion
§8.2#3 — only the 10 new messages paid LLM cost in every engine.

### Robustness (criterion §8.2#2 — the local-model axis)

- **LightRAG: zero failures** across 10,001 messages and 19.9 h, and the
  only engine where the `<think>`-strip template applies *fully* (a
  custom async `llm_model_func` runs `clean_response()` on every
  extraction, gleaning, and answer completion). Its low fact recall is
  concentrated entirely in the answer stage: of 17 turns, **5 came back
  empty**, **5 were a bare `### References` list**, **5 were explicit "I
  do not have enough information" refusals**, and only 2 were real prose
  (one of which scored a perfect 1.000). The empties are the reasoning
  model leaving its prose inside `<think>` — the adapter strips it as
  designed, and nothing remains.
  **In 6 of the 17 turns LightRAG surfaced ≥ 0.75 of the required source
  messages and still scored zero facts** (a seventh scored 0.33 while
  citing all of them): retrieval reached the evidence and the answer
  composition did not use it. That is the observation the decision below
  rests on — and, because "cited the right guids" is not the same as "put
  the right lines in the prompt", it is an observation the acceptance
  gate exists to test rather than assume.
- **cognee: three full-profile attempts, two total losses.**
  1. `20260728T171835Z-graph.json` (**preserved, not deleted** — an
     engine answering all 17 goldens off an empty graph is robustness
     data): the build died 76 minutes in with **0 entities**, and the 17
     queries then spent 7.7 h at latencies quantized to 600 s multiples —
     the fingerprint of LiteLLM's unconfigured 600 s completion deadline
     firing on the *queue* of an uncapped extraction fan-out against a
     single-slot server, with instructor re-sending each victim.
  2. Attempt 2 (log only, no results document): with the fan-out bounded
     (`REQUEST_TIMEOUT=3600`, `chunks_per_batch=8`) extraction stopped
     timing out and started **think-spiralling** — prompt 4,805 +
     completion 11,579 tokens = a full 16,384-token window returned
     `content=''` with `finish_reason='length'`; instructor's retry
     embeds the failed response verbatim (4,805 → 9,450 prompt tokens),
     so each attempt is worse than the last; the exhausted `RetryError`
     raised `PipelineRunFailedError` and cognee's
     `cognify_rollback_handler` deleted **every node and edge of the
     run**. 3.1 h, 0 entities, no partial progress. Nothing in cognee
     caps a completion's length, and LiteLLM+instructor offers **no
     injection point for `<think>` stripping** (verified against the
     installed sources).
  3. Attempt 3 — the scored run — survived only because ingestion was
     restructured *for this engine*: add+cognify in groups of 100
     documents, so one rollback is one group's blast radius. Both groups
     still recorded an `IncompleteOutputException` max-tokens failure and
     each triggered its own provenance rollback, and the graph
     nonetheless finished at 233 entities / 649 relations and answered
     all 17 questions. The isolation worked; it is also the reason there
     is a cognee row at all.
  - The same un-strippable spiral is what the **25.9-minute mean query
    latency** is: a spiralling search call costs `REQUEST_TIMEOUT`
    (3,600 s) × retries, and 6 of 17 answers came back empty *after* that
    spend.
- **Graphiti: one recorded failure per build, and it is deliberate.**
  `build_communities()` is skipped: 0.29.2's `label_propagation` is a
  synchronous `while True` with **no iteration cap** and oscillates on
  non-trivial graphs — it burned 9 h of pure CPU on the 226-message graph
  before being killed. The per-episode path (`update_communities=True`)
  is separately broken (it unpacks a variable-length gather, so every
  episode touching ≠2 community nodes raises). **Both community paths in
  0.29.2 are defective**, 0.29.2 is the current release, and communities
  are the aggregation tier Graphiti was seated for. Deliberately not
  monkey-patched — an iteration cap would change the engine under test.
  Its answers were also the most consistently *present* (2 of 17 empty).

## Decision

**LightRAG is the recommended engine**, with one measured condition
attached and a named fallback order. Three sub-decisions follow from it.

1. **Stage 2 generates its own answers.** LightRAG is used in
   retrieval-only mode (`only_need_context=True`, a first-class API) and
   the repo's existing grounded-generation path writes the prose — which
   is required anyway for the `[SOURCE]` citation contract, the SSE token
   stream, and `clean_response()`.
2. **Acceptance gate (cheap, and it can overturn this ADR).** The
   argument above claims LightRAG's 0.078 is a number about a component
   we would not ship. That claim is **not yet measured**, so before the
   engine is locked into stage 2: re-score the *already built* 10,001-
   message graph — it is on disk, 28 MB — with our synthesis in place of
   LightRAG's answer pass, using the harness's existing escape hatch
   (`eval graph --engine lightrag --mode <synthesis> --skip-build`,
   minutes, no re-index; the Graphiti adapter already contains the
   synthesis code to lift). **Gate: fact recall ≥ 0.37** (its own smoke
   score, and cognee's full-profile score is 0.382).
   *Outcome (2026-08-03): passed at **0.4216** on `mix+synthesis` with
   the e5 query prefix — see the amendment below.*
3. **If the gate fails**, the fallback order is (a) **Graphiti** with
   extraction moved off the single slot — it wins provenance and latency
   outright and only loses on arithmetic; (b) **cognee** with a
   non-reasoning extraction model, which is what its failure mode
   actually indicts. Re-deciding costs one `eval graph --profile full
   --engine <name>` run because the seam and both loser adapters are
   still in the tree.

### Rationale

- **Criterion #2 (small-model tolerance) is where this stack lives, and
  LightRAG owns it.** It is the only engine with a complete
  `<think>`-strip hook — the trap that already cost this repo twice
  ([ADR-011](ADR-011-chat-engine-condense.md) condense,
  [ADR-016](ADR-016-hyde-retrieval.md) HyDE) — and it is the only engine
  that finished the full profile on the first attempt with zero failures.
  Its failure mode is an empty answer that costs 44 s. cognee's failure
  mode costs an hour per query and rolls back extraction.
- **The failures land in different layers, and only one of them is
  ours.** LightRAG's measured weakness is answer composition — the layer
  stage 2 replaces by contract. cognee's weaknesses are extraction
  robustness and retrieval provenance, which we cannot replace: with no
  `<think>` hook and no completion cap, the spiral is structural on a
  reasoning model, and question-scoped graph records are simply not
  reachable through cognee 1.4's search API (`INSIGHTS` is gone).
  Graphiti's weakness is arithmetic that scales with the one thing that
  grows for life.
- **Criterion #8 disqualifies Graphiti on the spec's own corpus size,
  not on quality.** 40.0 s/message is 4.6 days for 10,001 messages, so it
  was never run uncapped at all — two capped attempts, the first of which
  (`--message-target 1000`) still died externally at 663/1000 episodes;
  a real decade of iMessage is several times that. It is the best engine
  here on provenance (26.4× lift), latency (10.4 s), relation questions (0.571),
  and the only one whose evidence carries relations — so it is the first
  engine to revisit the moment extraction stops being one-at-a-time
  (batched serving, a smaller extraction model, or the GLiNER2-style
  non-LLM extraction the spec keeps as a lever). Its broken community
  layer removes the aggregation story it was seated for in the meantime.
- **cognee's storage advantage does not survive its query latency.** Its
  plain-SQL Postgres adapter is the best stack fit anyone offered
  (criterion #6's top preference), and its verification answers were the
  best prose in the bake-off (0.667, with correct dated quotes). But a
  25.9-minute mean — 1 h 55 m worst case — is not a chat turn, the cause
  has no fix on this stack, and its provenance is at chance.
- **Provenance grain is a known, priced regret.** LightRAG's 3.0× lift
  over random citation is real but weak next to Graphiti's 26.4×. It is
  tolerable *here* because a transcript's key is
  `{thread_id}::{day-span}` — a citation resolves to a thread and a day,
  which is exactly the shape a Q2 "on that day" answer needs — and it is
  tunable: smaller `thread_transcripts` caps trade extraction calls for
  provenance grain, measurable with this same harness.

### Storage backend (winner-conditional, stage-2 directive)

**File storage in a new named volume (`graphdata`), no new compose
service.** The four storage classes the adapter already pins stay:
`JsonKVStorage` / `NanoVectorDBStorage` / `NetworkXStorage` /
`JsonDocStatusStorage`.

Criterion §8.2#6 prefers the existing Postgres — but that preference is
not reachable for LightRAG: its Postgres *graph* storage requires the
**Apache AGE** extension, and this repo's database is
`pgvector/pgvector:pg16`. Buying a 28 MB graph with an image swap under
the primary corpus, conversations, settings, and migrations (or with a
second Postgres service) is the wrong trade. Files in a named volume are
criterion #6's second preference, cost one volume, and inherit the
`esdata` reset story. Measured footprint: **28 MB per 10,001 messages**
(~280 MB extrapolated at 100k) — revisit if that stops being true, or if
one image ever ships AGE and pgvector together.

**Consequence to design around:** file storage is single-writer. The API
already runs the flows in-process, so builds and queries share one
process by construction; a CLI graph build **must not** run against a
live API's volume. Stage 2 states this in the runbook the way
[ADR-013](ADR-013-corpus-gauges-vs-counters.md)'s per-process metrics
caveat is stated.

### Visualization library (spec_graphrag §9, stage-2 directive)

**sigma.js + graphology, via `@react-sigma`** (all MIT) — the
engine-synergy rule: LightRAG's own WebUI graph panel is React 19 + Bun +
sigma.js, our exact toolchain, so adopting LightRAG hands us a working
reference implementation. `react-force-graph-2d` (MIT) stays the
documented fallback if `@react-sigma`'s React-19 support regresses.

**Recorded trap:** `@cosmos.gl/graph` is MIT, but the polished wrappers
`@cosmograph/cosmograph` and `@cosmograph/react` are **CC-BY-NC-4.0** —
non-commercial only. Considered and avoided: if GPU-accelerated layout is
ever needed, build on the MIT engine, never the wrapper.

Scale is settled by measurement, not guesswork: 10,001 messages produced
**347 entities / 841 relations**, so the view renders hundreds-to-low-
thousands of nodes and every candidate copes. Ergonomics and synergy
decide. The graph must never render one node per message
([ADR-015](ADR-015-codebase-map.md)'s hand-rolled-SVG scope explicitly
does not extend to this).

### Degrade semantics (stage-2 directive)

`GRAPH_ENABLED` (default `true`) is a **kill switch, not an engine** —
the `RERANK_ENABLED` / `CONDENSE_ENABLED` shape
([ADR-006](ADR-006-reranking-wired.md),
[ADR-011](ADR-011-chat-engine-condense.md)) — deliberately orthogonal to
`GRAPH_ENGINE`, which keeps its registry-hardcoded validator tuple and
tuple↔registry regression test:

- A **graph-targeted question** with `GRAPH_ENABLED=false` is answered
  from the chunk-RAG corpus and logged at INFO. The turn still persists
  the requested target corpus and engine name with graph evidence NULL —
  the honest record of a degrade, exactly how a degraded
  `condense_context` turn persists its engine with `condensed_query`
  NULL.
- A **graph engine failure at query time** (after the client's own
  retries) degrades identically per-turn at WARNING: a chunk-RAG answer
  beats a 500.
- A **graph upload or build** with `GRAPH_ENABLED=false` has no
  meaningful degrade — there is nowhere else to put messages — so the
  route returns a structured error (`graph_disabled`), never a silent
  no-op. This is the page-preview posture (`available:false` + a machine
  reason), not an exception.

### Defaults (stage-2 directive)

Every value below was pinned by a live gate or a measurement in stage 1,
not chosen on paper:

| Setting | Value | Why |
|---|---|---|
| `GRAPH_ENGINE` | `lightrag` | this ADR |
| `GRAPH_ENABLED` | `true` | kill switch, above |
| Query mode | `hybrid` | the measured primary; `local` / `global` / `naive` remain reachable via a `GRAPH_QUERY_MODE` setting |
| Answer generation | repo's grounded generation over `only_need_context=True` | decision 1, gated by decision 2 |
| Transcript cap | 8,000 chars, day-split, key `{thread_id}::{day-span}` | upsert identity; also sizes every extraction prompt |
| `MAX_ASYNC_LLM` / `MAX_PARALLEL_INSERT` | `1` / `1` | llama.cpp serves one slot (`--parallel 1`) |
| `EMBEDDING_FUNC_MAX_ASYNC` | `2` | infinity tolerates it; embeddings are not the bottleneck |
| `LLM_TIMEOUT` / `EMBEDDING_TIMEOUT` | `1800` s / `600` s | generous by design: on a single slot a client deadline fires on the *queue*, which is exactly how cognee's attempt 1 died |
| Per-call `max_tokens` | 2,048, clamped to the window with 512 headroom, floor 256 | llama.cpp hard-500s at the window |
| Insert batch | 20 documents | bounded fan-out per insert call |
| Storage classes | `JsonKVStorage` / `NanoVectorDBStorage` / `NetworkXStorage` / `JsonDocStatusStorage` | pinned so a stray environment cannot redirect a build at a real database |
| Embedding dim | 1,024 | e5-large-instruct, asymmetric prefixes owned by the engine's embedding hook |

**Open lever, deliberately unmeasured:** LightRAG's `global` mode
(dual-level keywords, *not* community reports) was never run. Aggregation
is everyone's weakest slice (0.133–0.200), so
`eval graph --engine lightrag --mode global --skip-build` against the
existing graph is the cheapest experiment stage 2 has. No promotion
without that number.

### Decode path (R3, closed)

**`pytypedstream` (import `typedstream`) is the decode path** — verified
2026-07-25 against the owner's *real* iOS-backup `sms.db`, not a fixture:
`attributedBody`-only rows (NULL `text`, the newer-OS default) decode to
correct bodies, both Apple epoch eras render sane dates across a decade,
tapbacks fold onto their targets, and counts are plausible against
Messages.app. It is LGPL-3.0-or-later — fine as an imported pure-Python
library — and one innocuous owner-reviewed blob rides in the test suite
(`tests/fixtures/graph/attributed_body_sample.bin`) so the decoder has a
regression test without any real data in the repo.

**`imessage-exporter` stays a documented contingency, not built code.**
It is GPL-3.0 (legal only as a subprocess preprocessor) and its txt export
**loses `message.guid`** — which is the identity the entire design rests
on: upsert across overlapping exports, per-message provenance, and golden
`required_guids` all die with it. Reach for it only if a future macOS
format defeats the pure-Python decoder, and accept the provenance loss
explicitly if so.

### Loser adapters

Removed **at the start of stage 2** (not now — the ADR's numbers must
stay reproducible while it is under review): delete
`varagity/graph/engines/cognee.py` and `graphiti.py`, their `bakeoff`
group entries, mypy overrides, and import-lightness test names. Deleting
cognee also releases `[tool.uv] override-dependencies =
["rich>=15.0.0"]`, which exists solely for cognee → instructor's
`rich<15` cap, and with it the five transitive downgrades that group
imposes on the whole lock (pandas, packaging, websockets, jiter, redis).

**The seam stays**: the `GraphEngine` / `GraphSession` protocols, the
registry, `varagity/graph/records.py`, and `render.py`'s `merge_batches`
+ `thread_transcripts`. `episode_payloads` belongs to the episode-shaped
engine and goes with the Graphiti adapter (git history preserves both,
and re-seating an engine is one file) — a judgment call the stage-2
planner may reverse if it wants the rendering kept warm.

## Consequences

- **Stage 2 has no open engine questions.** Engine, query mode, answer
  ownership, storage backend, viz library, degrade semantics, extraction
  budgets, concurrency pins, and the decode path are all fixed above; the
  one deliberately open item is `global`-mode aggregation, and it has a
  named cheap experiment.
- **The first backfill is a multi-day background job, not a click.** At
  the measured **7.17 s/message** a 10,001-message corpus is ~20 h, and
  the owner's real archive is larger and grows for life. The stage-2
  build flow must therefore be resumable and observable (per-batch
  durability, real progress, and a bounded or date-ranged ingest escape
  hatch), and the corpus UI must not present it as an operation that
  finishes in a session. Incremental growth is cheap by comparison —
  only new messages pay LLM cost (measured: 140 s for 10 new messages).
- **Provenance is document-grain.** The evidence panel cites a thread and
  a day, not a single message, and the honest UI copy says so. Per-message
  provenance is a measured 26.4×-vs-3.0× regret against Graphiti and the
  first thing to re-open if Q2 answers feel unanchored in practice.
- **Answers are ours, retrieval is the engine's** — the same split the
  repo already enforces for HyDE and condense: generated text may steer
  search, never the answer contract.
- **`eval graph` becomes the regression guard** (the ADR-011 shape):
  `--profile smoke` is the cheap pre-merge check (all three engines,
  ~4 h 51 m today; minutes for one engine with `--skip-build`), and
  `--profile full --engine <name>` is how the decision gets re-taken.
  Note the recorded caveat: `--skip-build` re-scores fact recall and
  latency but **not** provenance for engines whose citation→guid index is
  session state built during `build()`.
- **The corpus is discriminative, not saturated** — full-profile fact
  recall spread 0.078–0.382 and per-kind spread 0.000–0.667, versus the
  1.000-everywhere saturation that has caveated
  [ADR-004](ADR-004-ocr-engine-choice.md),
  [ADR-006](ADR-006-reranking-wired.md), and
  [ADR-016](ADR-016-hyde-retrieval.md). The long-deferred discriminative
  eval corpus now exists, in message form. It remains **synthetic**, and
  substring fact-anchoring rewards phrasing that contains the anchor —
  both are the price of having no LLM judge.
- **Revisit triggers**, explicitly: extraction moves off the single slot
  (Graphiti becomes viable); the served model stops being a reasoning
  model (cognee's failure mode disappears, and LightRAG's empty answers
  with it); graphiti-core ships a community fix past 0.29.2; cognee gains
  a completion-length cap or any `<think>` injection point. Each is one
  `eval graph --profile full` run away from a new ADR.

## Amendment (2026-08-03): the acceptance gate — passed

Decision #2's gate ran the day the ADR was accepted, against the same
on-disk 10,001-message graph (`--skip-build`, no re-index; six configs,
17 golden questions each, ~6 min a run). The numbers, from
`data/eval/results/20260803T{114412,115943,120937,121632,122256,123312}Z-graph.json`:

| Config | Fact recall | Agg / Verif / Rel | Mean latency |
|---|---|---|---|
| engine-composed `hybrid` (the bake-off row) | 0.078 | 0.200 / 0.067 / 0.000 | 44.2 s |
| `hybrid+synthesis` | 0.108 | 0.200 / 0.167 / 0.000 | 21.6 s |
| `global+synthesis` | 0.235 | 0.333 / 0.267 / 0.143 | 34.4 s |
| `hybrid+synthesis`, e5 query prefix | 0.265 | 0.467 / 0.233 / 0.143 | 23.8 s |
| `global+synthesis`, e5 query prefix | 0.275 | 0.333 / 0.400 / 0.143 | 20.0 s |
| **`mix+synthesis`, e5 query prefix** | **0.4216** | 0.600 / 0.433 / 0.286 | 35.4 s |

**Verdict: 0.4216 ≥ 0.37 — gate passed, engine locked** (owner-accepted
2026-08-03). The winning config also beats both fallbacks' full-profile
fact recall (Graphiti 0.373, cognee 0.382), so Decision #3's fallback
order was never exercised. The gate's claim held in a refined form — the
answer stage was one of **three** stacked problems, each worth measuring
separately:

1. **Our synthesis** (the thing the gate was designed to test) fixed the
   answer-stage catastrophe — the 5 empty / 5 references-only / 5
   refusals became grounded, honest answers — but alone moved fact recall
   only 0.078 → 0.108. Most residual misses said *"the evidence does not
   mention X"*, and repro confirmed they were true: the fact-bearing
   chunks were not in the retrieval.
2. **The e5 query prefix** — stage 1's recorded deviation, now the
   `GRAPH_QUERY_PREFIX` setting riding LightRAG's
   `EmbeddingFunc(supports_asymmetric=True)` seam (verified live: 1.5.4's
   query paths pass `context="query"`; passages stay unprefixed, so the
   already-built graph needed no re-embedding) — was worth more than the
   synthesis itself: 0.108 → 0.265 on `hybrid`.
3. **`mix` mode** supplies what needle-fact questions still lacked:
   `hybrid` retrieves only KG-mediated chunks (its raw search logs read
   "0 vector chunks"), so facts no entity/relation path reaches — the
   birthday date, the cabin inventory — never enter the context. `mix`
   adds direct vector retrieval over chunks: 0.265 → **0.4216**.

**Shipped defaults, chosen from the numbers** (owner-accepted):
`GRAPH_QUERY_MODE=mix`, `GRAPH_QUERY_PREFIX=true` — landing with their
consumers per the stage-2 plan (the prefix default flips in its Phase 2;
the mode setting is born in its Phase 4). The ~12 s mean-latency premium
of `mix` over `hybrid` is priced in. Relation questions remain LightRAG's
weakest kind (0.286 vs. Graphiti's 0.571) — the document-grain regret
this ADR already priced, unchanged.

Measurement deviations, recorded rather than hidden:

- **`SYNTHESIS_MAX_TOKENS` 1024 → 2048.** At 1024 the reasoning model
  burned the whole budget inside its reasoning stage on roughly half of
  calls; llama.cpp routes an unfinished think phase into
  `reasoning_content`, so the non-streaming `generate()` saw empty
  `content` and 5 of 17 answers scored as infrastructure zeros (the
  condense 128→512 and HyDE 512→1024 precedents, one size larger).
  ~2/17 residual empties persist at 2048, so gate numbers slightly
  *under*-measure every synthesis config.
- **The `mix` config was a diagnostic beyond the plan's three named
  runs** — added once repro showed the residual misses were chunk-
  targeting, not answering. It turned out to be the passing
  configuration.
- **Provenance reads null under `--skip-build`** (the known caveat above;
  the gate is fact-recall-only by design).

The LightRAG **rerank hook** (`rerank_model_func` → the infinity
cross-encoder already serving `/v1/rerank`) was considered and left
unmeasured: it changes retrieval against the measured bake-off and
deserves its own pass. It is the first lever to try if `mix`'s relation
recall needs to move.
