# ADR-013: Store-derived corpus gauges over ingest-counter arithmetic

**Status:** Accepted (2026-07-19)

## Context

The v2 Ingestion dashboard read **zero on every panel** while the corpus
demonstrably held 15 documents / 58 chunks — with the underlying metrics
*correct*. The failure was PromQL semantics, not instrumentation:
ingestion is rare and bursty, and the panels asked counters
rate-of-change questions (`increase(varagity_ingest_docs_total[$__rate_interval])`)
that are structurally zero for this workload. Two facts compound:

- a counter sits flat for hours between ingests, so any short window
  reads 0; and
- **a labelled counter's child series is *born at its full value***.
  `varagity_ingest_docs_total{file_type="md"}` does not exist until the
  first `.inc()` after a process start — Prometheus's first sample of it
  is already `4`, there is no `0 → 4` rise inside the series, and
  `increase()` extrapolates *within* a series, so it answers **0 over
  any window, `$__range` included**.

The asymmetry that proved the diagnosis: the *unlabelled*
`varagity_contextualize_latency_seconds` histogram is initialised to 0
at definition (process start), so its rise **is** observed — which is
why the latency-quantile panel worked while every counter panel sat at
zero, and why spec_v3 §6.1's proposed
`increase(…_sum)/increase(…_total)` ratio returns **+Inf** (a nonzero
histogram increase divided by a labelled counter's zero).

## Decision

- **Corpus size is a gauge question answered by the store, not a counter
  question answered by the process.** A `CorpusCollector`
  (`varagity/observability/corpus.py`) queries pgvector **at scrape
  time** and exposes `varagity_corpus_documents`, `varagity_corpus_chunks`,
  `…_documents_by_type{file_type}`, and
  `…_chunks_by_strategy{chunking_strategy}` (the strategy read from the
  `chunks.metadata` JSONB — it is a key there, not a column). Companion
  `varagity_ingest_last_run_{timestamp_seconds,duration_seconds,documents}`
  gauges answer "did the last run happen, when, how big".
- **The counters and histograms stay** — they answer *per-event*
  questions (per-run deltas, latency distributions) and remain correct
  for them. The Ingestion panels were rewritten onto the gauges (and the
  raw cumulative `sum(…_sum)/sum(…_total)` form where a ratio is wanted);
  the Infra dashboard's corpus stats read `sum()` over the same gauges.
- **A scrape-time TTL of 10 s, as a module constant** — deliberately
  *not* an env setting. Prometheus scrapes every 15 s; 10 s keeps one
  scrape to one round of store queries while collapsing manual curls.
  spec_v3 §9's proposed `CORPUS_GAUGE_TTL_SECONDS` was struck: an env
  var invites 30, which silently serves a stale snapshot on every other
  scrape. Tests get their seam through the
  `CorpusCollector(ttl_seconds=…)` constructor.
- **A store outage degrades, never 500s `/metrics`**: the collector
  re-serves the last good snapshot (going stale); a fresh process with
  no snapshot emits the gauge families with no samples.
  `varagity_dependency_up` is what reports store health — these gauges
  deliberately do not double-report it.
- **A dashboard lint makes the bug class unrepresentable**
  (`tests/unit/test_dashboards.py`, spec_v3 §6.4): every panel
  expression must reference cataloged metrics with declared labels, and
  `increase()`/`rate()` over any `varagity_ingest_*` counter fails the
  unit suite. The shipped rule is **window-independent** — no
  `$__rate_interval` exemption, because no window fixes a series born at
  full value.

## Rationale

- **The store is the ground truth for "how big"** — a gauge read from
  pgvector survives API restarts by construction and is even correct
  when the write happened elsewhere. The alternative (persisting counter
  state, or a Pushgateway) adds moving parts to approximate what one
  `SELECT count(*)` states exactly.
- **Fixing the panels' PromQL instead was rejected**: there is no PromQL
  over a restart-reset, born-at-full-value labelled counter that
  recovers the corpus total. The instrument was wrong for the question,
  not the query.
- **Bounded cardinality, stated**: `file_type` (≈8) and
  `chunking_strategy` (5) are closed sets; anything per-`doc_id` is
  unbounded and deliberately absent — recorded so it isn't added
  casually.

## Consequences

- Every Ingestion panel reads truthfully against the live corpus, and
  keeps doing so across API restarts.
- **CLI visibility splits by instrument, honestly**: the corpus *gauges*
  see CLI ingests incidentally (they read the store), but the `_total`
  counters and latency histograms remain per-process and API-only — a
  CLI ingest still never reaches them
  ([ADR-007](ADR-007-observability-stack.md)'s consequence, narrowed but
  not repealed; the dashboard carries a text panel saying exactly this).
- `/metrics` now costs up to four store queries per 15 s scrape (TTL-
  bounded, ~ms at dev scale); a scrape during an outage serves stale
  numbers rather than an error — the honest trade for a non-500ing
  endpoint.
- The `increase()`-on-bursty-counter lesson is enforced, not documented:
  a future panel regressing to the counter form fails `pytest`, with the
  failure message explaining why.
- Shipped in `9406c6a` (before the v3 plan was even final) with the
  Ingestion rewrite; the Infra enrichment and the exporter-window fix
  ([ADR-007's amendment](ADR-007-observability-stack.md#amendment-2026-07-19-v3-phase-2))
  completed the observability repair.
