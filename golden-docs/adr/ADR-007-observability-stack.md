# ADR-007: In-app Prometheus metrics + provisioned Grafana

**Status:** Accepted (2026-07-14) · Amended (2026-07-19 — [below](#amendment-2026-07-19-the-observability-repair))

## Context

v1 built its three output channels and Prefect seams explicitly "so that
metrics can be layered on without refactoring" (spec §14); v2
collects that debt. Two candidate sources existed for pipeline metrics:
direct `prometheus_client` instrumentation inside the flows, or scraping the
Prefect server through a community exporter. The flows run **in-process**
from both front-ends (no workers/deployments — [ADR-003 §2](ADR-003-vertical-build-and-ops-choices.md)),
and the dashboards must work with zero click-ops (v2 DoD).

## Decision

- **Primary source: in-app `prometheus_client` collectors**
  (`varagity/observability/metrics.py` — the nine-family spec_v2 §6.2
  catalog: per-stage query latency, retrieval scores by rank, rerank delta,
  query/ingest counters, contextualize latency, LLM tokens,
  `dependency_up`). Instrumented in the flow bodies and task wrappers; the
  `rerank` sub-stage is timed inside the retriever, and the flow's
  `retrieve` observation deliberately includes it so Grafana can chart
  rerank's share.
- **Optional, off by default: the community
  `prefecthq/prometheus-prefect-exporter:3.6.1`** (orchestration-level run
  states, e.g. `prefect_info_flow_runs{state_name}` on the Infra dashboard)
  and **`nvidia/dcgm-exporter:4.5.2-4.8.1-ubuntu22.04`** (GPU VRAM/util;
  consumer-card caveat — many DCGM fields are empty on the 2080 Ti / 5060).
  Each rides its own compose profile (`prefect-exporter`, `gpu-metrics`);
  `prometheus.yml` lists their targets statically, so a disabled profile
  just reads `up == 0` and enabling one needs no config edit.
- **Provisioned-only Grafana** (`grafana/grafana:12.3.8`, scraper
  `prom/prometheus:v3.13.1`): a file-provisioned datasource with the
  **load-bearing uid `prometheus`** every checked-in dashboard references,
  file-provider dashboards mounted `:ro`, anonymous **Viewer** access so
  `:3001` renders with zero click-ops, and the image-default `admin/admin`
  left standing (dev posture, with the rest of the
  [security posture](../runbook.md#security-posture-dev-only)).
- **`GET /metrics` is a plain FastAPI route, not a mounted
  `make_asgi_app()`**: Starlette mounts 307-redirect the bare `/metrics`
  path (verified empirically at implementation), and `generate_latest(REGISTRY)`
  is exactly what that ASGI app would serve for a **single-process
  registry** — which the single-uvicorn-worker decision
  ([ADR-005 §7](ADR-005-web-stack-and-api.md)) guarantees.

## Rationale

- **In-process flows make an exporter-as-primary a category error**: the
  Prefect server only sees task states, never per-stage latencies, retrieval
  scores, or token counts — the metrics the Query dashboard exists for. And
  in-app collectors have no exporter-version coupling (the exporter pins
  `prefect==3.7.6` internally; upgrades are its problem, on an optional
  profile).
- **Fallback over defer** (the plan's working style): the exporter was
  verified Prefect-3-compatible and pinned, but demoted to optional because
  the in-app path already covers the DoD.
- **Provisioned-only Grafana** keeps dashboards reviewable in git and
  reproducible on `down -v` — click-ops dashboards are state, not code
  (`allowUiUpdates: false`, mounts `:ro`).

## Consequences

- **Metrics are per-process** — the honest cost of in-app collectors:
  Prometheus scrapes only the API process, so a **CLI ingest records into
  its own short-lived process and never reaches Grafana**. The Ingestion
  dashboard populates from API-driven ingests; `POST /api/ingest`
  closed that gap. (Query metrics were never affected — the GUI path runs
  in the API process.)
- **`varagity_rerank_delta` exposes no `_sum` sample**: the histogram has
  signed/negative buckets (−32…+32), and `prometheus_client` omits `_sum`
  for negative buckets by Prometheus semantics. Chart it with bucket
  quantiles (the Query dashboard uses p95/median/p05), never sum/count
  averages — recorded here so nobody "fixes" it into an average.
- `METRICS_ENABLED` gates only the route (off → structured 404); collectors
  always record. `dependency_up` refreshes inside the health check the api
  container already probes every 15 s — no poller.
- Live-validated end-to-end at implementation: three hybrid queries through
  `POST /api/chat` yielded p95 by stage embed 0.098 s / retrieve 0.098 s /
  generate 19.5 s, `varagity_query_total{method="hybrid",outcome="ok"} = 3`,
  and mean retrieval score decaying 1.0 → 0.08 across ranks 1–10 — the
  drift signal the score panel exists for.
- Rejected: multiprocess registry (`PROMETHEUS_MULTIPROC_DIR` complexity
  for a single-user stack; scale by replicas instead), alerting and
  long-term storage tuning (out of scope until something pages a human).

## Amendment (2026-07-19): the observability repair

v3's observability repair (spec_v3 §6) corrected one consequence above
and closed the exporter question this record left open.

- **The per-process consequence understated the counter problem.** It is
  not just that CLI ingests never reach Grafana — a *labelled* counter's
  child series is **born at its full value** after a process start
  (`…{file_type="md"}` doesn't exist until its first `.inc()`, so
  Prometheus never observes a rise inside the series), which makes
  `increase()`/`rate()` over the ingest counters read **0 over any
  window**. That is why every Ingestion panel read zero against a
  15-document corpus while the metrics themselves were correct. Corpus
  size is now answered by **store-derived gauges** read from pgvector at
  scrape time ([ADR-013](ADR-013-corpus-gauges-vs-counters.md)); the
  counters stay for per-event questions; and a dashboard lint
  (`tests/unit/test_dashboards.py`) fails the unit suite on any
  `increase()`/`rate()` over a `varagity_ingest_*` counter.
- **The prefect-exporter's zeros were never upstream issue #120.** Read
  from the running `3.6.1` image's own source: the exporter windows its
  flow-*run* queries to runs started within the last `OFFSET_MINUTES` —
  **default 3** — so this stack's rare, bursty flows correctly read
  `prefect_flow_runs_total = 0` between runs (the counter bug class
  again, one layer up; the unwindowed `prefect_flows_total` was always
  right). The compose service now sets `OFFSET_MINUTES=1440` — "did
  anything run today?" is the question the panel exists for — verified
  live: `prefect_info_flow_runs` 0 → 20 within one scrape.
  [prometheus-prefect-exporter#120](https://github.com/PrefectHQ/prometheus-prefect-exporter/issues/120)
  (open since 2026-03-06, no root cause, no maintainer response) shares
  the symptom and is **not our bug**; `3.6.1` stays pinned.
- Two operational facts recorded with the fix: **`PREFECT_API_URL` must
  keep its `/api` suffix** — the exporter's healthz probe concatenates
  `/health` with no fixup and `SystemExit`s on failure, so a wrong URL
  is a crash-loop, not an empty panel. And **no published compatibility
  matrix exists** between exporter and Prefect-server versions — `3.6.1`
  working against a Prefect 3.x server is an empirically verified
  pairing, not a supported one.
- `prefect_deployments_total` / `prefect_work_pools_total` read **0
  forever by design** on this stack (flows run in-process; there are no
  deployments or workers to count) — their panel says so in its
  description rather than being deleted.
