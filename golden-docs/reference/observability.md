# Observability

The Prometheus metric catalog (spec_v2 §6.2) and its recording helpers.
The probe points live in the pipeline flows, the reranked retriever, and
the API health probes; the catalog is exposed at `GET /metrics` and
charted by the provisioned Grafana dashboards under `observability/`.

Two kinds of instrument, and the difference decides which panel can answer
which question:

| | Source | Survives an API restart? | Sees CLI ingests? |
|---|---|---|---|
| `varagity_corpus_*` gauges | pgvector, at scrape time | yes | yes |
| counters / histograms | this process | no | no ([ADR-007](../adr/ADR-007-observability-stack.md)) |

Ingestion is rare and bursty, so `increase()`/`rate()` over an ingest
counter reads `0` over *any* window — the counter's series is born at its
full value after a restart, so Prometheus never observes the rise. Corpus
size is therefore a gauge question, not a counter question (spec_v3 §6.1;
[ADR-013](../adr/ADR-013-corpus-gauges-vs-counters.md) is the full record);
`tests/unit/test_dashboards.py` fails the build if a panel regresses to the
counter form.

::: varagity.observability

::: varagity.observability.metrics

::: varagity.observability.corpus
