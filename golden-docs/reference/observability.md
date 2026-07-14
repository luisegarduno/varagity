# Observability

The Prometheus metric catalog (spec_v2 §6.2) and its recording helpers.
The probe points live in the pipeline flows, the reranked retriever, and
the API health probes; the catalog is exposed at `GET /metrics` and
charted by the provisioned Grafana dashboards under `observability/`.

::: varagity.observability

::: varagity.observability.metrics
