"""In-app observability: the Prometheus metric catalog (spec_v2 §6).

:mod:`varagity.observability.metrics` holds the spec_v2 §6.2 catalog plus
the recording helpers the pipeline probe points call — instruments that
record *what this process did*.

:mod:`varagity.observability.corpus` adds the complement: gauges read from
pgvector at scrape time, answering *what is in the store*. Process history
resets when the API restarts and misses CLI ingests entirely, which is what
left the Ingestion dashboard reading zero through v2 (spec_v3 §6.1).

:mod:`varagity.observability.graph` applies that same shape to the graph
corpus, reading the engine's working directory (never the engine) at scrape
time — a backfill that ran for hours yesterday is a state question too.

The API's ``GET /metrics`` route exposes the same process-wide registry.
"""
