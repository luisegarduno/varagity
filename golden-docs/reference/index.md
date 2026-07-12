# API Reference

Rendered from the package's Google-style docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) — docstring presence and
format are machine-enforced (ruff pydocstyle, `google` convention), so these
pages are generated from the same text a reader sees in the source.

One page per package:

- [Configuration & logging](config.md) — `config`, `logging_setup`, `tokens`
- [Model clients](models.md) — the llama.cpp / infinity clients and factory
- [Ingestion](ingest.md) — discovery, parsers (text, PDF/OCR), the loader
- [Chunking](chunking.md) — strategy registry + `recursive_character`
- [Contextual Retrieval](context.md) — `situate_context()`
- [Stores](stores.md) — `ChunkRecord`, pgvector, Elasticsearch BM25
- [Retrieval](retrieval.md) — semantic / bm25 / hybrid + fusion
- [Generation](generation.md) — context prompt & grounded answers
- [Orchestration](pipeline.md) — the Prefect flows
- [HTTP API](api.md) — the FastAPI service: SSE chat, conversations, health
- [Evaluation](eval.md) — golden set, metrics, OCR benchmark, testcontainers
- [CLI & debug output](cli.md) — subcommands and the `v_<name>` renderers
