# API Reference

Rendered from the package's Google-style docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) — docstring presence and
format are machine-enforced (ruff pydocstyle, `google` convention), so these
pages are generated from the same text a reader sees in the source.

One page per package:

- [Configuration & logging](config.md) — `config`, `logging_setup`, `tokens`
- [Model clients](models.md) — the llama.cpp / infinity clients (chat,
  embeddings, rerank), the streaming `<think>` splitter, and the factory
- [Ingestion](ingest.md) — discovery, parsers (text, PDF/OCR, office, web,
  image — over a shared Docling core), the loader
- [Chunking](chunking.md) — strategy registry + the five strategies
  (`recursive_character`, `token_based`, `markdown_aware`, `docling_hybrid`,
  `semantic`)
- [Contextual Retrieval](context.md) — `situate_context()`
- [Stores](stores.md) — `ChunkRecord`, pgvector, Elasticsearch BM25,
  conversation persistence, app settings, the migration runner
- [Retrieval](retrieval.md) — semantic / bm25 / hybrid + fusion, and the
  `reranked` composition over them
- [Chat engines](chat.md) — the `PreparedQuery` two-string split, the
  registry, and the `simple` / `condense_context` engines
- [Generation](generation.md) — context prompt & grounded answers
- [Orchestration](pipeline.md) — the Prefect flows
- [HTTP API](api.md) — the FastAPI service: SSE chat, conversations,
  settings, documents/corpus, ingest, health
- [Page preview](preview.md) — evidence-panel page previews: locate
  (trigram scoring + pdfium rects), render, source resolution, PPTX
  conversion
- [Observability](observability.md) — the Prometheus metric catalog and its
  recording helpers
- [Evaluation](eval.md) — golden set, metrics, OCR benchmark, testcontainers
- [CLI & debug output](cli.md) — subcommands and the `v_<name>` renderers
