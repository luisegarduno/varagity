/**
 * The curated codebase map of Varagity — entries, the Prefect flows, the chat
 * engines, the three models, the pluggable registries, and the datastores
 * (spec_codebase_map.md §7, with every correction from the implementation
 * plan's Phase 1 tables applied).
 *
 * This is a hand-maintained artifact, not generated. Edit it when the
 * architecture changes; the `sourceRef` drift guard in
 * `__tests__/codebase-map.test.ts` fails CI on a rename, but edge *semantics*
 * are the human's job (ADR-015). Totals: 41 nodes, 59 edges, 3 models,
 * 8 tools, 7 integrations, 3 groups (Web app 4 · Chat engines 3 · Ingestion 5).
 *
 * Authored as `… satisfies CodebaseMap` so the `kind` / `EdgeKind` string
 * unions genuinely type-check (a JSON literal would widen them away).
 */

import type { CodebaseMap } from "./codebase-map";

export const CODEBASE_MAP = {
  project: {
    name: "Varagity",
    date: "2026-07-19",
    summary:
      "Self-hosted contextual-retrieval RAG: contextual embeddings + contextual BM25 + rank fusion + cross-encoder reranking over local GPUs, with a CLI and a Next.js chat GUI sharing one set of Prefect flows.",
  },
  topModels: ["llm-chat", "model-embed", "model-rerank"],
  topTools: [
    "retriever-hybrid",
    "retriever-reranked",
    "retriever-semantic",
    "retriever-bm25",
    "parser-registry",
    "chunker-registry",
    "ocr-engines",
    "preview-renderer",
  ],
  topIntegrations: [
    "svc-llamacpp",
    "svc-infinity",
    "store-postgres",
    "store-elasticsearch",
    "svc-prefect",
    "svc-prometheus",
    "svc-grafana",
  ],
  graph: {
    nodes: [
      // --- Web app (browser-facing entries) ---
      {
        id: "web-chat",
        label: "Chat GUI",
        kind: "entry",
        sub: "Next.js app router, SSE stream",
        group: "Web app",
        detail:
          "The only browser-facing surface; talks only to the API. Renders answers with inline citation chips and an evidence rail.",
        sourceRef: "web/app/c/[id]/page.tsx",
        domain: "nextjs.org",
      },
      {
        id: "web-corpus",
        label: "Corpus page",
        kind: "entry",
        sub: "upload, folder tree, reingest",
        group: "Web app",
        sourceRef: "web/app/corpus/page.tsx",
      },
      {
        id: "web-settings",
        label: "Settings drawer",
        kind: "entry",
        sub: "live settings + display prefs",
        group: "Web app",
        detail:
          "Generates controls from GET /api/settings; edits stage locally and apply in one PATCH so linked constraints validate together.",
        sourceRef: "web/components/settings/SettingsDrawer.tsx",
      },
      {
        // No sourceRef this phase — web/app/map/page.tsx does not exist yet;
        // it is added in the Phase 3 commit that creates the page (guard and
        // artifact land together, per the openapi.json precedent).
        id: "web-map",
        label: "Codebase Map",
        kind: "entry",
        sub: "this map; developer mode only",
        group: "Web app",
      },

      // --- CLI (v1 terminal front-end) ---
      {
        id: "cli",
        label: "CLI",
        kind: "entry",
        sub: "ingest · chat · eval",
        detail:
          "The v1 terminal front-end. A peer of the API over the same Prefect flows — never a client of it.",
        sourceRef: "varagity/cli/app.py",
      },

      // --- API entries (FastAPI routes) ---
      {
        id: "api-chat",
        label: "POST /api/chat",
        kind: "entry",
        sub: "SSE: retrieval → reasoning → token",
        detail:
          "Evidence is streamed before prose. The retrieval event carries the condensed query and the per-chunk RetrievalTrace. Loads the last N turns as condense history.",
        sourceRef: "varagity/api/routes/chat.py:290",
      },
      {
        id: "api-conversations",
        label: "Conversations API",
        kind: "entry",
        sub: "list · create · get · delete",
        sourceRef: "varagity/api/routes/conversations.py",
      },
      {
        id: "api-documents",
        label: "Documents API",
        kind: "entry",
        sub: "upload · list · delete · preview",
        sourceRef: "varagity/api/routes/documents.py",
      },
      {
        id: "api-ingest",
        label: "POST /api/ingest",
        kind: "entry",
        sub: "202; progress SSE on /status",
        detail:
          "Only a completed API-driven reingest clears the stale-corpus flag — CLI reingest and patching the setting back do not.",
        sourceRef: "varagity/api/routes/ingest.py:30",
      },
      {
        id: "api-settings",
        label: "Settings API",
        kind: "entry",
        sub: "GET/PATCH runtime overrides",
        sourceRef: "varagity/api/routes/settings.py",
      },
      {
        id: "api-metrics",
        label: "GET /metrics",
        kind: "entry",
        sub: "Prometheus exposition",
        detail:
          "Metrics are per-process: a CLI ingest records into the CLI's own never-scraped registry and never reaches Grafana.",
        sourceRef: "varagity/api/routes/metrics.py:22",
      },

      // --- Chat engines ---
      {
        id: "engine-simple",
        label: "Simple chat engine",
        kind: "agent",
        sub: "identity pass-through, no model",
        group: "Chat engines",
        detail:
          "The default. search_query == original_query. Kept over condense because reranked follow-up recall@1 (0.727) did not justify condense's 8.6 s mean latency.",
        sourceRef: "varagity/chat/simple.py:9",
      },
      {
        id: "engine-condense",
        label: "Condense engine",
        kind: "agent",
        sub: "rewrites follow-ups standalone",
        group: "Chat engines",
        detail:
          "Rewrites a follow-up into a self-contained search query using recent history; degrades to identity on kill switch, empty history, LLM failure, or an empty or over-length result.",
        sourceRef: "varagity/chat/condense.py:33",
      },
      {
        id: "engine-registry",
        label: "Chat engine registry",
        kind: "service",
        sub: "@register + get_chat_engine",
        group: "Chat engines",
        detail:
          "Selected by CHAT_ENGINE; adding an engine is one file plus its import line, zero caller edits.",
        sourceRef: "varagity/chat/base.py",
      },

      // --- Model-driving agents (ingestion + generation) ---
      {
        id: "contextualizer",
        label: "Chunk contextualizer",
        kind: "agent",
        sub: "Anthropic-style situating blurbs",
        group: "Ingestion",
        detail:
          "Writes a whole-document-aware blurb per chunk, stored prepended to the content before embedding and before BM25 indexing — the 'contextual' in contextual retrieval.",
        sourceRef: "varagity/context/contextual.py:75",
      },
      {
        id: "answer-gen",
        label: "Answer generator",
        kind: "agent",
        sub: "grounded, cited, streaming",
        detail:
          "Streams the answer over retrieved context; strips <think> blocks and emits [SOURCE] citations (path or bracket form) the web app rewrites into chips before markdown parsing.",
        sourceRef: "varagity/generation/answer.py",
      },

      // --- Models (three models, one server each) ---
      {
        id: "llm-chat",
        label: "Qwythos-9B (GGUF)",
        kind: "model",
        sub: "one server, three jobs",
        detail:
          "The single chat LLM: answers, contextualization blurbs, and condense all resolve to the same llama.cpp client via models/registry.py.",
        sourceRef: "varagity/models/llm.py:200",
        domain: "ggml.ai",
      },
      {
        id: "model-embed",
        label: "multilingual-e5-large",
        kind: "model",
        sub: "1024-dim, asymmetric formatting",
        detail:
          "Passages get NO prefix; queries get 'Instruct: {task}\\nQuery: {q}'. Getting this backwards degrades recall silently.",
        sourceRef: "varagity/models/embeddings.py:59",
        domain: "huggingface.co",
      },
      {
        id: "model-rerank",
        label: "bge-reranker-v2-m3",
        kind: "model",
        sub: "cross-encoder, ONNX on GPU 1",
        detail:
          "Cohere-protocol /rerank on the same infinity container as the embedder. Torch has no sm_120 kernels, so it runs the optimum engine with a '32;4' batch cap.",
        sourceRef: "varagity/models/rerank.py:75",
        domain: "huggingface.co",
      },

      // --- Model servers (external) ---
      {
        id: "svc-llamacpp",
        label: "llama.cpp server",
        kind: "external",
        sub: ":8080/v1 · GPU 0",
        detail:
          "OpenAI-compatible. /health returns 503 for ~30 s while loading; it hard-500s at the context window rather than stopping gracefully.",
        domain: "ggml.ai",
      },
      {
        id: "svc-infinity",
        label: "infinity server",
        kind: "external",
        sub: ":8081/v1 · GPU 1 · embed + rerank",
        detail:
          "Hosts both models. Its optimum engine ignores INFINITY_DEVICE_ID — GPU pinning happens via compose device_ids.",
        domain: "michaelfeil.eu",
      },

      // --- Prefect flows (the load-bearing internal pipeline) ---
      {
        id: "flow-ingest",
        label: "Prefect ingest flow",
        kind: "service",
        sub: "6 tracked tasks, retries=2",
        group: "Ingestion",
        detail:
          "discover → parse → chunk → contextualize → embed → store. Model and store tasks carry retries; nothing else does.",
        sourceRef: "varagity/pipeline/ingest_flow.py:234",
      },
      {
        id: "flow-query",
        label: "Prefect query flow",
        kind: "service",
        sub: "condense → embed → retrieve → answer",
        detail:
          "Two variants: query_flow and query_stream_flow. Query-path tasks deliberately carry no retries — the user is waiting.",
        sourceRef: "varagity/pipeline/query_flow.py:275",
      },

      // --- Registries (tools) ---
      {
        id: "parser-registry",
        label: "Parsers",
        kind: "tool",
        sub: "4 registered: pdf/text/office/web",
        group: "Ingestion",
        detail:
          "All but text share a Docling core. PPTX slides and XLSX sheets are Docling pages, so page identity rides the same field as PDFs.",
        sourceRef: "varagity/ingest/parsers/base.py",
      },
      {
        id: "ocr-engines",
        label: "OCR engines",
        kind: "tool",
        sub: "factory; two-pass PDF fallback",
        group: "Ingestion",
        detail:
          "A fast text pass runs first; OCR fires on a near-empty pass or a high textless-page ratio (or when forced). EasyOCR needs libGL — its absence masquerades as 'EasyOCR is not installed'.",
        sourceRef: "varagity/ingest/parsers/pdf.py:104",
      },
      {
        id: "chunker-registry",
        label: "Chunking strategies",
        kind: "tool",
        sub: "5 registered; CHUNK_SIZE unit varies",
        group: "Ingestion",
        detail:
          "recursive_character (default) · token_based · markdown_aware · semantic · docling_hybrid. CHUNK_SIZE means characters or tokens depending on the strategy.",
        sourceRef: "varagity/chunking/base.py",
      },

      // --- Retrievers (each a tool; what the eval matrix compares) ---
      {
        id: "retriever-semantic",
        label: "semantic",
        kind: "tool",
        sub: "pgvector cosine",
        sourceRef: "varagity/retrieval/semantic.py:17",
      },
      {
        id: "retriever-bm25",
        label: "bm25",
        kind: "tool",
        sub: "Elasticsearch over contextual text",
        sourceRef: "varagity/retrieval/bm25.py:61",
      },
      {
        id: "retriever-hybrid",
        label: "hybrid",
        kind: "tool",
        sub: "weighted rank fusion of both arms",
        detail:
          "Fuses ranked lists from both stores, then hydrates full rows from pgvector. Records per-arm ranks and the fused score into the RetrievalTrace.",
        sourceRef: "varagity/retrieval/hybrid.py:125",
      },
      {
        id: "retriever-reranked",
        label: "reranked",
        kind: "tool",
        sub: "composes a base retriever",
        detail:
          "Over-fetches RERANK_CANDIDATES from RERANK_BASE_METHOD, cross-encodes, keeps RERANK_TOP_N. RERANK_ENABLED=false degrades it to its base method rather than erroring.",
        sourceRef: "varagity/retrieval/reranked.py:75",
      },
      {
        id: "preview-renderer",
        label: "Page preview",
        kind: "tool",
        sub: "renders the cited page image",
        detail:
          "Degrades per-document with a reason instead of 500ing. PDFium is not thread-safe, so one module lock serializes rendering.",
        sourceRef: "varagity/preview",
      },

      // --- Stateful services ---
      {
        id: "svc-conversations",
        label: "Conversation store",
        kind: "service",
        sub: "turns + evidence snapshots",
        detail:
          "message_sources.trace snapshots content/context/source so historical conversations still explain themselves after a reingest changes chunk_ids.",
        sourceRef: "varagity/stores/conversation_store.py",
      },
      {
        id: "svc-settings",
        label: "Runtime settings",
        kind: "service",
        sub: "override layer over .env",
        detail:
          "Ingest-time changes raise the stale-corpus flag. Modules read get_settings(), never os.getenv.",
        sourceRef: "varagity/stores/app_settings_store.py",
      },
      {
        id: "svc-migrations",
        label: "Migration runner",
        kind: "service",
        sub: "ordered, idempotent, on startup",
        detail:
          "schema.sql is the fresh-install fast path; migrations reconcile existing volumes. Both must stay in sync.",
        sourceRef: "varagity/stores/migrate.py",
      },
      {
        id: "svc-eval",
        label: "Eval harness",
        kind: "service",
        sub: "retrieval matrix · chunkers · chat",
        detail:
          "Fact-anchored recall@k and pass@k over golden snippets; the source of the 'default stays simple' and 'default stays recursive_character' decisions.",
        sourceRef: "varagity/eval/evaluate.py",
      },

      // --- Stores ---
      {
        // No sourceRef: DOCS_PATH is gitignored, so a path here could never
        // resolve on a fresh CI checkout; the sibling store nodes carry none.
        id: "store-corpus",
        label: "DOCS_PATH corpus",
        kind: "store",
        sub: "gitignored ingest input",
        detail:
          "doc_id hashes the path relative to DOCS_PATH plus the file's byte hash — so config changes alone never re-trigger ingestion.",
      },
      {
        id: "store-postgres",
        label: "Postgres + pgvector",
        kind: "store",
        sub: "chunks, vectors, conversations",
        detail:
          "Canonical chunk metadata and dense vectors, plus conversation persistence. The pgdata volume keeps the first-boot password.",
        domain: "postgresql.org",
      },
      {
        id: "store-elasticsearch",
        label: "Elasticsearch",
        kind: "store",
        sub: "contextual BM25 index",
        detail:
          "Single-node, so cluster health 'yellow' is healthy. Host disk over 90% trips the disk watermark and writes hang.",
        domain: "elastic.co",
      },

      // --- Observability (external observers) ---
      {
        id: "svc-prefect",
        label: "Prefect",
        kind: "external",
        sub: ":4200 · in-process flow runs",
        detail:
          "No workers or deployments — flows run in-process from both front-ends. PREFECT_API_URL must be exported before prefect is imported.",
        domain: "prefect.io",
      },
      {
        id: "svc-prometheus",
        label: "Prometheus",
        kind: "external",
        sub: "scrapes api:8000 every 15 s",
        domain: "prometheus.io",
      },
      {
        id: "svc-grafana",
        label: "Grafana",
        kind: "external",
        sub: "Query · Ingestion · Infra boards",
        domain: "grafana.com",
      },
    ],
    edges: [
      // Web app → API
      { from: "web-chat", to: "api-chat", kind: "calls", label: "streams one turn" },
      { from: "web-chat", to: "api-conversations", kind: "reads" },
      { from: "web-corpus", to: "api-documents", kind: "calls", label: "upload / delete" },
      { from: "web-corpus", to: "api-ingest", kind: "triggers", label: "clears stale flag" },
      { from: "web-settings", to: "api-settings", kind: "writes", label: "one staged PATCH" },
      { from: "web-map", to: "web-settings", kind: "reads", label: "gated by dev mode" },

      // CLI → flows / eval
      { from: "cli", to: "flow-ingest", kind: "triggers" },
      { from: "cli", to: "flow-query", kind: "triggers" },
      { from: "cli", to: "svc-eval", kind: "triggers", label: "eval · eval chat · ocr" },

      // API → engines / flows / stores / services
      { from: "api-chat", to: "engine-registry", kind: "calls", label: "picks CHAT_ENGINE" },
      { from: "api-chat", to: "flow-query", kind: "triggers", label: "sync flow in threadpool" },
      { from: "api-chat", to: "svc-conversations", kind: "writes", label: "persists turn + evidence" },
      { from: "api-conversations", to: "svc-conversations", kind: "reads" },
      { from: "api-documents", to: "store-corpus", kind: "writes", label: "lands uploads" },
      { from: "api-documents", to: "preview-renderer", kind: "calls" },
      { from: "api-documents", to: "store-postgres", kind: "reads", label: "lists ingested docs" },
      { from: "api-ingest", to: "flow-ingest", kind: "triggers" },
      { from: "api-settings", to: "svc-settings", kind: "writes" },
      // Reversed vs §7: Prometheus initiates the scrape (prometheus.yml:13–18).
      { from: "svc-prometheus", to: "api-metrics", kind: "reads", label: "scrapes /metrics · 15 s" },

      // Chat engine resolution (the flow resolves the engine per turn)
      { from: "engine-registry", to: "engine-simple", kind: "calls" },
      { from: "engine-registry", to: "engine-condense", kind: "calls" },
      { from: "engine-condense", to: "llm-chat", kind: "calls", label: "rewrite to standalone" },

      // Query path
      { from: "flow-query", to: "engine-registry", kind: "calls", label: "prepare() per turn" },
      { from: "flow-query", to: "model-embed", kind: "calls", label: "query-mode prefix" },
      { from: "flow-query", to: "retriever-reranked", kind: "calls", label: "RETRIEVAL_METHOD" },
      { from: "flow-query", to: "answer-gen", kind: "calls", label: "grounded on top-N" },
      { from: "retriever-reranked", to: "retriever-hybrid", kind: "calls", label: "over-fetch 40" },
      { from: "retriever-reranked", to: "model-rerank", kind: "calls", label: "cross-encode, keep 5" },
      { from: "retriever-hybrid", to: "retriever-semantic", kind: "calls" },
      { from: "retriever-hybrid", to: "retriever-bm25", kind: "calls" },
      { from: "retriever-semantic", to: "store-postgres", kind: "reads", label: "cosine over 1024-dim" },
      { from: "retriever-bm25", to: "store-elasticsearch", kind: "reads" },
      { from: "retriever-bm25", to: "store-postgres", kind: "reads", label: "hydrates matches" },
      { from: "retriever-hybrid", to: "store-postgres", kind: "reads", label: "hydrates fused rows" },
      { from: "answer-gen", to: "llm-chat", kind: "calls", label: "streams cited prose" },

      // Ingest path
      { from: "flow-ingest", to: "store-corpus", kind: "reads", label: "skips unchanged hashes" },
      { from: "flow-ingest", to: "parser-registry", kind: "calls" },
      { from: "parser-registry", to: "ocr-engines", kind: "calls", label: "on weak text pass" },
      { from: "flow-ingest", to: "chunker-registry", kind: "calls" },
      { from: "flow-ingest", to: "contextualizer", kind: "calls" },
      { from: "contextualizer", to: "llm-chat", kind: "calls", label: "blurb per chunk" },
      { from: "flow-ingest", to: "model-embed", kind: "calls", label: "passage mode, no prefix" },
      { from: "flow-ingest", to: "store-postgres", kind: "writes", label: "chunks + vectors" },
      { from: "flow-ingest", to: "store-elasticsearch", kind: "writes", label: "same chunks, both stores" },

      // Models → their servers
      { from: "llm-chat", to: "svc-llamacpp", kind: "calls" },
      { from: "model-embed", to: "svc-infinity", kind: "calls" },
      { from: "model-rerank", to: "svc-infinity", kind: "calls", label: "same container" },

      // Observability + persistence writes
      { from: "flow-ingest", to: "svc-prefect", kind: "writes", label: "per-stage run logs" },
      { from: "flow-query", to: "svc-prefect", kind: "writes" },
      { from: "svc-conversations", to: "store-postgres", kind: "writes", label: "snapshots evidence" },
      { from: "svc-settings", to: "store-postgres", kind: "writes" },
      { from: "svc-migrations", to: "store-postgres", kind: "writes", label: "on API startup" },
      { from: "preview-renderer", to: "store-corpus", kind: "reads", label: "one lock; PDFium" },

      // Eval drives ingest + all four retrievers directly
      { from: "svc-eval", to: "flow-ingest", kind: "triggers", label: "subflow per ingest" },
      { from: "svc-eval", to: "retriever-hybrid", kind: "calls", label: "5-config matrix" },
      { from: "svc-eval", to: "retriever-semantic", kind: "calls" },
      { from: "svc-eval", to: "retriever-bm25", kind: "calls" },
      { from: "svc-eval", to: "retriever-reranked", kind: "calls" },

      // Grafana queries Prometheus (reversed vs §7: Grafana initiates)
      { from: "svc-grafana", to: "svc-prometheus", kind: "reads", label: "PromQL" },
    ],
  },
} satisfies CodebaseMap;
