/**
 * The curated codebase map — the checked-in, hand-maintained picture of how
 * Varagity fits together (ADR-015; update rule in golden-docs/architecture.md).
 *
 * This is the condensed 26-node graph adopted one-shot from a 2026-07-20
 * foglamp scan of the repo; the scan artifact is not retained, so this
 * literal is the source of truth: one node per moving part, model usage
 * expressed as edges into the three `model` nodes (the renderer folds those
 * into on-card chips), and three groups (Ingestion · Query path ·
 * Observability) drawn as containers.
 *
 * Authored as a TS literal ending in `satisfies CodebaseMap` so the
 * `kind`/`EdgeKind` unions genuinely type-check — a JSON literal would widen
 * to `string` (microsoft/TypeScript#26552). Every `sourceRef` is
 * drift-guarded by `__tests__/codebase-map.test.ts` against the repo root.
 */

import type { CodebaseMap } from "./codebase-map";

export const CODEBASE_MAP = {
  project: {
    name: "Varagity",
    date: "2026-07-20",
    tagline: "Contextual Retrieval RAG, fully self-hosted on local GPUs",
  },
  topModels: [
    {
      id: "qwythos-9b",
      label: "Qwythos-9B-Claude-Mythos-5",
      domain: "huggingface.co",
    },
    { id: "e5-large", label: "multilingual-e5-large", domain: "huggingface.co" },
    { id: "bge-reranker", label: "bge-reranker-v2-m3", domain: "huggingface.co" },
  ],
  // The pluggable families (the four @register registries + the OCR factory,
  // spec §5.1) and the preview capability — internal, so no favicon domains.
  topTools: [
    { id: "parsers", label: "Parser registry" },
    { id: "chunkers", label: "Chunker registry" },
    { id: "retrievers", label: "Retriever registry" },
    { id: "chat-engines", label: "Chat engine registry" },
    { id: "ocr", label: "OCR engines" },
    { id: "preview", label: "Page previews" },
  ],
  topIntegrations: [
    { id: "llamacpp", label: "llama.cpp", domain: "ggml.ai" },
    { id: "infinity", label: "Infinity", domain: "github.com" },
    { id: "pgvector", label: "Postgres + pgvector", domain: "postgresql.org" },
    { id: "elasticsearch", label: "Elasticsearch", domain: "elastic.co" },
    { id: "prefect", label: "Prefect", domain: "prefect.io" },
    { id: "prometheus", label: "Prometheus", domain: "prometheus.io" },
    { id: "grafana", label: "Grafana", domain: "grafana.com" },
    { id: "docling", label: "Docling", domain: "github.com" },
    { id: "easyocr", label: "EasyOCR", domain: "jaided.ai" },
    { id: "libreoffice", label: "LibreOffice", domain: "libreoffice.org" },
  ],
  graph: {
    nodes: [
      {
        id: "web",
        label: "Next.js chat GUI",
        kind: "entry",
        sub: "web/ · :3000",
        // The map's update rule applied to itself (ADR-015): pinning /map's
        // own page here means deleting the map route fails the drift guard.
        sourceRef: "web/app/map/page.tsx",
        detail:
          "SSE chat with an evidence panel showing how each answer was built, corpus manager, live settings drawer, command palette, and a /map architecture map behind developer mode.",
      },
      {
        id: "cli",
        label: "CLI",
        kind: "entry",
        sub: "main.py · ingest/chat/eval",
        sourceRef: "varagity/cli/app.py",
        detail:
          "ingest / chat / eval entry point; runs the same Prefect flows in-process as the API.",
      },
      {
        id: "api",
        label: "FastAPI backend",
        kind: "service",
        sub: ":8000 · SSE + CRUD",
        sourceRef: "varagity/api/main.py",
        detail:
          "Async edge over sync Prefect flows (threadpool). SSE per turn: retrieval → reasoning → token → done — evidence before prose. Applies idempotent SQL migrations on startup.",
      },

      {
        id: "corpus",
        label: "Corpus manager",
        kind: "service",
        sub: "upload · ingest · delete",
        group: "Ingestion",
        sourceRef: "varagity/api/routes/documents.py",
        detail:
          "Composer uploads of files and folders (client-side 409 queueing), per-doc and folder delete, stale-corpus flag; uploads auto-ingest with reingest=false.",
      },
      {
        id: "ingest",
        label: "Ingestion flow",
        kind: "service",
        sub: "Prefect · parse→embed→index",
        group: "Ingestion",
        sourceRef: "varagity/pipeline/ingest_flow.py:235",
        detail:
          "Every stage a tracked Prefect task. doc_id hashes the DOCS_PATH-relative path + file bytes, so unchanged files skip until --reingest.",
      },
      {
        id: "parsers",
        label: "Parser registry",
        kind: "service",
        sub: "pdf · text · office · web",
        group: "Ingestion",
        sourceRef: "varagity/ingest/parsers/pdf.py:190",
        detail:
          "Docling-based PDF, office and web parsing with a two-pass OCR fallback for scanned or textless PDFs (ADR-004).",
      },
      {
        id: "chunkers",
        label: "Chunker registry",
        kind: "service",
        sub: "5 strategies",
        group: "Ingestion",
        sourceRef: "varagity/chunking/recursive_character.py:19",
        detail:
          "recursive_character (default), token_based, markdown_aware, semantic, docling_hybrid — default decided by a fact-anchored benchmark sweep; CHUNK_SIZE unit is per-strategy.",
      },
      {
        id: "contextualizer",
        label: "Contextualizer",
        kind: "agent",
        sub: "LLM blurb per chunk",
        group: "Ingestion",
        sourceRef: "varagity/context/contextual.py:75",
        detail:
          "Anthropic-style contextual retrieval: an LLM situating blurb per chunk, prepended before embedding and BM25 indexing. Overruns degrade to no-context, never a failed ingest.",
      },

      {
        id: "qflow",
        label: "Query flow",
        kind: "service",
        sub: "Prefect · retrieve→answer",
        group: "Query path",
        sourceRef: "varagity/pipeline/query_flow.py:276",
        detail:
          "Retrieve → (rerank) → answer, streaming. Every RetrievedChunk carries a RetrievalTrace — per-arm ranks, fused score, rerank delta — rendered identically by CLI, GUI and history.",
      },
      {
        id: "condenser",
        label: "Condense engine",
        kind: "agent",
        sub: "chat: condense_context",
        group: "Query path",
        sourceRef: "varagity/chat/condense.py:33",
        detail:
          "Rewrites follow-ups into standalone search queries from recent history; output scrubbed of <think> blocks with raw-query fallback. Benchmarks kept simple the default (ADR-011).",
      },
      {
        id: "retrievers",
        label: "Retriever registry",
        kind: "service",
        sub: "semantic·bm25·hybrid·reranked",
        group: "Query path",
        sourceRef: "varagity/retrieval/hybrid.py:125",
        detail:
          "hybrid fuses both arms (0.8 dense / 0.2 BM25) and hydrates rows from pgvector; reranked composes a base retriever (over-fetch, cross-encode, keep 5) rather than forking fusion.",
      },
      {
        id: "answerer",
        label: "Answer generator",
        kind: "agent",
        sub: "grounded + cited",
        group: "Query path",
        sourceRef: "varagity/generation/answer.py:162",
        detail:
          "Streams an answer grounded strictly in retrieved context with [SOURCE] citations; the web app rewrites those to chips before markdown parsing.",
      },

      {
        id: "preview",
        label: "Page previews",
        kind: "service",
        sub: "PDFium · LibreOffice",
        sourceRef: "varagity/preview/render.py",
        detail:
          "Locates a chunk's page and renders it for the evidence panel; PPTX converts via LibreOffice first. Degrades per-document (available:false + reason), never 500.",
      },
      {
        id: "eval",
        label: "Eval harness",
        kind: "service",
        sub: "retrieval · ocr · chat",
        sourceRef: "varagity/eval/evaluate.py",
        detail:
          "Retrieval matrix, chunker sweep, OCR benchmark and multi-turn chat eval against golden datasets — defaults here are benchmark-decided (ADR-004, ADR-011).",
      },

      {
        id: "llm",
        label: "Qwythos-9B (llama.cpp)",
        kind: "model",
        domain: "huggingface.co",
        sub: "GPU 0 · :8080 · 16k ctx",
        detail:
          "Qwythos-9B-Claude-Mythos-5 Q8_0 gguf on llama.cpp server-cuda: single slot for a full 16k window per request, MoE expert tensors offloaded to CPU.",
      },
      {
        id: "e5",
        label: "multilingual-e5-large",
        kind: "model",
        domain: "huggingface.co",
        sub: "Infinity · GPU 1 · 1024-dim",
        detail:
          "1024-dim instruct embeddings, asymmetric by contract: passages embed bare, queries get the Instruct/Query prefix — wrong formatting silently degrades recall.",
      },
      {
        id: "reranker",
        label: "bge-reranker-v2-m3",
        kind: "model",
        domain: "huggingface.co",
        sub: "Infinity /v1/rerank",
        detail:
          "Cross-encoder on the same Infinity container via the optimum/ONNX engine (no sm_120 torch kernels for the 5060) with a 32;4 batch cap.",
      },

      {
        id: "pg",
        label: "Postgres + pgvector",
        kind: "store",
        domain: "postgresql.org",
        sub: ":5432 · chunks + convos",
        detail:
          "Canonical chunk metadata + dense vectors, plus conversations whose message_sources snapshot evidence and trace — history survives reingests (soft chunk refs).",
      },
      {
        id: "es",
        label: "Elasticsearch",
        kind: "store",
        domain: "elastic.co",
        sub: ":9200 · contextual BM25",
        detail:
          "Contextual BM25 index over situated chunks; single-node, so yellow cluster health is healthy.",
      },
      {
        id: "docsdir",
        label: "docs/ corpus",
        kind: "store",
        sub: "gitignored ingest input",
        detail:
          "The gitignored RAG input corpus on disk — a bind mount shared by CLI and API so both agree on the corpus.",
      },

      {
        id: "docling",
        label: "Docling",
        kind: "external",
        domain: "github.com",
        sub: "layout-aware conversion",
        detail:
          "IBM's layout-aware document converter (PDF, office, HTML); also backs the docling_hybrid chunker. Layout/table models download on first use.",
      },
      {
        id: "ocr",
        label: "OCR engines",
        kind: "external",
        sub: "EasyOCR · Tesseract",
        sourceRef: "varagity/ingest/parsers/pdf.py:104",
        detail:
          "Factory in the PDF parser; EasyOCR is the benchmark-decided default (error-free where Tesseract drops words), Tesseract trades accuracy for ~5x speed.",
      },
      {
        id: "prefect",
        label: "Prefect server",
        kind: "external",
        domain: "prefect.io",
        sub: ":4200 · run tracking",
        detail:
          "Tracks every flow/task run; flows execute in-process from CLI and API (no workers), every task NO_CACHE.",
      },

      {
        id: "prom",
        label: "Prometheus",
        kind: "external",
        domain: "prometheus.io",
        sub: ":9090 · 15s scrape",
        group: "Observability",
        detail:
          "Scrapes the API's /metrics every 15s; optional exporter targets ride compose profiles.",
      },
      {
        id: "graf",
        label: "Grafana",
        kind: "external",
        domain: "grafana.com",
        sub: ":3001 · 3 dashboards",
        group: "Observability",
        detail:
          "Provisioned Query / Ingestion / Infra dashboards, anonymous viewer. Corpus gauges read pgvector at scrape time, so CLI ingests count too (ADR-013).",
      },
      {
        id: "pexp",
        label: "prefect-exporter",
        kind: "external",
        domain: "prefect.io",
        sub: "--profile · 24h window",
        group: "Observability",
        detail:
          "Optional profile. OFFSET_MINUTES=1440 — the image's 3-minute default reads 0 on bursty flows; PREFECT_API_URL must keep its /api suffix or it crash-loops.",
      },
    ],
    edges: [
      { from: "web", to: "api", kind: "triggers", label: "chat (SSE)" },
      { from: "cli", to: "qflow", kind: "triggers", label: "terminal Q&A" },
      { from: "cli", to: "ingest", kind: "triggers", label: "ingest" },
      { from: "cli", to: "eval", kind: "triggers", label: "eval suite" },
      { from: "api", to: "condenser", kind: "calls", label: "condense_context" },
      { from: "api", to: "qflow", kind: "triggers", label: "simple (default)" },
      { from: "condenser", to: "llm", kind: "calls", label: "rewrite follow-up" },
      { from: "condenser", to: "qflow", kind: "triggers", label: "standalone query" },
      { from: "qflow", to: "retrievers", kind: "calls" },
      { from: "retrievers", to: "e5", kind: "calls", label: "embed query" },
      { from: "retrievers", to: "pg", kind: "reads", label: "dense kNN" },
      { from: "retrievers", to: "es", kind: "reads", label: "BM25 search" },
      { from: "retrievers", to: "reranker", kind: "calls", label: "40 → top 5" },
      { from: "qflow", to: "answerer", kind: "calls" },
      { from: "answerer", to: "llm", kind: "calls", label: "grounded prompt" },
      { from: "api", to: "pg", kind: "writes", label: "turns + trace snapshots" },
      { from: "qflow", to: "prefect", kind: "writes", label: "task runs" },

      { from: "web", to: "corpus", kind: "triggers", label: "📎 files & folders" },
      { from: "corpus", to: "docsdir", kind: "writes", label: "saves uploads" },
      { from: "corpus", to: "ingest", kind: "triggers", label: "auto-ingest" },
      { from: "ingest", to: "docsdir", kind: "reads", label: "changed files only" },
      { from: "ingest", to: "parsers", kind: "calls" },
      { from: "parsers", to: "docling", kind: "calls", label: "convert + layout" },
      { from: "parsers", to: "ocr", kind: "calls", label: "scanned pages" },
      { from: "ingest", to: "chunkers", kind: "calls" },
      { from: "ingest", to: "contextualizer", kind: "calls", label: "per chunk" },
      { from: "contextualizer", to: "llm", kind: "calls", label: "situating blurb" },
      { from: "ingest", to: "e5", kind: "calls", label: "embed passages" },
      { from: "ingest", to: "pg", kind: "writes", label: "chunks + vectors" },
      { from: "ingest", to: "es", kind: "writes", label: "contextual BM25 docs" },
      { from: "ingest", to: "prefect", kind: "writes", label: "task runs" },

      { from: "web", to: "preview", kind: "triggers", label: "evidence page peek" },
      { from: "preview", to: "docsdir", kind: "reads", label: "renders source page" },
      { from: "eval", to: "qflow", kind: "calls", label: "5-config matrix" },

      { from: "prom", to: "api", kind: "reads", label: "/metrics 15s" },
      { from: "graf", to: "prom", kind: "reads" },
      { from: "prom", to: "pexp", kind: "reads", label: "--profile" },
      { from: "pexp", to: "prefect", kind: "reads", label: "flow runs (24h)" },
    ],
  },
} satisfies CodebaseMap;
