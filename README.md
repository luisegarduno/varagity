# Varagity

Full-stack RAG application with **Contextual Retrieval** — self-hosted and
GPU-accelerated, with a web GUI and a terminal client over the same pipeline.
Documents in `docs/` (PDF, Office, HTML, txt/md — with automatic OCR fallback
for scans) are parsed, chunked, situated with LLM-generated context blurbs,
and indexed into **both** pgvector (semantic) and Elasticsearch (contextual
BM25); questions are answered with grounded, cited answers via hybrid
rank-fusion retrieval, optionally re-ranked by a cross-encoder (the ≈67% tier
of the [Anthropic ladder](https://www.anthropic.com/news/contextual-retrieval)).
Every answer in the web app ships with a **"How this answer was built"**
evidence panel: per-chunk semantic/BM25/fusion/re-rank provenance, the
situating blurb, and inline `[SOURCE]` citation chips that link into it.
Every pipeline stage is a tracked Prefect task.

> 📄 **Design**: [`spec.md`](spec.md) (v1) · `spec_v2.md` (v2) · **As-built docs**:
> [`golden-docs/`](golden-docs/index.md) (`uv run mkdocs serve`) — architecture,
> data model, pipelines, runbook, ADRs, the HTTP API contract, Python API reference.


## Status

**v1 — complete.** Terminal Contextual-Retrieval Q&A: ingestion with OCR
fallback, contextual embeddings + contextual BM25, hybrid rank fusion,
recall@k/pass@k eval matrix + OCR benchmark, Prefect tracking, golden-docs.

**v2 — complete** (all ten phases shipped):

- [x] **Re-ranking** wired into the query path (`RETRIEVAL_METHOD=reranked`,
  `bge-reranker-v2-m3`) + a per-chunk `RetrievalTrace` (semantic/BM25/fusion
  ranks, re-rank delta); eval matrix extended to 5 configs
- [x] **HTTP API** (FastAPI): SSE streaming chat, conversation persistence,
  idempotent schema migrations
- [x] **Web GUI** (Next.js + TypeScript + Tailwind + shadcn/ui): streamed
  markdown chat, conversation sidebar, reasoning trace, light/dark themes
- [x] **Evidence panel + citations**: the "How this answer was built" rail;
  inline `[SOURCE]` chips scroll to their chunk card; citations not backed by
  retrieved evidence are flagged
- [x] **More modalities**: `.docx` / `.pptx` / `.xlsx` / `.html` via a
  generalized Docling parser (slides/sheets carry page-level provenance)
- [x] **More chunking strategies**: `token_based`, `markdown_aware`,
  `semantic`, `docling_hybrid` — benchmark sweep in the eval harness
  (default stays `recursive_character`)
- [x] **Observability**: Prometheus + provisioned Grafana dashboards
  (Query / Ingestion / Infra) at :3001, no login needed; optional
  Prefect-exporter and DCGM GPU-metrics compose profiles
- [x] **Corpus management + live settings UI**: drag-and-drop upload with
  live ingest progress, per-document listing/delete, runtime setting
  overrides with stale-corpus flagging
- [x] **Design-system polish**: design tokens (accent/density), responsive
  and mobile layouts, a11y pass, ⌘K command palette, opt-in Playwright e2e
- [x] **Hardening**: ADR-005…009, a top-level HTTP API docs page
  (drift-guarded OpenAPI snapshot), a web CI job + coverage floors,
  as-built docs refresh


-----------------------------

# Instructions

## Pre-Req's

*  \>=12GB's of disk space
* Two Nvidia GPUs (llama.cpp on GPU 0, infinity embeddings + reranker on
  GPU 1 — see the [runbook](golden-docs/runbook.md) to adapt the pinning)
* Model files on the host (bind-mounted, never copied into images):
  the llama.cpp `.gguf`, the `multilingual-e5-large-instruct` snapshot, and the
  `bge-reranker-v2-m3` snapshot with a pre-exported ONNX — paths & details in the
  [runbook](golden-docs/runbook.md#prerequisites)
* `Nvidia-Docker` must be setup
    <details><summary>Instructions (debian)</summary>

    1. Add the package repositories (modern method without apt-key)
        ```bash
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        ```

    2. Install nvidia-container-toolkit
        ```bash
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        ```

    3. Configure Docker to use the runtime
        ```bash
        sudo nvidia-ctk runtime configure --runtime=docker
        ```

    4. Lastly, restart docker daemon
        ```bash
        sudo systemctl restart docker
        ```
    </details>

## Running Varagity

1. Configure the environment (see the comments in the file — the checked-in
   values are the in-container ones):
    ```bash
    cp .env.example .env   # then set volumes, BASE_MODEL, keys/passwords
    ```

2. Put documents (`.pdf`, `.txt`, `.md`, `.docx`, `.pptx`, `.xlsx`, `.html`)
   into `./docs/`, then bring up the stack — ten services with healthchecks;
   `--wait` returns once all are green:
    ```bash
    docker compose up -d --wait
    bash scripts/smoke.sh        # optional: sequenced checks across all ten services
    ```

3. Open **[http://localhost:3000](http://localhost:3000)** and ask a question.
   The answer streams token-by-token, and the evidence panel shows every
   retrieved chunk — its score, *why* it ranked there (semantic #, BM25 #,
   fused score, re-rank Δ), the situating context blurb, and the expandable
   full text; inline `[SOURCE]` chips scroll to the matching card.
   Conversations persist across restarts.

4. Watch it work: Grafana at **[http://localhost:3001](http://localhost:3001)**
   renders the provisioned Query / Ingestion / Infra dashboards, no login
   needed (Prometheus itself: [http://localhost:9090](http://localhost:9090)).
   Metrics populate from API/GUI activity — CLI runs record into their own
   process, which is never scraped.

5. Prefer the terminal? The `app` container ingests on start; for an
   interactive session:
    ```bash
    docker compose run --rm app uv run main.py chat
    ```
    You get the matches table (with the same trace at `-v2`) and a grounded
    answer citing its `[SOURCE]`. Type `:quit` to exit.

To activate re-ranking, set `RETRIEVAL_METHOD=reranked` and
`RERANK_ENABLED=true` in `.env` — hybrid over-fetches `RERANK_CANDIDATES`,
the cross-encoder keeps the top `RERANK_TOP_N`.

### CLI

```bash
uv run main.py ingest              # ingest DOCS_PATH into both stores
uv run main.py ingest --reingest   # re-process after pipeline-setting changes
uv run main.py chat                # ingest, then Q&A loop (the default command)
uv run --group eval main.py eval       # retrieval-quality matrix (5 configs × k) + chunker sweep
uv run --group eval main.py eval ocr   # OCR engine benchmark (CER/WER, pages/s)
```

Run on the host against the containerized services with localhost overrides —
see [host-vs-container `.env` usage](golden-docs/runbook.md#host-vs-container-env-usage).

### HTTP API

The web app is a pure client of the FastAPI service at
[http://localhost:8000](http://localhost:8000) (interactive OpenAPI docs at
`/docs`). `POST /api/chat` streams Server-Sent Events in a fixed order —
evidence before prose: `retrieval` (the provenance payload) → `reasoning`
(`<think>` tokens, when the model emits them) → `token` (answer deltas) →
`done` (ids, usage, per-stage latency). Conversations are exposed as REST
(`GET/POST/DELETE /api/conversations…`); `GET /api/health` reports per-service
reachability and `GET /api/config` lists the registered chunkers/retrievers/
OCR engines the UI builds its controls from.

The full as-built contract — corpus/settings routes included, plus both SSE
protocols and the error envelope — lives in
[`golden-docs/api.md`](golden-docs/api.md), rendered from an OpenAPI snapshot
(`golden-docs/openapi.json`) that a unit test guards against drift; regenerate
it with `uv run python scripts/export_openapi.py`.

Prefect UI (flow/task runs, logs, timings): [http://localhost:4200](http://localhost:4200).
Grafana (provisioned dashboards, anonymous viewing): [http://localhost:3001](http://localhost:3001);
Prometheus: [http://localhost:9090](http://localhost:9090).
Embeddings API docs (infinity): `http://<bound-interface>:8081/v1/docs` — the
binding is interface-specific; resolve it with
`docker compose port infinity-embeddings 8081`.

## Development

Python (this project uses [`uv`](https://github.com/astral-sh/uv)):

```bash
uv sync                            # install deps (incl. dev/eval groups)
uv run pytest                      # unit suite incl. async API tests (coverage floor: 80%)
uv run pytest -m integration       # real Postgres/ES via testcontainers (Docker)
uv run pytest -m e2e               # full ingest→query over the fixtures corpus (Docker)
uv run ruff check . && uv run ruff format --check .
uv run mypy varagity
uv run pre-commit run --all-files  # lint + format + types + unit tests
uv run mkdocs serve                # the golden-docs site, live-reloading
uv run uvicorn varagity.api.main:create_app --factory --port 8000   # API on the host
```

Frontend (`web/`, uses [`pnpm`](https://pnpm.io)):

```bash
pnpm dev                           # dev server on :3000 (against NEXT_PUBLIC_API_URL)
pnpm test                          # Vitest unit tests (coverage-gated in CI)
pnpm e2e                           # opt-in Playwright e2e — bring the stack up first
pnpm lint && pnpm build
pnpm gen:types                     # regenerate lib/types.ts from the API's OpenAPI schema
```

`NEXT_PUBLIC_API_URL` is baked in at **build** time — after changing it, rerun
`docker compose build web` (or restart `pnpm dev`).

CI (GitHub Actions) runs two jobs on every push — **Python**: lint + format
check, mypy, the unit suite under the 80% coverage floor (API included), and
a strict docs build; **web**: pnpm lint, the Vitest suite under a coverage
floor, and a production build. The integration/e2e suites and the Playwright
browser tests stay local — they need Docker and the live stack.
