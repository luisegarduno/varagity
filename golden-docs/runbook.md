# Runbook

Operating the Varagity stack: from clean clone to answered question, plus the
operational gotchas collected while building it.

## Prerequisites

- **Docker + Docker Compose**, with the **nvidia-container-toolkit** runtime
  configured (the README has step-by-step Debian instructions).
- **An Nvidia GPU setup** matching the compose GPU pinning (see
  [GPU & VRAM](#gpu-vram) — this repo's config assumes two GPUs).
- **≥ 12 GB free disk** for images/volumes — and note Elasticsearch's disk
  watermarks ([below](#elasticsearch-notes)): a host disk > 90% full will
  break indexing outright.
- **Model files on the host** (never copied into images):
    - `${models_volume}/<BASE_MODEL>` — the llama.cpp `.gguf` chat model.
    - `${embeddings_volume}/multilingual-e5-large-instruct/` — the
      [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
      snapshot (its ONNX weights ship upstream; the infinity `optimum` engine
      serves them).
    - `${embeddings_volume}/bge-reranker-v2-m3/` — the
      [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
      snapshot **plus a pre-exported ONNX model under `onnx/`** (e.g. via
      `optimum-cli export onnx`): the `optimum` engine does not auto-export
      from safetensors, and the container serves both models
      ([why the reranker is here](#the-reranker-rides-the-embedding-container)).

## First-time setup

```bash
git clone https://github.com/luisegarduno/varagity && cd varagity
cp .env.example .env
```

Edit `.env`:

1. Point `models_volume` / `embeddings_volume` at your host model directories
   and set `secret_infinity_key` / `POSTGRES_PASSWORD`.
2. Set `BASE_MODEL` to your `.gguf` filename.
3. Leave the service URLs at their **in-container** values (`http://llamacpp:8080/v1`
   etc.) — the checked-in convention is container values with host overrides
   in comments ([host-vs-container](#host-vs-container-env-usage)).

Put some documents in `./docs/` (the gitignored ingest corpus — **not**
`golden-docs/`, which is this documentation), then:

```bash
docker compose up -d --wait     # exits 0 once every healthcheck is green
bash scripts/smoke.sh           # 6 sequenced infra checks
docker compose logs -f varagity # watch the app ingest, then prompt
```

The `app` container runs `chat` non-interactively (it ingests, prints the
prompt, and exits on stdin EOF). For an interactive session run the CLI
attached, or [from the host](#host-vs-container-env-usage):

```bash
docker compose run --rm app uv run main.py chat
```

Ask something answerable from your corpus; you should get a grounded answer
citing its `[SOURCE]`. `:quit` exits.

## Bring-up order & healthchecks

`docker compose up -d --wait` handles ordering: `app` declares
`depends_on: condition: service_healthy` on all five services, and `--wait`
honors the healthchecks. What "healthy" means per service:

| Service | Probe | Semantics to know |
|---|---|---|
| `llamacpp` | `curl /health` | Returns **503 while the model loads** (~30 s for the current 9B `.gguf`); the generous `retries: 20` covers it. Also unprioritized under load — a busy server can look slow to probe. |
| `infinity-embeddings` | `curl /health` | `/health` is **not** under the `/v1` prefix (API routes are). First boot may re-optimize ONNX graphs, which takes a minute. |
| `postgres` | `pg_isready` | `schema.sql` runs on **first boot only** (empty data dir). |
| `elasticsearch` | `curl /_cluster/health` | Reachability only — **single-node clusters are `yellow` by design** (replicas can never assign). Never gate on `green`. |
| `prefect` | python-urllib `/api/health` | The image ships no `curl`/`wget`, hence the stdlib probe. |

`scripts/smoke.sh` then verifies substance, not just liveness: llama.cpp lists
`BASE_MODEL`, infinity returns a 1024-dim embedding, the pg schema and all
three chunk indexes exist, ES cluster health is yellow/green, Prefect responds.

## Day-to-day usage

```bash
uv run main.py ingest              # ingest DOCS_PATH into both stores
uv run main.py ingest --reingest   # delete + re-process every discovered doc
uv run main.py chat                # ingest, then the Q&A loop (the default)
uv run main.py -v 2 chat           # verbose: full chunk/retrieval panels
uv run --group eval main.py eval       # 4-config retrieval matrix
uv run --group eval main.py eval ocr   # OCR engine benchmark
```

**`--reingest` semantics**: idempotency keys on file *bytes*
(`content_hash`), so pipeline-setting changes (`CONTEXTUALIZE`, chunk params,
`OCR_ENGINE`) do **not** mark files as changed — unchanged files are skipped
until you `--reingest`. It clears each document from **both** stores first,
keeping them consistent.

**Retrieval method** is an env toggle: `RETRIEVAL_METHOD=semantic|bm25|hybrid`
(default `hybrid`). Handy for A/B-ing a query that one method struggles with.

## Host-vs-container `.env` usage

The single `.env` is consumed twice — Docker Compose interpolates
`${lowercase}` vars into service definitions, and the app loads the same file
via pydantic-settings. The checked-in values are the **in-container** ones
(service names: `http://llamacpp:8080/v1`). When running the app on the
**host** (dev loops, tests, eval), override the endpoints per run:

```bash
BASE_MODEL_API_URL=http://localhost:8080/v1 \
POSTGRES_HOST=localhost \
ELASTICSEARCH_URL=http://localhost:9200 \
PREFECT_API_URL=http://localhost:4200/api \
EMBEDDING_API_URL="http://$(docker compose port infinity-embeddings 8081)/v1" \
DOCS_PATH=./docs \
uv run main.py chat
```

!!! warning "infinity's host binding is interface-specific"
    The compose maps infinity to `192.168.86.21:8081`, not `0.0.0.0` — plain
    `localhost:8081` does **not** reach it from the host. Resolve the bound
    address with `docker compose port infinity-embeddings 8081` (as above and
    in `scripts/smoke.sh`).

## Volumes and resets

| Volume | Holds | Reset effect |
|---|---|---|
| `pgdata` | pgvector data — the ingested corpus + metadata | re-runs `schema.sql` on next boot |
| `esdata` | the BM25 index | index recreated on next ingest |
| `model_cache` | Docling layout/table models + EasyOCR weights (`~/.cache` in the app container) | re-downloads on next PDF parse |
| `prefect` | the Prefect server's SQLite backing store | run history lost |

`docker compose down -v` drops all of them — the full factory reset.

!!! warning "Postgres credentials freeze at first boot"
    The `pgdata` volume keeps the password from **first boot**. Editing
    `POSTGRES_PASSWORD` in `.env` later changes what clients *send* but not
    what the server *expects* — host TCP connections start failing auth while
    `docker compose exec postgres psql` (trust-based local socket) still
    works, which is confusing to debug. Fix: `ALTER USER varagity WITH
    PASSWORD '…'` inside the container, or `docker compose down -v` to
    re-initialize.

## GPU & VRAM

**There is no VRAM isolation between containers** — both GPU services see
real device memory, so the total must fit. Current topology on this host:

| GPU | Card | Service | Steady-state VRAM |
|---|---|---|---|
| 0 | RTX 2080 Ti (22.5 GB) | `llamacpp` | ~9.0 GB (MoE experts offloaded to CPU) |
| 1 | RTX 5060 (8 GB) | `infinity-embeddings` | e5 + reranker ONNX, batch-capped to fit |

Pinning gotchas learned the hard way:

- **infinity's `optimum` engine ignores `INFINITY_DEVICE_ID`** — pin at the
  Docker layer with `device_ids: ["1"]` in the compose reservation. Inside
  the container the visible device is then `0`, which is what
  `INFINITY_DEVICE_ID` is set to.
- **`llamacpp` uses `count: 1`**, which grabs the first GPU (device 0).
- The llama.cpp command keeps `-ot ".ffn_(up|down)_exps.=CPU"` — the
  configured `BASE_MODEL` is a MoE model, and this offloads expert FFN
  weights to CPU. It is why an ~9B Q8 model fits in ~9 GB, and also why
  **prompt evaluation is noticeably slower than decode** (~54 tok/s decode;
  slow prefill is the CPU-offload signature, not a bug).

### The reranker rides the embedding container

`bge-reranker-v2-m3` is served by the **same** infinity instance
(semicolon-separated multi-model syntax) and exposed at `/v1/rerank`. It is
**not called by the v1 query path** (`RERANK_ENABLED=false` is staged config
for the post-v1 rerank step). Operational facts if you touch this:

- **The engine must stay `optimum`** on this GPU: the image's torch build has
  no CUDA kernels for the 5060's Blackwell `sm_120` and crashes at warmup;
  onnxruntime works.
- **`INFINITY_BATCH_SIZE: '32;4'`** caps the reranker's batch at 4 — the
  default 32 OOMs an 8 GB card on a single ~2.2 GB attention buffer at
  warmup. Order matches `INFINITY_MODEL_ID` (e5 first).
- **The ONNX must be pre-exported** into the model dir (`onnx/model.onnx`);
  `optimum` does not export from safetensors at serve time.

## First-run model downloads

Beyond the bind-mounted LLM/embedding models, three things download on first
use and are cached afterwards:

- **Docling layout/table models** → `~/.cache/huggingface` (first PDF parse).
- **EasyOCR weights** → pinned to `~/.cache/docling/models/EasyOcr` by our
  engine factory (first OCR fallback).
- **tiktoken's `cl100k_base`** ranks file (first token count; if it can't
  download, counting degrades to a chars/4 estimate with a warning — ingest
  never fails on it).

In-container, the `model_cache` volume (mounted over `/home/user/.cache`)
covers the first two, so they survive rebuilds. Expect the first PDF ingest to
be minutes slower than steady state.

## OCR fallback operations

PDFs take a fast text-layer pass first; OCR (pass 2) triggers automatically
when a document yields < `PDF_OCR_MIN_CHARS` non-whitespace chars, has
≥ `PDF_OCR_TEXTLESS_PAGE_RATIO` textless pages, or pass 1 raised. Chunks
recovered this way carry `extraction: "ocr_fallback"` provenance
(`SELECT ... WHERE metadata->>'extraction' = 'ocr_fallback'`).

- Engines: `OCR_ENGINE=easyocr` (default — ADR-004) or `tesseract`. CPU-only
  by design in v1; throughput measured at ~0.10 pages/s (EasyOCR) vs ~0.55
  (Tesseract) — acceptable because only textless documents pay it, in offline
  batch ingestion.
- `PDF_OCR_FORCE_FULL_PAGE=true` is the escape hatch for **corrupt-text-layer**
  PDFs (garbage embedded text passes the content triggers by definition): it
  skips pass 1 entirely and OCRs every page.
- A PDF where even OCR recovers nothing ends in the empty-extraction guard: a
  0-chunk `documents` row, a warning, and a summary count — never a silent
  drop.

## Elasticsearch notes

- **`yellow` is healthy** on this single-node cluster; only `red` is a
  problem.
- **Disk watermarks**: with the host disk > 90% full, ES's default
  *percentage-based* watermarks refuse to allocate new primary shards — the
  cluster goes `red` and every write hangs until timeout. The compose service
  deliberately keeps the default watermarks (they protect the data volume);
  free host disk instead. The **ephemeral testcontainers** stores used by
  tests/eval set `cluster.routing.allocation.disk.threshold_enabled=false` —
  throwaway containers must never depend on host disk pressure.
- The ES **client major version must match the server major** (9.x ↔ 9.x);
  the dependency is pinned `elasticsearch>=9,<10` accordingly.

## Prefect

- UI at [http://localhost:4200](http://localhost:4200): one flow run per
  `ingest`/question/eval, with per-stage task runs, durations, and logs.
- Backing store is the default **SQLite** in the `prefect` volume (ADR-003) —
  fine for a single-user dev stack; the official Postgres+Redis compose is a
  production posture v1 doesn't need.
- Flows run **in-process** from the CLI; there are no workers, deployments, or
  schedules to manage.
- `PREFECT_API_URL` must be in the environment **before** `prefect` is
  imported (Prefect captures it at import time). The app handles this itself
  (`varagity/pipeline/__init__.py`); it only bites if you script against the
  modules directly.

## Security posture (dev-only)

This stack is a **single-user development posture** — do not expose it:

- Elasticsearch runs with `xpack.security.enabled=false` (no auth, no TLS).
- PostgreSQL uses a static password from `.env`.
- llama.cpp is unauthenticated (`BASE_MODEL_API_KEY` is a placeholder the SDK
  requires).
- infinity has an API key (`secret_infinity_key`) but no TLS; its host
  binding is at least interface-specific.

## Performance expectations

Numbers from this host (see [ADRs](adr/index.md) and eval results for
context):

| Operation | Observed |
|---|---|
| llama.cpp model load | ~30 s |
| Chat decode | ~54 tok/s (slow prefill = MoE CPU-offload signature) |
| Contextualization | ~8 s/chunk (one LLM call per chunk, per-document prompt-cache grouping) |
| Fixtures corpus ingest (16 chunks) | ~42 s non-contextual; ~3 min contextual |
| OCR fallback | ~0.10 pages/s EasyOCR / ~0.55 pages/s Tesseract (CPU) |
| Query (hybrid, top-10) | ~7.5 s, LLM generation dominated; Prefect overhead ≈0.06 s |

## Troubleshooting quick reference

| Symptom | Likely cause → fix |
|---|---|
| `llamacpp` unhealthy for ~5 min after up | Model still loading (503 is normal); check `docker compose logs llamacpp` for the `.gguf` load line |
| Host psql auth failures, but `compose exec postgres psql` works | Password drift — `pgdata` keeps the first-boot password ([above](#volumes-and-resets)) |
| ES `red`, ingest hangs/times out | Host disk > 90% (watermarks) — free disk; testcontainers are immune by config |
| `localhost:8081` unreachable | infinity binds a specific interface — `docker compose port infinity-embeddings 8081` |
| infinity crashes at warmup after GPU/engine changes | torch has no `sm_120` kernels — keep `INFINITY_ENGINE=optimum`; reranker OOM → keep the `32;4` batch cap |
| Config change didn't take effect on ingest | Content hashes unchanged — run `ingest --reingest` |
| First PDF ingest very slow | One-time Docling/EasyOCR model downloads ([above](#first-run-model-downloads)) |
| Flow runs missing from the Prefect UI (host run) | `PREFECT_API_URL` not set for the process — pass the localhost override |
