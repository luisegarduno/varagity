# Varagity

Full-stack RAG application with **Contextual Retrieval** — self-hosted,
GPU-accelerated, terminal-first. Documents in `docs/` are parsed (PDF/txt/md,
with automatic OCR fallback for scans), chunked, situated with LLM-generated
context blurbs, and indexed into **both** pgvector (semantic) and
Elasticsearch (contextual BM25); questions are answered with grounded, cited
answers via hybrid rank-fusion retrieval. Every pipeline stage is a tracked
Prefect task.

> 📄 **Design**: [`spec.md`](spec.md) · **As-built docs**: [`golden-docs/`](golden-docs/index.md)
> (`uv run mkdocs serve`) — architecture, data model, pipelines, runbook, ADRs.


## Status

Project ToDo's:
- [x] Create pre-commit rules

Stack ToDo's:
- [x] Add: PostgreSQL + pgvector
- [x] Add: Elasticsearch (contextual BM25)
- [x] Add: llama.cpp server (self-hosted LLM)
- [x] Add: Prefect
- [ ] Add: Prefect-Prometheus-Exporter *(post-v1)*

RAG ToDo's:
- [x] Model Loader
- [x] Contextual Embeddings
- [x] Contextual BM25
- [x] Evaluation harness (recall@k / pass@k matrix + OCR benchmark)
- [ ] Re-ranker *(post-v1 — `bge-reranker-v2-m3` is already served at `/v1/rerank`;
  the query-path step is staged behind `RERANK_ENABLED=false`)*


-----------------------------

# Instructions

## Pre-Req's

*  \>=12GB's of disk space
* Two Nvidia GPUs (llama.cpp on GPU 0, infinity embeddings on GPU 1 — see the
  [runbook](golden-docs/runbook.md) to adapt the pinning)
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

2. Put documents (`.pdf`, `.txt`, `.md`) into `./docs/`, then bring up the
   stack — six services with healthchecks; `--wait` returns once all are green:
    ```bash
    docker compose up -d --wait
    bash scripts/smoke.sh        # optional: 6 sequenced infra checks
    ```

3. Ask questions. The `app` container ingests on start; for an interactive
   session:
    ```bash
    docker compose run --rm app uv run main.py chat
    ```
    You get the top-10 matches table and a grounded answer citing its
    `[SOURCE]`. Type `:quit` to exit.

### CLI

```bash
uv run main.py ingest              # ingest DOCS_PATH into both stores
uv run main.py ingest --reingest   # re-process after pipeline-setting changes
uv run main.py chat                # ingest, then Q&A loop (the default command)
uv run --group eval main.py eval       # retrieval-quality matrix (4 configs × k)
uv run --group eval main.py eval ocr   # OCR engine benchmark (CER/WER, pages/s)
```

Run on the host against the containerized services with localhost overrides —
see [host-vs-container `.env` usage](golden-docs/runbook.md#host-vs-container-env-usage).

Prefect UI (flow/task runs, logs, timings): [http://localhost:4200](http://localhost:4200).
Embeddings API docs (infinity): `http://<bound-interface>:8081/v1/docs` — the
binding is interface-specific; resolve it with
`docker compose port infinity-embeddings 8081`.

## Development

This project uses [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync                            # install deps (incl. dev/eval groups)
uv run pytest                      # unit suite (coverage floor: 80%)
uv run pytest -m integration       # real Postgres/ES via testcontainers (Docker)
uv run pytest -m e2e               # full ingest→query over the fixtures corpus (Docker)
uv run ruff check . && uv run ruff format --check .
uv run mypy varagity
uv run pre-commit run --all-files  # lint + format + types + unit tests
uv run mkdocs serve                # the golden-docs site, live-reloading
```

CI (GitHub Actions) runs lint, format check, types, the unit suite, and a
strict docs build on every push; integration/e2e stay local (they need Docker).
