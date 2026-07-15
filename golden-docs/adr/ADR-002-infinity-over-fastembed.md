# ADR-002: infinity embedding service over FastEmbed

**Status:** Accepted (spec §21 #2)

## Context

The original stack sketch listed **FastEmbed** (in-process embedding, as the
reference implementation did with `HuggingFaceEmbeddings`). The architecture,
however, is service-oriented: self-hosted models behind OpenAI-compatible
HTTP endpoints, GPU work isolated in dedicated containers.

## Decision

Serve embeddings with **infinity** (`michaelf34/infinity`, TRT/ONNX image,
tag pinned) running `multilingual-e5-large-instruct` (1024-dim) on a
dedicated GPU. FastEmbed is dropped.

## Rationale

- **Consistency of the serving architecture.** The LLM is already a service
  (llama.cpp); embeddings as a service means the app container stays
  CPU-only and thin, and GPU placement is a compose concern, not a Python
  one.
- **GPU isolation and pinning.** In-process embedding would drag CUDA into
  the app image and contend for whichever GPU the app landed on. As a
  service, infinity is pinned to GPU 1 at the Docker layer.
- **OpenAI-compatible `/v1`** means the same `openai` SDK client pattern as
  the LLM — one client library, two base URLs.
- **Optimized inference for free**: the TRT/ONNX image runs the e5 model via
  optimum/onnxruntime with graph optimizations.
- **It was already in the compose file**, serving this exact model, before
  the v1 build began.

## Consequences

- The client (`varagity/models/embeddings.py`) owns the **asymmetric e5
  formatting** — passages raw, queries `Instruct: {task}\nQuery: {q}` —
  because getting it wrong degrades recall *silently*. Encapsulating both
  modes in one client is the direct consequence of embedding-over-HTTP:
  no caller can format inconsistently.
- Batching (`EMBEDDING_BATCH_SIZE`) and `tenacity` retries live in the
  client; the SDK's own retry is disabled so behavior is single-layered.
- The multi-model serving capability came for free later: the same
  container now also serves `bge-reranker-v2-m3` at `/v1/rerank` for the
  post-v1 rerank step (staged config, `RERANK_ENABLED=false`) — see the
  [runbook](../runbook.md#the-reranker-rides-the-embedding-container) for
  the `sm_120`/optimum/batch-cap operational notes. *(v2 update: no longer
  staged — reranking is wired into the query path; see
  [ADR-006](ADR-006-reranking-wired.md).)*
- Upstream does not CI-build the `latest-trt-onnx` tag, so the compose pins
  an exact version (`0.0.76-trt-onnx`) instead of a moving `latest`.
