# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Varagity is a Full Stack RAG (Retrieval-Augmented Generation) application built with Docker. The project leverages GPU-accelerated embedding services and is designed to run on Nvidia GPU infrastructure.

## Architecture

The application uses a microservices architecture with two main components:

1. **infinity-embeddings service**: GPU-accelerated embedding server using the `michaelf34/infinity:latest-trt-onnx` image
   - Provides embedding endpoints at port 8081
   - Uses TensorRT and ONNX optimization for inference
   - Serves the `infloat/multilingual-e5-large-instruct` model
   - Requires Nvidia GPU with nvidia-docker runtime

2. **app service**: Main Varagity application (currently minimal)
   - Built from local Dockerfile using `uv` package manager
   - Python 3.12+ application
   - Entry point: `main.py`

## Development Setup

### Prerequisites

- Minimum 12GB disk space
- Nvidia GPU with nvidia-docker runtime configured
- Docker and Docker Compose

### Environment Configuration

Copy `.env.example` to `.env` and configure:
- `embeddings_volume`: Path to directory containing the embedding model
- `secret_infinity_key`: API key for the infinity embeddings service

### Building and Running

Start all services:
```bash
docker compose up
```

Build the app container manually:
```bash
docker build -t varagity-app . --progress=plain
```

Run the app container standalone:
```bash
docker run --rm varagity-app
```

### Testing the Embeddings API

The FastAPI embeddings endpoint is available at http://localhost:8081/v1/docs

Example embeddings request:
```bash
curl -X 'POST' \
  'http://0.0.0.0:8081/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer YOUR_INFINITY_API_KEY_HERE' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "infloat/multilingual-e5-large-instruct",
  "encoding_format": "float",
  "user": "string",
  "dimensions": 0,
  "input": [
    "This is a sample sentence!"
  ],
  "modality": "text"
}'
```

## Package Management

This project uses `uv` for Python dependency management (https://github.com/astral-sh/uv):

Install dependencies:
```bash
uv sync
```

Run Python scripts:
```bash
uv run main.py
```

Add dependencies by editing `pyproject.toml` dependencies array, then run `uv sync`.

## Planned Features

The project roadmap includes:
- Qdrant-GPU for vector storage
- FastEmbed integration
- Prefect for workflow orchestration
- Prefect-Prometheus-Exporter for monitoring
- Model loader implementation
- Contextual embeddings and BM25
- Re-ranker component
- Pre-commit hooks
