# Notebook LM Offline Backend

This package exposes the ingestion, retrieval, and summarisation APIs that power the offline Notebook LM experience. It is built around a FastAPI application with modular services for document parsing, embedding, vector search, and local LLM inference.

## Features (Planned)
- Document ingestion pipeline with chunking, metadata extraction, and vector store persistence.
- Local embeddings using `sentence-transformers` models (CPU-friendly) with optional GPU acceleration.
- Retrieval-Augmented Generation endpoint powered by quantised `llama.cpp` models via `llama-cpp-python`.
- Background task scheduler for incremental updates and summarisation routines.
- Speech (WhisperX) and TTS (Piper) integrations gated behind optional extras.

## Project Layout
```
backend/
├── notebooklm_backend/
│   ├── app.py            # FastAPI application factory
│   ├── config.py         # Settings and path management
│   ├── models/           # Pydantic DTOs
│   ├── routes/           # API routers
│   └── services/         # Ingestion, embeddings, RAG workers
├── pyproject.toml
├── README.md
└── tests/
```

## Development
Install dependencies with [uv](https://docs.astral.sh/uv) or `pip`:

```bash
cd backend
uv venv
uv pip install -e ".[dev]"
uv pip compile pyproject.toml > requirements.lock
```

Run the API locally:

```bash
uv run uvicorn notebooklm_backend.app:create_app --factory --reload
```

Smoke test:

```bash
uv run pytest
```

> **Note**  
> Large model weights (Phi-3, WhisperX) are not bundled by default. Add download hooks that cache them under `~/.cache/notebooklm/models` or the path configured in `.env`.

