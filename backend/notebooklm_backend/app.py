from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .config import AppConfig, get_settings
from .routes import health, notebooks
from .services.embeddings import create_embedding_backend
from .services.vector_store import create_vector_store
from .services.ingestion import IngestionService
from .services.rag import RAGService


def create_app() -> FastAPI:
    """
    Application factory for the offline Notebook LM backend.

    Creates routers for health checks, library management, and retrieval endpoints.
    Additional startup hooks ensure required directories exist (cache, vector store, models).
    """
    settings: AppConfig = get_settings()

    settings.ensure_directories()

    app = FastAPI(
        title="Offline Notebook LM",
        version="0.1.0",
        description="Local-first API for offline Notebook LM clone.",
        contact={"name": "Offline Notebook LM Team"},
    )

    # Bootstrap core services
    embedding_backend = create_embedding_backend(settings)
    vector_store = create_vector_store(settings, embedding_backend)
    ingestion_service = IngestionService(settings, vector_store)
    rag_service = RAGService(settings, vector_store)

    app.state.settings = settings
    app.state.embedding_backend = embedding_backend
    app.state.vector_store = vector_store
    app.state.ingestion_service = ingestion_service
    app.state.rag_service = rag_service

    # Allow renderer (http://localhost:5173) to call the API in dev; loosened in v1 for simplicity.
    # Electron app file:// and packaged mode will call localhost as well.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_origin_regex=".*",  # tolerate dev variations
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(notebooks.router, prefix="/api")

    @app.get("/api/config", tags=["config"])
    async def read_config() -> dict[str, str | bool]:
        return {
            "models_dir": str(settings.models_dir),
            "indexes_dir": str(settings.index_dir),
            "workspace_root": str(settings.workspace_root),
            "enable_audio": settings.enable_audio,
        }

    return app

