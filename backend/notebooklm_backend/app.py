from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .config import AppConfig, get_settings
from .routes import health, chat, documents, rag
from .services.chat import ChatService
from .services.embeddings import create_embedding_backend
from .services.ingestion import IngestionService
from .services.llm import create_llm_backend
from .services.rag import RAGService
from .services.rag_llamaindex import LlamaIndexRAGService
from .services.vector_store import create_vector_store


def create_app() -> FastAPI:
    """Create the FastAPI application for the chat-first Notebook LM backend."""
    settings: AppConfig = get_settings()
    settings.ensure_directories()

    app = FastAPI(
        title="Offline Notebook LM",
        version="0.1.0",
        description="Local-first API for chatting with local models (Ollama) with RAG support.",
        contact={"name": "Offline Notebook LM Team"},
    )

    # Bootstrap services
    embedding_backend = create_embedding_backend(settings)
    vector_store = create_vector_store(settings, embedding_backend)
    # Choose RAG engine based on settings (default to LlamaIndex if enabled)
    if settings.use_llamaindex_rag:
        rag_service = LlamaIndexRAGService(settings, vector_store)
    else:
        rag_service = RAGService(settings, vector_store)
    
    app.state.settings = settings
    app.state.chat_service = ChatService(create_llm_backend(settings), settings, rag_service=rag_service)
    app.state.ingestion_service = IngestionService(settings, vector_store)
    app.state.rag_service = rag_service
    app.state.vector_store = vector_store

    # Allow renderer (http://localhost:5173) to call the API in dev; loosened in v1 for simplicity.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "null",  # Electron file:// origin
        ],
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(rag.router, prefix="/api")

    @app.get("/api/config", tags=["config"])
    async def read_config() -> dict[str, str]:
        return {
            "llm_provider": settings.llm_provider,
            "ollama_model": settings.ollama_model,
            "ollama_base_url": settings.ollama_base_url,
        }

    return app
