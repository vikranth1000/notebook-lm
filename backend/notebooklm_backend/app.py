from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .config import AppConfig, get_settings
from .routes import health, chat, documents, rag, notebooks, metrics, speech, export, agent
from .services.chat import ChatService
from .services.embeddings import create_embedding_backend
from .services.ingestion import IngestionService
from .services.llm import create_llm_backend
from .services.rag import RAGService
from .services.rag_llamaindex import LlamaIndexRAGService
from .services.vector_store import create_vector_store
from .services.notebook_store import NotebookStore
from .services.metrics_store import MetricsStore
from .services.speech import SpeechService
from .services.agent import AgentService
from .services.model_profiles import resolve_ollama_model


def create_app() -> FastAPI:
    """Create the FastAPI application for the chat-first Notebook LM backend."""
    settings: AppConfig = get_settings()
    settings.ensure_directories()
    if settings.llm_provider == "ollama":
        resolve_ollama_model(settings)

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
    # Create both services, but use LlamaIndex as primary with custom RAG as fallback
    custom_rag_service = RAGService(settings, vector_store)
    if settings.use_llamaindex_rag:
        try:
            rag_service = LlamaIndexRAGService(settings, vector_store)
            # Store custom RAG as fallback in the service for error recovery
            rag_service._fallback_rag = custom_rag_service
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to initialize LlamaIndex RAG, falling back to custom RAG: {e}")
            rag_service = custom_rag_service
    else:
        rag_service = custom_rag_service
    
    app.state.settings = settings
    notebook_store = NotebookStore(settings)
    metrics_store = MetricsStore(settings)
    agent_service = AgentService(settings)

    app.state.chat_service = ChatService(
        create_llm_backend(settings),
        settings,
        rag_service=rag_service,
        metrics_store=metrics_store,
    )
    app.state.ingestion_service = IngestionService(settings, vector_store)
    app.state.rag_service = rag_service
    app.state.vector_store = vector_store
    app.state.notebook_store = notebook_store
    app.state.metrics_store = metrics_store
    app.state.speech_service = SpeechService(settings)
    app.state.agent_service = agent_service

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
    app.include_router(notebooks.router, prefix="/api")
    app.include_router(metrics.router, prefix="/api")
    app.include_router(speech.router, prefix="/api")
    app.include_router(export.router, prefix="/api")
    app.include_router(agent.router, prefix="/api")

    @app.get("/api/config", tags=["config"])
    async def read_config() -> dict[str, object]:
        return {
            "llm_provider": settings.llm_provider,
            "ollama_model": settings.ollama_model,
            "resolved_ollama_model": settings.resolved_ollama_model or settings.ollama_model,
            "model_selection_reason": settings.model_selection_reason,
            "ollama_base_url": settings.ollama_base_url,
            "use_langchain_splitter": settings.use_langchain_splitter,
            "use_llamaindex_rag": settings.use_llamaindex_rag,
            "embedding_model": settings.embedding_model,
            "enable_speech_stt": settings.enable_speech_stt,
            "enable_speech_tts": settings.enable_speech_tts,
        }

    return app
