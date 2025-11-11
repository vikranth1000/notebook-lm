from __future__ import annotations

from dataclasses import dataclass
from typing import List
import logging

from ..config import AppConfig
from .vector_store import VectorStoreManager
from .rag import RAGResponse, SourceAttribution

logger = logging.getLogger(__name__)


class LlamaIndexRAGService:
    """
    LlamaIndex-based RAG that reuses our existing Chroma collections.
    Stays fully offline by using the local Ollama LLM.
    """

    def __init__(self, settings: AppConfig, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store

    async def query(self, notebook_id: str, question: str, top_k: int = 10) -> RAGResponse:
        """
        Query using LlamaIndex, returning RAGResponse for compatibility with ChatService.
        """
        try:
            # Import lazily to avoid cost when disabled
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.llms.ollama import Ollama
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.info(f"LlamaIndex RAG query for notebook {notebook_id[:8]}...")

            collection = self.vector_store.get_collection(notebook_id)
            
            # Check if collection has any documents
            collection_count = collection.count()
            if collection_count == 0:
                logger.warning(f"Collection {notebook_id} is empty")
                return RAGResponse(
                    answer="No documents found in this notebook. Please upload documents first.",
                    sources=[],
                )

            # Use the same embedding model as our vector store
            embedding_model = HuggingFaceEmbedding(model_name=self.settings.embedding_model)
            Settings.embed_model = embedding_model

            li_store = ChromaVectorStore(chroma_collection=collection)

            # Build an index that points at the existing vector store
            index = VectorStoreIndex.from_vector_store(li_store)

            # Offline LLM via Ollama
            llm = Ollama(model=self.settings.ollama_model, base_url=self.settings.ollama_base_url, request_timeout=120.0)

            # Query engine with similarity_top_k
            query_engine = index.as_query_engine(
                similarity_top_k=max(top_k, 10),
                llm=llm,
            )
            
            logger.info(f"Executing LlamaIndex query...")
            result = query_engine.query(question)

            # Extract answer
            answer = str(getattr(result, "response", result))
            logger.info(f"LlamaIndex query completed, answer length: {len(answer)}")

            # Extract sources if available
            sources: list[SourceAttribution] = []
            try:
                source_nodes = getattr(result, "source_nodes", []) or []
                logger.info(f"Found {len(source_nodes)} source nodes")
                for node_with_score in source_nodes:
                    node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
                    meta = getattr(node, "metadata", {}) or {}
                    text = getattr(node, "text", "") or str(node)
                    score = getattr(node_with_score, "score", None) if hasattr(node_with_score, "score") else None
                    
                    sources.append(
                        SourceAttribution(
                            source_path=str(meta.get("source_path", "unknown")),
                            content=text,
                            distance=float(score) if score is not None else None,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error extracting sources: {e}")

            return RAGResponse(answer=answer, sources=sources)

        except ImportError as e:
            logger.error(f"LlamaIndex import failed: {e}")
            raise RuntimeError("LlamaIndex dependencies not installed. Run: pip install llama-index-core llama-index-vector-stores-chroma llama-index-llms-ollama") from e
        except Exception as e:
            logger.error(f"LlamaIndex RAG query failed: {e}", exc_info=True)
            # If we have a fallback RAG service, use it instead of raising
            if hasattr(self, "_fallback_rag"):
                logger.info("Falling back to custom RAG service")
                return await self._fallback_rag.query(notebook_id, question, top_k)
            raise


