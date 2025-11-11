from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..config import AppConfig
from .vector_store import VectorStoreManager


@dataclass
class LI_SourceAttribution:
    source_path: str
    content: str
    score: float | None


@dataclass
class LI_RAGResponse:
    answer: str
    sources: List[LI_SourceAttribution]


class LlamaIndexRAGService:
    """
    LlamaIndex-based RAG that reuses our existing Chroma collections.
    Stays fully offline by using the local Ollama LLM.
    """

    def __init__(self, settings: AppConfig, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store

    async def query(self, notebook_id: str, question: str, top_k: int = 10) -> LI_RAGResponse:
        # Import lazily to avoid cost when disabled
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import VectorStoreIndex
        from llama_index.llms.ollama import Ollama

        collection = self.vector_store.get_collection(notebook_id)
        li_store = ChromaVectorStore(chroma_collection=collection)

        # Build an index that points at the existing vector store
        index = VectorStoreIndex.from_vector_store(li_store)

        # Offline LLM via Ollama
        llm = Ollama(model=self.settings.ollama_model, base_url=self.settings.ollama_base_url)

        # Query engine with similarity_top_k
        query_engine = index.as_query_engine(similarity_top_k=max(top_k, 10), llm=llm)
        result = query_engine.query(question)

        # Extract sources if available
        sources: list[LI_SourceAttribution] = []
        try:
            for node_with_score in getattr(result, "source_nodes", []) or []:
                node = node_with_score.node
                meta = node.metadata or {}
                sources.append(
                    LI_SourceAttribution(
                        source_path=str(meta.get("source_path", "unknown")),
                        content=str(getattr(node, "text", "")),
                        score=float(getattr(node_with_score, "score", 0.0)) if node_with_score else None,
                    )
                )
        except Exception:
            pass

        return LI_RAGResponse(answer=str(getattr(result, "response", result)), sources=sources)


