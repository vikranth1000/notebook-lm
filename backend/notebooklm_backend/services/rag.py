from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..config import AppConfig
from .vector_store import VectorStoreManager
from .llm import create_llm_backend, LLMBackend


@dataclass
class SourceAttribution:
    source_path: str
    content: str
    distance: float | None


@dataclass
class RAGResponse:
    answer: str
    sources: List[SourceAttribution]


class RAGService:
    def __init__(self, settings: AppConfig, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self._llm: LLMBackend | None = None

    def _ensure_llm(self) -> LLMBackend:
        if self._llm is None:
            self._llm = create_llm_backend(self.settings)
        return self._llm

    async def query(self, notebook_id: str, question: str, top_k: int = 5) -> RAGResponse:
        # Unified RAG query - works for all document types and question types
        # Retrieve significantly more chunks to ensure we get content from all documents
        # This helps when multiple documents are in the same notebook
        top_k = max(top_k, 20)
        
        query_results = self.vector_store.query(notebook_id=notebook_id, query=question, top_k=top_k)

        documents = query_results.get("documents", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]
        distances = query_results.get("distances", [[]])[0] if query_results.get("distances") else []

        if not documents:
            return RAGResponse(
                answer="No relevant documents found in the notebook. Make sure you have uploaded documents.",
                sources=[],
            )

        # Group chunks by source file to understand document diversity
        source_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (doc, metadata) in enumerate(zip(documents, metadatas)):
            source_path = metadata.get("source_path", "unknown") if isinstance(metadata, dict) else "unknown"
            if source_path not in source_groups:
                source_groups[source_path] = []
            source_groups[source_path].append((idx, doc))
        
        # Build context with source information
        prompt_parts = []
        for source_path, chunks in source_groups.items():
            source_name = Path(source_path).name if source_path != "unknown" else "Document"
            prompt_parts.append(f"From {source_name}:")
            for idx, doc in chunks:
                prompt_parts.append(f"  [Source {idx+1}]: {doc}")
            prompt_parts.append("")
        
        prompt_context = "\n".join(prompt_parts)
        
        # Unified prompt that works for all document types and question types
        # The LLM will naturally understand the user's intent from their question
        prompt = (
            "You are a helpful assistant. The user has uploaded one or more documents and is asking questions about them. "
            "Answer their question using ONLY the information provided in the document excerpts below.\n\n"
            "Important: The user may have multiple documents uploaded. Pay attention to which document(s) are most relevant to their question. "
            "If they mention a specific document type (e.g., 'research paper', 'resume'), focus on content from that type of document.\n\n"
            "Guidelines:\n"
            "- Base your answer strictly on the provided document content\n"
            "- If the question asks for analysis, feedback, or improvement, provide thoughtful insights based on the content\n"
            "- If the question asks for a summary, provide a comprehensive overview\n"
            "- If the question asks for specific information, extract and present it clearly\n"
            "- Cite sources as [Source #] when referencing specific excerpts\n"
            "- If the answer cannot be found in the provided content, say so clearly\n"
            "- Be helpful, accurate, and specific\n\n"
            f"Document excerpts:\n{prompt_context}\n\n"
            f"User's question: {question}\n\n"
            "Answer:"
        )

        if self.settings.llm_provider == "none":
            summary_lines = [
                f"[Source {idx + 1}] {doc[:320]}{'...' if len(doc) > 320 else ''}"
                for idx, doc in enumerate(documents)
            ]
            answer = "Offline summary (no LLM configured).\n" + "\n".join(summary_lines)
        else:
            llm = self._ensure_llm()
            answer = await llm.generate(prompt, max_tokens=self.settings.llm_max_tokens)

        sources = [
            SourceAttribution(
                source_path=metadata.get("source_path", "unknown") if isinstance(metadata, dict) else "unknown",
                content=document,
                distance=distances[idx] if idx < len(distances) else None,
            )
            for idx, (document, metadata) in enumerate(zip(documents, metadatas))
        ]

        return RAGResponse(answer=answer, sources=sources)

