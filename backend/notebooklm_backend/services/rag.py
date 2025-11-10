from __future__ import annotations

from dataclasses import dataclass
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
        # Expand query for study/help requests to get more relevant chunks
        expanded_query = question
        study_keywords = ["study", "help", "understand", "explain", "learn", "review", "analyze"]
        if any(keyword in question.lower() for keyword in study_keywords):
            # For study requests, retrieve more chunks and use a broader query
            top_k = max(top_k, 10)
            # Add context words to help find relevant content
            expanded_query = f"{question} research paper document content summary key points"
        
        query_results = self.vector_store.query(notebook_id=notebook_id, query=expanded_query, top_k=top_k)

        documents = query_results.get("documents", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]
        distances = query_results.get("distances", [[]])[0] if query_results.get("distances") else []

        if not documents:
            return RAGResponse(
                answer="No relevant documents found in the notebook. Make sure you have uploaded documents.",
                sources=[],
            )

        prompt_context = "\n\n".join(f"Source {idx+1}: {doc}" for idx, doc in enumerate(documents))
        
        # Better prompt for study/help requests
        if any(keyword in question.lower() for keyword in study_keywords):
            prompt = (
                "You are a helpful study assistant. The user wants help studying their uploaded research paper/document. "
                "Provide a comprehensive summary and analysis of the document content. Include:\n"
                "- Main topic and research question\n"
                "- Key findings and conclusions\n"
                "- Important methodologies or approaches\n"
                "- Notable insights or contributions\n\n"
                "Use the provided document chunks to give a thorough overview. Cite sources as [Source #].\n\n"
                f"Document content:\n{prompt_context}\n\n"
                f"User's request: {question}\n\n"
                "Provide a helpful study guide:"
            )
        else:
            prompt = (
                "You are an offline notebook assistant. Answer the user question using the provided sources from their uploaded documents. "
                "Cite sources as [Source #]. Be helpful and summarize the key information from the documents when answering general questions. "
                "If the question is about what the document contains, provide a summary of the document's main content.\n\n"
                f"Context from uploaded documents:\n{prompt_context}\n\nQuestion: {question}\nAnswer:"
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

