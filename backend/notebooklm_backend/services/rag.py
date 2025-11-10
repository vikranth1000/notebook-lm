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

    def query(self, notebook_id: str, question: str, top_k: int = 5) -> RAGResponse:
        query_results = self.vector_store.query(notebook_id=notebook_id, query=question, top_k=top_k)

        documents = query_results.get("documents", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]
        distances = query_results.get("distances", [[]])[0] if query_results.get("distances") else []

        prompt_context = "\n\n".join(f"Source {idx+1}: {doc}" for idx, doc in enumerate(documents))
        prompt = (
            "You are an offline notebook assistant. Answer the user question strictly using the provided sources. "
            "Cite sources as [Source #]. If the answer is not contained in the sources, say you cannot find it.\n\n"
            f"Context:\n{prompt_context}\n\nQuestion: {question}\nAnswer:"
        )

        if self.settings.llm_provider == "none":
            summary_lines = [
                f"[Source {idx + 1}] {doc[:320]}{'...' if len(doc) > 320 else ''}"
                for idx, doc in enumerate(documents)
            ]
            answer = "Offline summary (no LLM configured).\n" + "\n".join(summary_lines)
        else:
            llm = self._ensure_llm()
            answer = llm.generate(prompt, max_tokens=self.settings.llm_max_tokens)

        sources = [
            SourceAttribution(
                source_path=metadata.get("source_path", "unknown") if isinstance(metadata, dict) else "unknown",
                content=document,
                distance=distances[idx] if idx < len(distances) else None,
            )
            for idx, (document, metadata) in enumerate(zip(documents, metadatas))
        ]

        return RAGResponse(answer=answer, sources=sources)

