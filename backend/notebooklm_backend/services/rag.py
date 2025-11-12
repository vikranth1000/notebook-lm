from __future__ import annotations

import time
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
    metrics: dict[str, float] | None = None


@dataclass
class RAGContext:
    prompt: str
    sources: List[SourceAttribution]
    metrics: dict[str, float]


class RAGService:
    def __init__(self, settings: AppConfig, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self._llm: LLMBackend | None = None

    def _ensure_llm(self) -> LLMBackend:
        if self._llm is None:
            self._llm = create_llm_backend(self.settings)
        return self._llm

    async def prepare_prompt(self, notebook_id: str, question: str, top_k: int = 5) -> RAGContext:
        """Prepare the grounded prompt, sources, and retrieval metrics."""
        total_start = time.perf_counter()
        metrics: dict[str, float] = {}
        # Stage 1: Query document summaries to find relevant documents
        stage1_start = time.perf_counter()
        relevant_summaries = self.vector_store.query_document_summaries(
            notebook_id=notebook_id,
            query=question,
            top_k=3,  # Get top 3 most relevant documents
        )
        metrics["stage1_ms"] = (time.perf_counter() - stage1_start) * 1000
        
        # If we have summaries, use two-stage retrieval with strict per-document queries
        if relevant_summaries:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Two-stage retrieval: Found {len(relevant_summaries)} relevant documents")

            # Select top-N documents
            doc_select_k = max(1, self.settings.rag_doc_select_k)
            selected = relevant_summaries[:doc_select_k]
            selected_paths = [s.source_path for s in selected]

            # Stage 2: query each selected document separately
            retrieval_start = time.perf_counter()
            per_doc = max(1, (self.settings.rag_top_k or top_k) // len(selected_paths))
            documents: list[str] = []
            metadatas: list[dict] = []
            distances: list[float] = []

            for spath in selected_paths:
                try:
                    res = self.vector_store.query(
                        notebook_id=notebook_id,
                        query=question,
                        top_k=per_doc,
                        where={"source_path": {"$eq": spath}},
                    )
                    docs = res.get("documents", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    dists = res.get("distances", [[]])[0] if res.get("distances") else []
                except Exception:
                    # Fallback: unfiltered query + manual filter
                    res = self.vector_store.query(notebook_id=notebook_id, query=question, top_k=per_doc * 2)
                    docs = res.get("documents", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    dists = res.get("distances", [[]])[0] if res.get("distances") else []
                    filtered = [(d, m, dists[i] if i < len(dists) else None) for i, (d, m) in enumerate(zip(docs, metas)) if isinstance(m, dict) and m.get("source_path") == spath]
                    if filtered:
                        docs, metas, dists = zip(*filtered)
                        docs, metas, dists = list(docs), list(metas), list(dists)
                    else:
                        docs, metas, dists = [], [], []

                documents.extend(docs[:per_doc])
                metadatas.extend(metas[:per_doc])
                # distances may be None if not available; normalize
                distances.extend([(d or 0.0) for d in dists[:per_doc]])

            metrics["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000
        else:
            # Fallback to single-stage if no summaries available
            retrieval_start = time.perf_counter()
            query_results = self.vector_store.query(notebook_id=notebook_id, query=question, top_k=max(top_k, 20))
            metrics["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000
            documents = query_results.get("documents", [[]])[0]
            metadatas = query_results.get("metadatas", [[]])[0]
            distances = query_results.get("distances", [[]])[0] if query_results.get("distances") else []

        if not documents:
            return RAGContext(
                prompt="",
                sources=[],
                metrics=metrics,
            )

        # Group chunks by source file to understand document diversity
        source_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (doc, metadata) in enumerate(zip(documents, metadatas)):
            source_path = metadata.get("source_path", "unknown") if isinstance(metadata, dict) else "unknown"
            if source_path not in source_groups:
                source_groups[source_path] = []
            source_groups[source_path].append((idx, doc))
        
        # Build context with strict source separation
        prompt_parts = []
        for source_path, chunks in source_groups.items():
            source_name = Path(source_path).name if source_path != "unknown" else "Document"
            prompt_parts.append(f"From {source_name}:")
            for idx, doc in chunks:
                prompt_parts.append(f"  [Source {idx+1}]: {doc}")
            prompt_parts.append("")
        
        prompt_context = "\n".join(prompt_parts)
        
        # Strict prompt that prevents cross-document mixing
        selected_names = ", ".join([Path(sp).name for sp in source_groups.keys()])
        prompt = (
            "You are answering a question using excerpts FROM SPECIFIC DOCUMENTS ONLY.\n"
            f"Selected documents: {selected_names}.\n"
            "Rules:\n"
            "- Use only the excerpts from the selected documents below. Do not use outside knowledge.\n"
            "- If the question refers to one document (e.g., a resume), ignore all others unless needed to clarify.\n"
            "- If the answer is not present in the excerpts, reply: 'I could not find this in the provided documents.'\n"
            "- When citing, use [Source #] as shown in the excerpts.\n\n"
            f"Excerpts grouped by document:\n{prompt_context}\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        sources = [
            SourceAttribution(
                source_path=metadata.get("source_path", "unknown") if isinstance(metadata, dict) else "unknown",
                content=document,
                distance=distances[idx] if idx < len(distances) else None,
            )
            for idx, (document, metadata) in enumerate(zip(documents, metadatas))
        ]

        metrics["prep_ms"] = (time.perf_counter() - total_start) * 1000
        metrics["total_ms"] = metrics["prep_ms"]
        if relevant_summaries:
            metrics["documents_considered"] = float(len(relevant_summaries))
        return RAGContext(prompt=prompt, sources=sources, metrics=metrics)

    async def query(self, notebook_id: str, question: str, top_k: int = 5) -> RAGResponse:
        """
        Two-stage RAG query that returns the fully generated answer.
        """
        context = await self.prepare_prompt(notebook_id=notebook_id, question=question, top_k=top_k)
        if not context.sources:
            return RAGResponse(
                answer="No relevant documents found in the notebook. Make sure you have uploaded documents.",
                sources=[],
                metrics=context.metrics,
            )

        metrics = context.metrics
        if self.settings.llm_provider == "none":
            summary_lines = [
                f"[Source {idx + 1}] {src.content[:320]}{'...' if len(src.content) > 320 else ''}"
                for idx, src in enumerate(context.sources)
            ]
            answer = "Offline summary (no LLM configured).\n" + "\n".join(summary_lines)
            metrics["llm_ms"] = 0.0
        else:
            llm_start = time.perf_counter()
            llm = self._ensure_llm()
            answer = await llm.generate(context.prompt, max_tokens=self.settings.llm_max_tokens)
            metrics["llm_ms"] = (time.perf_counter() - llm_start) * 1000
            metrics["total_ms"] = metrics.get("total_ms", 0.0) + metrics["llm_ms"]

        return RAGResponse(answer=answer, sources=context.sources, metrics=metrics)
