from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig
from .llm import create_llm_backend

logger = logging.getLogger(__name__)


@dataclass
class DocumentSummary:
    """Summary metadata for a document."""
    source_path: str
    summary: str
    chunk_count: int
    document_type: str | None = None  # e.g., "resume", "research_paper", "contract"


class DocumentSummaryService:
    """Service for generating and managing document summaries."""
    
    def __init__(self, settings: AppConfig):
        self.settings = settings
        self._llm: None = None
    
    def _ensure_llm(self):
        """Lazy load LLM backend."""
        if self._llm is None:
            self._llm = create_llm_backend(self.settings)
        return self._llm
    
    async def generate_summary(self, text: str, source_path: str) -> DocumentSummary:
        """
        Generate a one-line summary of a document.
        
        Uses LLM to create a concise summary that helps with document filtering.
        """
        # Take first 2000 chars for summary generation (enough context)
        preview_text = text[:2000] + ("..." if len(text) > 2000 else "")
        
        prompt = (
            "Generate a concise one-line summary (maximum 100 words) of this document. "
            "Focus on the main topic, purpose, or content type. "
            "Be specific enough to help identify if this document is relevant to a query.\n\n"
            f"Document preview:\n{preview_text}\n\n"
            "Summary:"
        )
        
        if self.settings.llm_provider == "none":
            return self._fallback_summary(preview_text, source_path)

        try:
            llm = self._ensure_llm()
            summary = await llm.generate(prompt, max_tokens=150)  # Short summary
            summary = (summary or "").strip()
            if not summary:
                return self._fallback_summary(preview_text, source_path)
        except Exception as e:
            logger.warning(f"Failed to generate summary for {source_path}: {e}")
            return self._fallback_summary(preview_text, source_path)

        doc_type = self._infer_document_type(summary, source_path)

        return DocumentSummary(
            source_path=source_path,
            summary=summary,
            chunk_count=0,  # Will be set by caller
            document_type=doc_type,
        )

    def _fallback_summary(self, preview_text: str, source_path: str) -> DocumentSummary:
        """Generate a deterministic summary without calling an LLM."""
        snippet = preview_text.splitlines()[0] if preview_text else ""
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        filename = Path(source_path).name or "document"
        summary = f"{filename}: {snippet or 'Text document ingested offline.'}"
        return DocumentSummary(
            source_path=source_path,
            summary=summary,
            chunk_count=0,
            document_type=self._infer_document_type(summary, source_path),
        )
    
    def _infer_document_type(self, summary: str, source_path: str) -> str | None:
        """Infer document type from summary text or filename."""
        summary_lower = summary.lower()
        path_lower = str(source_path).lower()
        
        # Common document types
        if any(term in summary_lower or term in path_lower for term in ["resume", "cv", "curriculum vitae"]):
            return "resume"
        elif any(term in summary_lower or term in path_lower for term in ["research", "paper", "publication", "arxiv"]):
            return "research_paper"
        elif any(term in summary_lower or term in path_lower for term in ["contract", "agreement", "legal"]):
            return "contract"
        elif any(term in summary_lower or term in path_lower for term in ["code", "programming", "software", ".py", ".js"]):
            return "code"
        elif any(term in summary_lower or term in path_lower for term in ["presentation", "slides", ".pptx", ".ppt"]):
            return "presentation"
        
        return None
