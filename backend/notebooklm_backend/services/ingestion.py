from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import AppConfig
from .chunking import TextChunk, chunk_text
from .chunking_lc import lc_split_text_to_chunks
from .document_loader import LoadedDocument, DocumentLoaderError, iter_supported_files, load_document
from .document_summary import DocumentSummaryService
from .vector_store import VectorStoreManager


@dataclass
class IngestionResult:
    notebook_id: str
    documents_processed: int
    chunks_indexed: int


class IngestionService:
    def __init__(self, settings: AppConfig, vector_store: VectorStoreManager) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.summary_service = DocumentSummaryService(settings)

    async def ingest_path(
        self,
        notebook_id: str,
        path: Path,
        recursive: bool = True,
    ) -> IngestionResult:
        if not path.exists():
            raise DocumentLoaderError(f"{path} does not exist.")

        files = list(iter_supported_files(path, recursive=recursive))
        if not files:
            raise DocumentLoaderError(f"No supported documents found in {path}.")

        all_chunks: list[TextChunk] = []
        documents_processed = 0

        for file_path in files:
            document = load_document(file_path)
            chunks = list(self._chunk_document(document))
            all_chunks.extend(chunks)
            
            # Generate document summary for two-stage retrieval
            relative_path = str(document.path)
            summary = await self.summary_service.generate_summary(
                text=document.text,
                source_path=relative_path,
            )
            summary.chunk_count = len(chunks)
            
            # Store summary in vector store
            self.vector_store.store_document_summary(notebook_id=notebook_id, summary=summary)
            
            documents_processed += 1

        chunk_count = self.vector_store.add_chunks(notebook_id=notebook_id, chunks=all_chunks)

        return IngestionResult(
            notebook_id=notebook_id,
            documents_processed=documents_processed,
            chunks_indexed=chunk_count,
        )

    def _chunk_document(self, document: LoadedDocument) -> Iterable[TextChunk]:
        relative_path = str(document.path)
        # Prefer LangChain splitter if enabled and available
        if self.settings.use_langchain_splitter:
            try:
                return list(lc_split_text_to_chunks(document.text, source_path=relative_path))
            except Exception:
                # Fall back to simple splitter if LC is unavailable
                pass
        return chunk_text(document.text, source_path=relative_path)

