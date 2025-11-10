from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..models.notebook import (
    IngestionJobStatus,
    NotebookIngestionRequest,
    NotebookMetadata,
)
from ..models.rag import RAGQueryRequest, RAGQueryResponse, RAGSource
from ..services.notebook_store import store
from ..services.ingestion import IngestionService
from ..services.rag import RAGService

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


@router.get("/", response_model=list[NotebookMetadata])
async def list_notebooks() -> list[NotebookMetadata]:
    return store.list_notebooks()


@router.post("/ingest", response_model=IngestionJobStatus, status_code=202)
async def enqueue_notebook_ingestion(request: Request, payload: NotebookIngestionRequest) -> IngestionJobStatus:
    if not payload.path:
        raise HTTPException(status_code=400, detail="Path is required")
    job = store.start_ingestion(payload)

    # Kick off ingestion synchronously for MVP (later: background worker)
    ingestion_service: IngestionService = request.app.state.ingestion_service
    try:
        result = ingestion_service.ingest_path(
            notebook_id=job.notebook_id,
            path=Path(payload.path),
            recursive=payload.recursive,
        )
        completed = store.complete_ingestion(
            job_id=job.job_id,
            message=f"Ingested {result.documents_processed} documents, {result.chunks_indexed} chunks.",
            source_count=result.documents_processed,
            chunk_count=result.chunks_indexed,
        )
        return completed
    except Exception as exc:  # pragma: no cover - surfaces to API correctly
        failed = store.fail_ingestion(job.job_id, f"Ingestion failed: {exc}")
        return failed


@router.get("/jobs", response_model=list[IngestionJobStatus])
async def list_ingestion_jobs() -> list[IngestionJobStatus]:
    return store.list_jobs()


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: Request, payload: RAGQueryRequest) -> RAGQueryResponse:
    if not payload.notebook_id or not payload.question:
        raise HTTPException(status_code=400, detail="notebook_id and question are required")

    rag_service: RAGService = request.app.state.rag_service
    result = rag_service.query(notebook_id=payload.notebook_id, question=payload.question, top_k=payload.top_k)
    return RAGQueryResponse(
        answer=result.answer,
        sources=[RAGSource(source_path=s.source_path, content=s.content, distance=s.distance) for s in result.sources],
    )

