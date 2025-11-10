from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..models.rag import RAGQueryRequest, RAGQueryResponse, RAGSource
from ..services.rag import RAGService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: Request, payload: RAGQueryRequest) -> RAGQueryResponse:
    """Query a notebook using RAG."""
    service: RAGService = request.app.state.rag_service
    
    try:
        result = await service.query(
            notebook_id=payload.notebook_id,
            question=payload.question,
            top_k=payload.top_k,
        )
        
        return RAGQueryResponse(
            answer=result.answer,
            sources=[
                RAGSource(
                    source_path=src.source_path,
                    content=src.content,
                    distance=src.distance,
                )
                for src in result.sources
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

