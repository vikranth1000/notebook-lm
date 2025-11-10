from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook to search")
    question: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20)


class RAGSource(BaseModel):
    source_path: str
    content: str
    distance: float | None = None


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[RAGSource]

