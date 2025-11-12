from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NotebookMetadata(BaseModel):
    notebook_id: str = Field(..., description="Unique notebook identifier")
    title: str
    description: str | None = None
    source_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class NotebookIngestionRequest(BaseModel):
    notebook_id: str | None = None
    path: str
    title: str | None = None
    description: str | None = None


class IngestionJobStatus(BaseModel):
    job_id: str
    notebook_id: str
    status: Literal["pending", "running", "completed", "failed"]
    message: str
    started_at: datetime
    completed_at: datetime | None = None
    documents_processed: int | None = None
    chunks_indexed: int | None = None
