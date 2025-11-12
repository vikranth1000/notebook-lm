from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatMetricRecord(BaseModel):
    metric_id: str
    notebook_id: str | None = None
    provider: str
    prompt: str
    created_at: datetime
    total_ms: float | None = None
    llm_ms: float | None = None
    stage1_ms: float | None = None
    retrieval_ms: float | None = None
    documents_considered: float | None = None
    source_count: int | None = None
    tokens: int | None = None


class MetricsSummary(BaseModel):
    conversations: int
    avg_total_ms: float | None = None
    avg_llm_ms: float | None = None
    avg_retrieval_ms: float | None = None
    provider_breakdown: dict[str, int] = Field(default_factory=dict)
