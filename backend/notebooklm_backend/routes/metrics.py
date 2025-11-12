from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.metrics import MetricsSummary
from ..services.metrics_store import MetricsStore

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/summary", response_model=MetricsSummary)
async def metrics_summary(request: Request) -> MetricsSummary:
    store: MetricsStore = request.app.state.metrics_store
    return store.summary()
