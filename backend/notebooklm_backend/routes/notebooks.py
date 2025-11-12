from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.notebook import IngestionJobStatus, NotebookMetadata
from ..services.notebook_store import NotebookStore

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


@router.get("/", response_model=list[NotebookMetadata])
async def list_notebooks(request: Request) -> list[NotebookMetadata]:
    store: NotebookStore = request.app.state.notebook_store
    return store.list_notebooks()


@router.get("/jobs", response_model=list[IngestionJobStatus])
async def list_jobs(request: Request) -> list[IngestionJobStatus]:
    store: NotebookStore = request.app.state.notebook_store
    return store.list_jobs()
