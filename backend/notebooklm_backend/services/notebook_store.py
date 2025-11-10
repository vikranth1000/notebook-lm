from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from ..models.notebook import IngestionJobStatus, NotebookIngestionRequest, NotebookMetadata


class NotebookStore:
    """
    Simple in-memory store that will later be backed by SQLite.
    """

    def __init__(self) -> None:
        self._notebooks: Dict[str, NotebookMetadata] = {}
        self._jobs: Dict[str, IngestionJobStatus] = {}

    def list_notebooks(self) -> list[NotebookMetadata]:
        return list(self._notebooks.values())

    def get_notebook(self, notebook_id: str) -> NotebookMetadata | None:
        return self._notebooks.get(notebook_id)

    def upsert_notebook(self, metadata: NotebookMetadata) -> NotebookMetadata:
        metadata.updated_at = datetime.utcnow()
        self._notebooks[metadata.notebook_id] = metadata
        return metadata

    def start_ingestion(self, request: NotebookIngestionRequest) -> IngestionJobStatus:
        notebook_id = request.notebook_id or uuid.uuid4().hex
        job_id = uuid.uuid4().hex

        metadata = self.get_notebook(notebook_id)
        if metadata is None:
            metadata = NotebookMetadata(
                notebook_id=notebook_id,
                title=Path(request.path).stem or notebook_id,
                description=f"Imported from {request.path}",
                source_count=0,
            )

        metadata.updated_at = datetime.utcnow()
        self.upsert_notebook(metadata)

        job = IngestionJobStatus(
            job_id=job_id,
            notebook_id=notebook_id,
            status="running",
            message="Ingestion started.",
            started_at=datetime.utcnow(),
        )
        self._jobs[job_id] = job
        return job

    def complete_ingestion(
        self,
        job_id: str,
        message: str,
        source_count: int,
        chunk_count: int,
    ) -> IngestionJobStatus:
        job = self._jobs[job_id]
        job.status = "completed"
        job.message = message
        job.completed_at = datetime.utcnow()
        job.documents_processed = source_count
        job.chunks_indexed = chunk_count
        self._jobs[job_id] = job

        notebook = self._notebooks[job.notebook_id]
        notebook.source_count = source_count
        notebook.chunk_count = chunk_count
        notebook.updated_at = datetime.utcnow()
        self._notebooks[job.notebook_id] = notebook

        return job

    def fail_ingestion(self, job_id: str, message: str) -> IngestionJobStatus:
        job = self._jobs[job_id]
        job.status = "failed"
        job.message = message
        job.completed_at = datetime.utcnow()
        self._jobs[job_id] = job
        return job

    def list_jobs(self) -> list[IngestionJobStatus]:
        return list(self._jobs.values())


store = NotebookStore()

