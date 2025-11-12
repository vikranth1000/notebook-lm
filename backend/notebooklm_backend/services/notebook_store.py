from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..config import AppConfig
from ..models.notebook import IngestionJobStatus, NotebookIngestionRequest, NotebookMetadata


class NotebookStore:
    """
    SQLite-backed metadata store for notebooks and ingestion jobs.
    """

    def __init__(self, settings: AppConfig) -> None:
        self.db_path = settings.workspace_root / "metadata.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notebooks (
                    notebook_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    source_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    notebook_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    documents_processed INTEGER,
                    chunks_indexed INTEGER,
                    FOREIGN KEY (notebook_id) REFERENCES notebooks (notebook_id)
                )
                """
            )

    def list_notebooks(self) -> list[NotebookMetadata]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM notebooks ORDER BY updated_at DESC").fetchall()
        return [self._row_to_notebook(row) for row in rows]

    def get_notebook(self, notebook_id: str) -> NotebookMetadata | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM notebooks WHERE notebook_id = ?", (notebook_id,)).fetchone()
        return self._row_to_notebook(row) if row else None

    def upsert_notebook(self, metadata: NotebookMetadata) -> NotebookMetadata:
        metadata.updated_at = datetime.now(timezone.utc)
        if metadata.created_at is None:
            metadata.created_at = metadata.updated_at

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notebooks (notebook_id, title, description, source_count, chunk_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(notebook_id) DO UPDATE SET
                    title=excluded.title,
                    description=excluded.description,
                    source_count=excluded.source_count,
                    chunk_count=excluded.chunk_count,
                    updated_at=excluded.updated_at
                """,
                (
                    metadata.notebook_id,
                    metadata.title,
                    metadata.description,
                    metadata.source_count,
                    metadata.chunk_count,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                ),
            )
        return metadata

    def start_ingestion(self, request: NotebookIngestionRequest) -> IngestionJobStatus:
        notebook_id = request.notebook_id or uuid.uuid4().hex
        notebook = self.get_notebook(notebook_id)
        now = datetime.now(timezone.utc)

        if notebook is None:
            notebook = NotebookMetadata(
                notebook_id=notebook_id,
                title=request.title or Path(request.path).stem or notebook_id,
                description=request.description or f"Imported from {request.path}",
                source_count=0,
                chunk_count=0,
                created_at=now,
                updated_at=now,
            )
            self.upsert_notebook(notebook)

        job_id = uuid.uuid4().hex
        job = IngestionJobStatus(
            job_id=job_id,
            notebook_id=notebook_id,
            status="running",
            message="Ingestion started",
            started_at=now,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, notebook_id, status, message, started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    notebook_id,
                    job.status,
                    job.message,
                    job.started_at.isoformat(),
                ),
            )
        return job

    def complete_ingestion(
        self,
        job_id: str,
        message: str,
        source_count: int,
        chunk_count: int,
    ) -> IngestionJobStatus:
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown job_id {job_id}")
            notebook_id = row["notebook_id"]
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, message = ?, completed_at = ?, documents_processed = ?, chunks_indexed = ?
                WHERE job_id = ?
                """,
                ("completed", message, now.isoformat(), source_count, chunk_count, job_id),
            )
            conn.execute(
                """
                UPDATE notebooks
                SET source_count = ?, chunk_count = ?, updated_at = ?
                WHERE notebook_id = ?
                """,
                (source_count, chunk_count, now.isoformat(), notebook_id),
            )
        return IngestionJobStatus(
            job_id=job_id,
            notebook_id=notebook_id,
            status="completed",
            message=message,
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=now,
            documents_processed=source_count,
            chunks_indexed=chunk_count,
        )

    def fail_ingestion(self, job_id: str, message: str) -> IngestionJobStatus:
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown job_id {job_id}")
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, message = ?, completed_at = ?
                WHERE job_id = ?
                """,
                ("failed", message, now.isoformat(), job_id),
            )
        return IngestionJobStatus(
            job_id=row["job_id"],
            notebook_id=row["notebook_id"],
            status="failed",
            message=message,
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=now,
        )

    def list_jobs(self) -> list[IngestionJobStatus]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_jobs ORDER BY started_at DESC LIMIT 100"
            ).fetchall()
        return [
            IngestionJobStatus(
                job_id=row["job_id"],
                notebook_id=row["notebook_id"],
                status=row["status"],
                message=row["message"],
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                documents_processed=row["documents_processed"],
                chunks_indexed=row["chunks_indexed"],
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_notebook(row: sqlite3.Row | None) -> NotebookMetadata | None:
        if row is None:
            return None
        return NotebookMetadata(
            notebook_id=row["notebook_id"],
            title=row["title"],
            description=row["description"],
            source_count=row["source_count"],
            chunk_count=row["chunk_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
