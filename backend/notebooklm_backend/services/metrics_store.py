from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import AppConfig
from ..models.metrics import ChatMetricRecord, MetricsSummary


class MetricsStore:
    """
    SQLite-backed store for chat metrics and diagnostics.
    Shares the same metadata.db file as notebooks/jobs to simplify deployment.
    """

    def __init__(self, settings: AppConfig) -> None:
        self.db_path = settings.workspace_root / "metadata.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
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
                CREATE TABLE IF NOT EXISTS chat_metrics (
                    metric_id TEXT PRIMARY KEY,
                    notebook_id TEXT,
                    provider TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    total_ms REAL,
                    llm_ms REAL,
                    stage1_ms REAL,
                    retrieval_ms REAL,
                    documents_considered REAL,
                    source_count INTEGER,
                    tokens INTEGER
                )
                """
            )

    def record_chat(
        self,
        *,
        provider: str,
        prompt: str,
        notebook_id: str | None = None,
        metrics: dict[str, float] | None = None,
        source_count: int | None = None,
        tokens: int | None = None,
    ) -> ChatMetricRecord:
        metric = ChatMetricRecord(
            metric_id=uuid.uuid4().hex,
            provider=provider,
            prompt=prompt,
            notebook_id=notebook_id,
            created_at=datetime.now(timezone.utc),
            total_ms=metrics.get("total_ms") if metrics else None,
            llm_ms=metrics.get("llm_ms") if metrics else None,
            stage1_ms=metrics.get("stage1_ms") if metrics else None,
            retrieval_ms=metrics.get("retrieval_ms") if metrics else None,
            documents_considered=metrics.get("documents_considered") if metrics else None,
            source_count=source_count,
            tokens=tokens,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_metrics (
                    metric_id, notebook_id, provider, prompt, created_at,
                    total_ms, llm_ms, stage1_ms, retrieval_ms,
                    documents_considered, source_count, tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.metric_id,
                    metric.notebook_id,
                    metric.provider,
                    metric.prompt,
                    metric.created_at.isoformat(),
                    metric.total_ms,
                    metric.llm_ms,
                    metric.stage1_ms,
                    metric.retrieval_ms,
                    metric.documents_considered,
                    metric.source_count,
                    metric.tokens,
                ),
            )
        return metric

    def summary(self) -> MetricsSummary:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS cnt FROM chat_metrics").fetchone()["cnt"]
            averages = conn.execute(
                """
                SELECT AVG(total_ms) AS avg_total,
                       AVG(llm_ms) AS avg_llm,
                       AVG(retrieval_ms) AS avg_retrieval
                FROM chat_metrics
                """
            ).fetchone()
            provider_rows = conn.execute(
                "SELECT provider, COUNT(*) as cnt FROM chat_metrics GROUP BY provider"
            ).fetchall()
        provider_breakdown = {row["provider"]: row["cnt"] for row in provider_rows}
        return MetricsSummary(
            conversations=total,
            avg_total_ms=averages["avg_total"],
            avg_llm_ms=averages["avg_llm"],
            avg_retrieval_ms=averages["avg_retrieval"],
            provider_breakdown=provider_breakdown,
        )
