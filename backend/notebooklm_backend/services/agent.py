from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..config import AppConfig
from .llm import create_llm_backend


class AgentService:
    """
    Lightweight agentic workflow helper that keeps short-term memory summaries per notebook.
    """

    def __init__(self, settings: AppConfig) -> None:
        self.settings = settings
        self._backend = create_llm_backend(settings)
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
                CREATE TABLE IF NOT EXISTS agent_memory (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notebook_id TEXT,
                    created_at TEXT NOT NULL,
                    summary TEXT NOT NULL
                )
                """
            )

    def memories(self, notebook_id: str | None) -> list[str]:
        query = "SELECT summary FROM agent_memory"
        params: tuple = ()
        if notebook_id:
            query += " WHERE notebook_id = ?"
            params = (notebook_id,)
        query += " ORDER BY memory_id DESC LIMIT 5"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [row["summary"] for row in rows]

    async def plan(self, goal: str, notebook_id: str | None = None) -> str:
        memories_text = "\n".join(self.memories(notebook_id)) or "None recorded yet."
        prompt = (
            "You are an agent that helps organise work on local notebooks.\n"
            "Given the user's goal, produce a numbered plan with concrete steps.\n"
            "Apply the following context:\n"
            f"Goal: {goal}\n"
            f"Notebook: {notebook_id or 'N/A'}\n"
            f"Recent memory:\n{memories_text}\n\n"
            "Return concise steps (<=6) with actionable verbs."
        )
        reply = await self._backend.generate(prompt, max_tokens=400)
        self._persist_memory(reply, notebook_id)
        return reply.strip()

    def _persist_memory(self, summary: str, notebook_id: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_memory (notebook_id, created_at, summary) VALUES (?, ?, ?)",
                (
                    notebook_id,
                    datetime.now(timezone.utc).isoformat(),
                    summary.strip(),
                ),
            )
