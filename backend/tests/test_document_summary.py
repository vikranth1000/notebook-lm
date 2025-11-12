from __future__ import annotations

import pytest

from notebooklm_backend.config import AppConfig
from notebooklm_backend.services.document_summary import DocumentSummaryService


@pytest.mark.asyncio
async def test_document_summary_fallback_without_llm(tmp_path):
    settings = AppConfig(
        workspace_root=tmp_path,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        llm_provider="none",
    )
    service = DocumentSummaryService(settings)
    summary = await service.generate_summary(
        text="This resume describes Jane Doe's experience in AI research.",
        source_path=str(tmp_path / "resume.pdf"),
    )
    assert summary.summary
    assert "resume" in summary.summary.lower()
