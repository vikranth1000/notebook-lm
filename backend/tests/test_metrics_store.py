from __future__ import annotations

from fastapi.testclient import TestClient

from notebooklm_backend.app import create_app
from notebooklm_backend.config import AppConfig
from notebooklm_backend.services.metrics_store import MetricsStore


def test_metrics_store_records_and_summarizes(tmp_path):
    settings = AppConfig(
        workspace_root=tmp_path,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    )
    store = MetricsStore(settings)
    store.record_chat(provider="ollama", prompt="Hello", notebook_id=None, metrics={"total_ms": 100.0})
    store.record_chat(provider="ollama", prompt="World", notebook_id="abc", metrics={"total_ms": 200.0})
    summary = store.summary()
    assert summary.conversations == 2
    assert summary.avg_total_ms and summary.avg_total_ms > 0
    assert summary.provider_breakdown["ollama"] == 2


def test_metrics_summary_endpoint():
    client = TestClient(create_app())
    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
