from __future__ import annotations

from notebooklm_backend.config import AppConfig
from notebooklm_backend.models.notebook import NotebookIngestionRequest
from notebooklm_backend.services.notebook_store import NotebookStore


def test_notebook_store_persistence(tmp_path):
    settings = AppConfig(
        workspace_root=tmp_path,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    )
    store = NotebookStore(settings)
    request = NotebookIngestionRequest(path="sample.md", title="Sample Notebook")
    job = store.start_ingestion(request)
    assert job.status == "running"
    assert store.get_notebook(job.notebook_id) is not None

    completed = store.complete_ingestion(job.job_id, "done", source_count=2, chunk_count=10)
    assert completed.status == "completed"
    notebooks = store.list_notebooks()
    assert len(notebooks) == 1
    assert notebooks[0].source_count == 2
    assert notebooks[0].chunk_count == 10

    jobs = store.list_jobs()
    assert jobs[0].status == "completed"
