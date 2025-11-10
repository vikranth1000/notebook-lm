from pathlib import Path

from notebooklm_backend.config import reset_settings_cache
from fastapi.testclient import TestClient

from notebooklm_backend.services.embeddings import create_embedding_backend
from notebooklm_backend.services.ingestion import IngestionService
from notebooklm_backend.services.rag import RAGService
from notebooklm_backend.services.vector_store import create_vector_store
from notebooklm_backend.config import AppConfig


def build_settings(tmp_path: Path) -> AppConfig:
    config = AppConfig.model_validate(
        {
            "workspace_root": tmp_path / "workspace",
            "data_dir": tmp_path / "workspace" / "data",
            "models_dir": tmp_path / "workspace" / "models",
            "index_dir": tmp_path / "workspace" / "indexes",
            "cache_dir": tmp_path / "workspace" / "cache",
            "embedding_backend": "hash",
            "llm_provider": "none",
        }
    )
    config.ensure_directories()
    return config


def test_ingestion_and_rag(tmp_path):
    settings = build_settings(tmp_path)

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    file_path = sample_dir / "notes.txt"
    file_path.write_text("Alpine plants use antifreeze proteins to avoid ice damage.", encoding="utf-8")

    embedding_backend = create_embedding_backend(settings)
    vector_store = create_vector_store(settings, embedding_backend)

    ingestion = IngestionService(settings, vector_store)
    result = ingestion.ingest_path("demo-notebook", sample_dir)

    assert result.documents_processed == 1
    assert result.chunks_indexed >= 1

    rag_service = RAGService(settings, vector_store)
    response = rag_service.query("demo-notebook", "How do alpine plants avoid ice damage?", top_k=2)

    assert "antifreeze" in response.answer.lower()
    assert len(response.sources) >= 1


def test_app_factory_uses_env(monkeypatch, tmp_path):
    workspace = tmp_path / "app"
    (workspace / "data").mkdir(parents=True)
    (workspace / "models").mkdir()
    (workspace / "indexes").mkdir()
    (workspace / "cache").mkdir()

    monkeypatch.setenv("NOTEBOOKLM_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("NOTEBOOKLM_DATA_DIR", str(workspace / "data"))
    monkeypatch.setenv("NOTEBOOKLM_MODELS_DIR", str(workspace / "models"))
    monkeypatch.setenv("NOTEBOOKLM_INDEX_DIR", str(workspace / "indexes"))
    monkeypatch.setenv("NOTEBOOKLM_CACHE_DIR", str(workspace / "cache"))
    monkeypatch.setenv("NOTEBOOKLM_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("NOTEBOOKLM_LLM_PROVIDER", "none")

    reset_settings_cache()

    from notebooklm_backend.app import create_app

    app = create_app()
    client = TestClient(app)

    sample_file = tmp_path / "alpine.txt"
    sample_file.write_text("Plants use cushion morphology to retain heat.", encoding="utf-8")

    response = client.post(
        "/api/notebooks/ingest",
        json={"path": str(sample_file), "recursive": False},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] in {"running", "completed"}

    reset_settings_cache()

