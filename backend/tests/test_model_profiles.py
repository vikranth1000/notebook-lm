from __future__ import annotations

import types

from notebooklm_backend.config import AppConfig
from notebooklm_backend.services.model_profiles import resolve_ollama_model


def test_resolve_ollama_model_auto(monkeypatch, tmp_path):
    # Fake psutil virtual memory
    class DummyVMem:
        total = 8 * 1024**3

    monkeypatch.setattr("notebooklm_backend.services.model_profiles.psutil.virtual_memory", lambda: DummyVMem)
    monkeypatch.setattr(
        "notebooklm_backend.services.model_profiles._available_ollama_models",
        lambda *_: ["phi3:mini", "mistral"],
    )

    settings = AppConfig(
        workspace_root=tmp_path,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        llm_provider="ollama",
        ollama_model="auto",
    )

    resolve_ollama_model(settings)
    assert settings.ollama_model == "phi3:mini"
    assert settings.resolved_ollama_model == "phi3:mini"
    assert "auto" in (settings.model_selection_reason or "")
