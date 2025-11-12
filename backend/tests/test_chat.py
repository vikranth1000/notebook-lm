from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from notebooklm_backend.app import create_app
from notebooklm_backend.config import reset_settings_cache


@pytest.fixture
def client(monkeypatch):
    """Create a TestClient with deterministic config and no Ollama requirement."""
    monkeypatch.setenv("NOTEBOOKLM_LLM_PROVIDER", "none")
    monkeypatch.delenv("NOTEBOOKLM_OLLAMA_MODEL", raising=False)
    reset_settings_cache()
    app = create_app()
    return TestClient(app)


def test_chat_endpoint_requires_prompt(client):
    response = client.post("/api/chat/", json={})
    assert response.status_code == 422


def test_chat_endpoint_with_dummy_backend(client):
    response = client.post("/api/chat/", json={"prompt": "Hello, how are you?"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "provider" in data
    assert data["provider"] == "none"


def test_chat_endpoint_with_history(client):
    response = client.post(
        "/api/chat/",
        json={
            "prompt": "What did I just say?",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data


@pytest.mark.skipif(
    os.getenv("NOTEBOOKLM_LLM_PROVIDER") != "ollama",
    reason="Requires Ollama to be configured",
)
def test_chat_endpoint_with_ollama(client):
    response = client.post("/api/chat/", json={"prompt": "Say hello in one word"})
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "ollama"
    assert len(data["reply"]) > 0
os.environ.setdefault("NOTEBOOKLM_LLM_PROVIDER", "none")
