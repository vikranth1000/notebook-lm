from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from ..config import AppConfig


class LLMBackend(Protocol):
    async def generate(self, prompt: str, max_tokens: int) -> str:
        ...


@dataclass
class OllamaBackend:
    base_url: str
    model: str

    async def generate(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("response", "").strip()


@dataclass
class DummyBackend:
    async def generate(self, prompt: str, max_tokens: int) -> str:
        return (
            "Offline model placeholder response. Configure NOTEBOOKLM_LLM_PROVIDER=ollama "
            "and ensure Ollama is running to enable real answers."
        )


def create_llm_backend(settings: AppConfig) -> LLMBackend:
    if settings.llm_provider == "ollama":
        return OllamaBackend(base_url=settings.ollama_base_url, model=settings.ollama_model)
    return DummyBackend()
