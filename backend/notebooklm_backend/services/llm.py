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
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise ValueError(f"Ollama error: {data['error']}")
                return data.get("response", "").strip()
        except httpx.HTTPStatusError as e:
            error_text = e.response.text if e.response else str(e)
            raise ValueError(f"Ollama HTTP error: {e.response.status_code} - {error_text}")
        except httpx.RequestError as e:
            raise ValueError(f"Cannot connect to Ollama at {self.base_url}: {str(e)}")


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
