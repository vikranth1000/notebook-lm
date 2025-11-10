from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from ..config import AppConfig

try:
    from llama_cpp import Llama  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore


class LLMBackend(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> str:
        ...


@dataclass
class LlamaCppBackend:
    model_path: str
    n_ctx: int
    _llm: Llama | None = None

    def _ensure(self) -> Llama:
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed")
        if self._llm is None:
            self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx)
        return self._llm

    def generate(self, prompt: str, max_tokens: int) -> str:
        llm = self._ensure()
        out = llm(prompt, max_tokens=max_tokens, stop=["###"])
        return out["choices"][0]["text"].strip()


@dataclass
class OllamaBackend:
    base_url: str
    model: str
    client: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        return self.client or httpx.Client(timeout=60)

    def generate(self, prompt: str, max_tokens: int) -> str:
        # Use /api/generate non-streaming
        payload = {"model": self.model, "prompt": prompt, "options": {"num_predict": max_tokens}}
        resp = self._client().post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


@dataclass
class DummyBackend:
    def generate(self, prompt: str, max_tokens: int) -> str:
        return "Offline model placeholder response. Configure Ollama or llama.cpp to enable generation."


def create_llm_backend(settings: AppConfig) -> LLMBackend:
    provider = getattr(settings, "llm_provider", "none")
    if provider == "ollama":
        base = getattr(settings, "ollama_base_url", "http://127.0.0.1:11434")
        model = getattr(settings, "ollama_model", "mistral")
        return OllamaBackend(base_url=base, model=model)
    if provider == "llama-cpp" and settings.llm_model_path:
        return LlamaCppBackend(model_path=str(settings.llm_model_path), n_ctx=settings.llm_context_window)
    return DummyBackend()

