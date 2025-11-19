from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import AsyncIterator, Protocol

import httpx

from ..config import AppConfig

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Llama = None

try:
    import onnxruntime_genai as og  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    og = None


class LLMBackend(Protocol):
    async def generate(self, prompt: str, max_tokens: int) -> str:
        ...
    async def stream_generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
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

    async def stream_generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        # Ollama streams JSON per line
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if data.get("error"):
                            raise ValueError(f"Ollama error: {data['error']}")
                        if data.get("done"):
                            break
                        chunk = data.get("response")
                        if chunk:
                            yield chunk
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
    
    async def stream_generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        yield await self.generate(prompt, max_tokens)


@dataclass
class LlamaCppBackend:
    model_path: Path
    context_window: int = 4096
    gpu_layers: int = 35
    _llama: Llama | None = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> Llama:
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is required for llama-cpp backend. "
                "Install via pip install llama-cpp-python"
            )
        if self._llama is None:
            self._llama = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_window,
                n_gpu_layers=self.gpu_layers,
            )
        return self._llama

    async def generate(self, prompt: str, max_tokens: int) -> str:
        llama = self._ensure_model()
        completion = llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=False,
        )
        return completion["choices"][0]["text"].strip()

    async def stream_generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        llama = self._ensure_model()
        for chunk in llama.create_completion(prompt=prompt, max_tokens=max_tokens, stream=True):
            delta = chunk["choices"][0].get("text")
            if delta:
                yield delta


@dataclass
class OnnxRuntimeBackend:
    model_path: Path
    execution_provider: str = "cpu"
    context_window: int = 4096

    @cached_property
    def _model(self):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is required for the ONNX backend. "
                "Install via pip install onnxruntime-genai"
            )
        return og.Model(str(self.model_path))

    @cached_property
    def _tokenizer(self):
        return og.Tokenizer(self._model)

    def _create_session(self):
        chat = og.ChatSession(self._model, self._tokenizer)
        chat.set_preset("balanced")
        chat.set_max_output_tokens(self.context_window)
        chat.config.search_options.provider = self.execution_provider
        return chat

    async def generate(self, prompt: str, max_tokens: int) -> str:
        session = self._create_session()
        response = session.generate(prompt, max_tokens)
        return response.strip()

    async def stream_generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        session = self._create_session()
        stream = session.generate_stream(prompt, max_tokens)
        for token in stream:
            if token:
                yield token


def create_llm_backend(settings: AppConfig) -> LLMBackend:
    if settings.llm_provider == "ollama":
        return OllamaBackend(base_url=settings.ollama_base_url, model=settings.ollama_model)
    if settings.llm_provider == "llama-cpp":
        if not settings.llm_model_path:
            raise ValueError("Set NOTEBOOKLM_LLM_MODEL_PATH for llama-cpp provider")
        return LlamaCppBackend(
            model_path=settings.llm_model_path,
            context_window=settings.llm_context_window,
        )
    if settings.llm_provider == "onnx":
        if not settings.onnx_model_path:
            raise ValueError("Set NOTEBOOKLM_ONNX_MODEL_PATH for onnx provider")
        return OnnxRuntimeBackend(
            model_path=settings.onnx_model_path,
            execution_provider=settings.onnx_execution_provider,
            context_window=settings.llm_context_window,
        )
    return DummyBackend()
