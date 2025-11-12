from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    workspace_root: Path = Path.home() / "NotebookLM"
    data_dir: Path = Path.home() / "NotebookLM" / "data"
    models_dir: Path = Path.home() / "NotebookLM" / "models"
    index_dir: Path = Path.home() / "NotebookLM" / "indexes"
    cache_dir: Path = Path.home() / "NotebookLM" / "cache"
    enable_audio: bool = True

    embedding_backend: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"

    llm_provider: Literal["none", "ollama", "llama-cpp", "onnx"] = "ollama"  # Default to ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "auto"  # Auto-select lightweight model by default
    llm_model_path: Path | None = None  # Used by llama-cpp
    onnx_model_path: Path | None = None
    onnx_execution_provider: Literal["cpu", "cuda", "metal"] = "cpu"
    llm_context_window: int = 2048
    llm_max_tokens: int = 2048

    # Framework integration toggles
    use_langchain_splitter: bool = True
    use_llamaindex_rag: bool = True  # Re-enabled - will use improved integration
    enable_speech_stt: bool = False
    enable_speech_tts: bool = False

    # Runtime bookkeeping (not exposed via env)
    resolved_ollama_model: str | None = None
    model_selection_reason: str | None = None

    # RAG controls
    rag_doc_select_k: int = 2  # how many documents to select in stage 1
    rag_top_k: int = 20        # total chunks to consider in stage 2
    rag_history_turns: int = 1 # how many turns of history to include in RAG questions

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NOTEBOOKLM_", extra="ignore")

    def ensure_directories(self) -> None:
        for directory in (self.workspace_root, self.data_dir, self.models_dir, self.index_dir, self.cache_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return AppConfig()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
