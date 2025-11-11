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

    llm_provider: Literal["none", "ollama", "llama-cpp"] = "ollama"  # Default to ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"  # Match the model you're using
    llm_model_path: Path | None = None
    llm_context_window: int = 2048
    llm_max_tokens: int = 2048

    # Framework integration toggles
    use_langchain_splitter: bool = True
    use_llamaindex_rag: bool = True  # Re-enabled - will use improved integration

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NOTEBOOKLM_", extra="ignore")

    def ensure_directories(self) -> None:
        for directory in (self.workspace_root, self.data_dir, self.models_dir, self.index_dir, self.cache_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return AppConfig()


def reset_settings_cache() -> None:
    get_settings.cache_clear()

