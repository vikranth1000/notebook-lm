from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    llm_provider: Literal["none", "ollama"] = "none"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "mistral"
    llm_max_tokens: int = 512

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NOTEBOOKLM_", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return AppConfig()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
