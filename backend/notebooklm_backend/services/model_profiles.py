from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import psutil

from ..config import AppConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    max_ram_gb: float
    description: str


PROFILES: list[ModelProfile] = [
    ModelProfile(name="phi3:mini", max_ram_gb=12, description="Best for 8–12 GB machines"),
    ModelProfile(name="qwen2.5:3b", max_ram_gb=24, description="Balanced quality vs. speed"),
    ModelProfile(name="mistral", max_ram_gb=1e6, description="High-quality 7B baseline"),
]


def _system_ram_gb() -> float:
    return round(psutil.virtual_memory().total / (1024**3), 1)


def _available_ollama_models(base_url: str) -> list[str]:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        response.raise_for_status()
        data = response.json()
        return [entry.get("name", "") for entry in data.get("models", []) if entry.get("name")]
    except Exception as exc:  # pragma: no cover - network/daemon optional
        LOGGER.debug("Unable to list Ollama models: %s", exc)
        return []


def resolve_ollama_model(settings: AppConfig) -> None:
    """Select an Ollama model automatically when NOTEBOOKLM_OLLAMA_MODEL=auto."""
    requested = (settings.ollama_model or "").strip().lower()
    if requested and requested not in {"auto", "automatic"}:
        settings.resolved_ollama_model = settings.ollama_model
        settings.model_selection_reason = "env override"
        return

    ram_gb = _system_ram_gb()
    selected = PROFILES[-1]
    for profile in PROFILES:
        if ram_gb <= profile.max_ram_gb:
            selected = profile
            break

    installed = _available_ollama_models(settings.ollama_base_url)
    if installed and selected.name not in installed:
        LOGGER.info("Ollama model %s missing; using %s instead", selected.name, installed[0])
        selected_name = installed[0]
        reason = f"auto (installed model {selected_name})"
    else:
        selected_name = selected.name
        reason = f"auto based on {ram_gb} GB RAM ({selected.description})"
        if installed and selected_name not in installed:
            reason += " – pull model via 'ollama pull {selected_name}'"

    settings.ollama_model = selected_name
    settings.resolved_ollama_model = selected_name
    settings.model_selection_reason = reason
