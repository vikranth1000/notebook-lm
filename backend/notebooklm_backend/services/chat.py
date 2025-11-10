from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import AppConfig
from .llm import LLMBackend


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatService:
    def __init__(self, backend: LLMBackend, settings: AppConfig) -> None:
        self._backend = backend
        self._settings = settings

    async def generate_reply(self, prompt: str, history: Iterable[ChatMessage] | None = None) -> str:
        context = self._render_prompt(prompt, history)
        return await self._backend.generate(context, self._settings.llm_max_tokens)

    @property
    def provider(self) -> str:
        return self._settings.llm_provider

    def _render_prompt(self, prompt: str, history: Iterable[ChatMessage] | None) -> str:
        if not history:
            return prompt

        lines: list[str] = []
        for message in history:
            role = message.role.capitalize()
            lines.append(f"{role}: {message.content}")
        lines.append(f"User: {prompt}")
        lines.append("Assistant:")
        return "\n".join(lines)
