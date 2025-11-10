from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import AppConfig
from .llm import LLMBackend
from .rag import RAGService


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatService:
    def __init__(self, backend: LLMBackend, settings: AppConfig, rag_service: RAGService | None = None) -> None:
        self._backend = backend
        self._settings = settings
        self._rag_service = rag_service

    async def generate_reply(
        self,
        prompt: str,
        history: Iterable[ChatMessage] | None = None,
        notebook_id: str | None = None,
    ) -> str:
        # If notebook_id is provided and RAG service is available, use RAG
        if notebook_id and self._rag_service:
            try:
                # Combine history context with current question for better RAG
                full_question = prompt
                if history:
                    # Add recent conversation context to help with follow-up questions
                    recent_context = "\n".join([f"{msg.role}: {msg.content}" for msg in list(history)[-3:]])
                    full_question = f"Previous conversation:\n{recent_context}\n\nCurrent question: {prompt}"
                
                rag_result = await self._rag_service.query(notebook_id=notebook_id, question=full_question, top_k=20)
                # Use RAG answer as the response
                return rag_result.answer
            except Exception as e:
                # Log error but fall back to regular chat
                import logging
                logging.warning(f"RAG query failed: {e}")
                # Fall back to regular chat if RAG fails
                pass
        
        # Regular chat without RAG
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
