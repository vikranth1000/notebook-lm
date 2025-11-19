from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, List

from ..config import AppConfig
from .llm import LLMBackend
from .rag import RAGService, SourceAttribution, RAGContext
from .metrics_store import MetricsStore


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResult:
    reply: str
    metrics: dict[str, float] | None = None


class ChatService:
    def __init__(
        self,
        backend: LLMBackend,
        settings: AppConfig,
        rag_service: RAGService | None = None,
        metrics_store: MetricsStore | None = None,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._rag_service = rag_service
        self._metrics = metrics_store

    async def generate_reply(
        self,
        prompt: str,
        history: Iterable[ChatMessage] | None = None,
        notebook_id: str | None = None,
    ) -> ChatResult:
        total_start = time.perf_counter()
        metrics: dict[str, float] = {}
        # If notebook_id is provided and RAG service is available, use RAG
        if notebook_id and self._rag_service:
            try:
                # Combine history context with current question for better RAG
                full_question = prompt
                if history:
                    # Limit history to the last N turns to avoid retrieval drift
                    n = max(0, self._settings.rag_history_turns)
                    if n > 0:
                        recent = list(history)[-2 * n:]
                        recent_context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent])
                        plural = "s" if n > 1 else ""
                        full_question = (
                            "Previous conversation (last "
                            f"{n} turn{plural}):\n"
                            f"{recent_context}\n\nCurrent question: {prompt}"
                        )
                
                rag_result = await self._rag_service.query(notebook_id=notebook_id, question=full_question, top_k=20)
                if rag_result.metrics:
                    metrics.update(rag_result.metrics)
                metrics.setdefault("total_ms", (time.perf_counter() - total_start) * 1000)
                self._record_metrics(prompt, notebook_id, metrics, source_count=len(rag_result.sources))
                return ChatResult(reply=rag_result.answer, metrics=metrics)
            except Exception as e:
                # Log error but fall back to regular chat
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"RAG query failed: {e}", exc_info=True)
                # Fall back to regular chat if RAG fails
                pass
        
        # Regular chat without RAG
        context = self._render_prompt(prompt, history)
        llm_start = time.perf_counter()
        reply = await self._backend.generate(context, self._settings.llm_max_tokens)
        metrics["llm_ms"] = (time.perf_counter() - llm_start) * 1000
        metrics["total_ms"] = (time.perf_counter() - total_start) * 1000
        self._record_metrics(prompt, notebook_id, metrics, source_count=None)
        return ChatResult(reply=reply, metrics=metrics)

    async def stream_reply(
        self,
        prompt: str,
        history: Iterable[ChatMessage] | None = None,
        notebook_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream tokens using SSE-friendly event payloads:
        - meta: provider + sources + initial metrics
        - token: incremental text delta
        - done: final reply and metrics
        """
        metrics: dict[str, float] = {}
        prompt_text: str | None = None
        sources: list[dict[str, Any]] = []
        rag_context: RAGContext | None = None

        if notebook_id and self._rag_service:
            try:
                full_question = prompt
                if history:
                    recent_context = "\n".join([f"{msg.role}: {msg.content}" for msg in list(history)[-3:]])
                    full_question = f"Previous conversation:\n{recent_context}\n\nCurrent question: {prompt}"
                context = await self._rag_service.prepare_prompt(
                    notebook_id=notebook_id,
                    question=full_question,
                    top_k=20,
                )
                metrics.update(context.metrics)
                prompt_text = context.prompt or None
                sources = self._format_sources(context.sources)
                rag_context = context
                if not prompt_text:
                    # No relevant docs found
                    yield {
                        "type": "meta",
                        "provider": self.provider,
                        "sources": sources,
                        "metrics": metrics or None,
                    }
                    yield {
                        "type": "done",
                        "reply": "No relevant documents found in the notebook. Make sure you have uploaded documents.",
                        "metrics": metrics or None,
                    }
                    return
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Streaming RAG prep failed: {e}", exc_info=True)
                # Fall back to plain chat
                prompt_text = None
                metrics = {}
                sources = []

        if prompt_text is None:
            prompt_text = self._render_prompt(prompt, history)

        yield {
            "type": "meta",
            "provider": self.provider,
            "sources": sources,
            "metrics": metrics or None,
        }

        aggregated: list[str] = []
        llm_start = time.perf_counter()
        stream_fn = getattr(self._backend, "stream_generate", None)
        if stream_fn:
            async for delta in stream_fn(prompt_text, self._settings.llm_max_tokens):
                if not delta:
                    continue
                aggregated.append(delta)
                yield {"type": "token", "delta": delta}
        else:
            reply = await self._backend.generate(prompt_text, self._settings.llm_max_tokens)
            aggregated.append(reply)
            yield {"type": "token", "delta": reply}

        metrics["llm_ms"] = (time.perf_counter() - llm_start) * 1000
        metrics["total_ms"] = metrics.get("total_ms", 0.0) + metrics["llm_ms"]
        final_reply = "".join(aggregated).strip()
        final_reply = "".join(aggregated).strip()
        source_count = len(rag_context.sources) if rag_context else None
        self._record_metrics(prompt, notebook_id, metrics, source_count=source_count)
        yield {
            "type": "done",
            "reply": final_reply,
            "metrics": metrics or None,
        }

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

    def _format_sources(self, sources: List[SourceAttribution]) -> list[dict[str, Any]]:
        formatted = []
        for src in sources[:5]:
            preview = src.content.strip()
            if len(preview) > 160:
                preview = preview[:157] + "..."
            formatted.append(
                {
                    "source_path": src.source_path,
                    "preview": preview,
                    "distance": src.distance,
                }
            )
        return formatted

    def _record_metrics(
        self,
        prompt: str,
        notebook_id: str | None,
        metrics: dict[str, float] | None,
        source_count: int | None,
    ) -> None:
        if not self._metrics:
            return
        try:
            self._metrics.record_chat(
                provider=self.provider,
                prompt=prompt[:2000],
                notebook_id=notebook_id,
                metrics=metrics,
                source_count=source_count,
            )
        except Exception:
            # Metrics recording failures must never block user flows
            import logging

            logging.getLogger(__name__).warning("Failed to record chat metrics", exc_info=True)
