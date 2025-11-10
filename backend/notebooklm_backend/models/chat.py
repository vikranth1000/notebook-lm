from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="who produced the message (user or assistant)")
    content: str


class ChatRequest(BaseModel):
    prompt: str
    history: List[ChatMessage] | None = None


class ChatResponse(BaseModel):
    reply: str
    provider: str
