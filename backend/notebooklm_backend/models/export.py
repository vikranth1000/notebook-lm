from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .chat import ChatMessage


class ConversationExportRequest(BaseModel):
    title: str = Field(default="Conversation")
    messages: List[ChatMessage]
    notebook_id: str | None = None
