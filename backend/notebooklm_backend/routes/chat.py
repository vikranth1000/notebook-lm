from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.chat import ChatRequest, ChatResponse
from ..services.chat import ChatService, ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: Request, payload: ChatRequest) -> ChatResponse:
    service: ChatService = request.app.state.chat_service
    history = None
    if payload.history:
        history = [ChatMessage(role=msg.role, content=msg.content) for msg in payload.history]
    reply = await service.generate_reply(prompt=payload.prompt, history=history)
    return ChatResponse(reply=reply, provider=service.provider)
