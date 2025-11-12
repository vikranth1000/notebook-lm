from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models.chat import ChatRequest, ChatResponse
from ..services.chat import ChatService, ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: Request, payload: ChatRequest) -> ChatResponse:
    service: ChatService = request.app.state.chat_service
    history = None
    if payload.history:
        history = [ChatMessage(role=msg.role, content=msg.content) for msg in payload.history]
    try:
        result = await service.generate_reply(
            prompt=payload.prompt,
            history=history,
            notebook_id=payload.notebook_id,
        )
        return ChatResponse(reply=result.reply, provider=service.provider, metrics=result.metrics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/stream")
async def chat_stream_endpoint(request: Request, payload: ChatRequest) -> StreamingResponse:
    service: ChatService = request.app.state.chat_service
    history = None
    if payload.history:
        history = [ChatMessage(role=msg.role, content=msg.content) for msg in payload.history]

    async def event_generator():
        try:
            async for event in service.stream_reply(
                prompt=payload.prompt,
                history=history,
                notebook_id=payload.notebook_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
