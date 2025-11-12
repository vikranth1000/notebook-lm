from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..services.speech import SpeechService, SpeechUnavailableError

router = APIRouter(prefix="/speech", tags=["speech"])


def _get_service(request: Request) -> SpeechService:
    service: SpeechService | None = getattr(request.app.state, "speech_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Speech services unavailable")
    return service


@router.post("/transcribe")
async def transcribe_audio(request: Request, audio: UploadFile = File(...)) -> JSONResponse:
    service = _get_service(request)
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or 'audio').suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = Path(tmp.name)
    try:
        transcript = await service.transcribe(tmp_path)
        return JSONResponse({"transcript": transcript})
    except SpeechUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/speak")
async def speak_text(request: Request, payload: dict[str, str]) -> FileResponse:
    service = _get_service(request)
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    try:
        audio_path = await service.synthesize(text)
    except SpeechUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return FileResponse(
        path=str(audio_path),
        filename=audio_path.name,
        media_type="audio/wav",
    )
