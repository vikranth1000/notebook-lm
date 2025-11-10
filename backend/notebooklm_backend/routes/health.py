from __future__ import annotations

from fastapi import APIRouter

from ..services.system import system_probe

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthcheck() -> dict[str, str]:
    probe = system_probe()
    return {"status": "ok", "detail": probe}

