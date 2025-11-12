from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.agent import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentPlanRequest(BaseModel):
    goal: str = Field(..., description="Goal or problem to solve")
    notebook_id: str | None = None


class AgentPlanResponse(BaseModel):
    plan: str


@router.post("/plan", response_model=AgentPlanResponse)
async def create_plan(request: Request, payload: AgentPlanRequest) -> AgentPlanResponse:
    if not payload.goal.strip():
        raise HTTPException(status_code=400, detail="Goal is required")
    service: AgentService = request.app.state.agent_service
    plan = await service.plan(payload.goal, payload.notebook_id)
    return AgentPlanResponse(plan=plan)
