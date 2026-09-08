from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["recommend"])


class RecommendRequest(BaseModel):
    timestamp: datetime


class RecommendResponse(BaseModel):
    decision_id: str
    mode: str
    payload: dict


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    return RecommendResponse(
        decision_id="stub",
        mode="silent",
        payload={"message": "orchestrator not wired"},
    )
