from fastapi import APIRouter

router = APIRouter(tags=["tracing"])


@router.get("/trace/{decision_id}")
async def get_trace(decision_id: str) -> dict:
    return {"decision_id": decision_id, "messages": []}
