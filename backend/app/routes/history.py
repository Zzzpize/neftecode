from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["data"])


@router.get("/history")
async def get_history(tag: str, ts_from: datetime, ts_to: datetime) -> dict:
    return {"tag": tag, "points": []}
