from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["data"])


@router.get("/state")
async def get_state(ts: datetime) -> dict:
    return {"timestamp": ts.isoformat(), "tags": {}, "lims_age_hours": {}}
