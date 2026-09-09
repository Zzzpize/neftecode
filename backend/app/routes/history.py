from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.deps import SimulatorDep

router = APIRouter(tags=["data"])


@router.get("/history")
async def get_history(tag: str, ts_from: datetime, ts_to: datetime, sim: SimulatorDep) -> dict:
    try:
        df = sim.get_history(tag, ts_from, ts_to)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    points = [
        {"t": row["date"].isoformat(), "v": None if val != val else float(val)}
        for row in df.to_dict("records")
        for val in [row[tag]]
    ]
    return {"tag": tag, "points": points}
