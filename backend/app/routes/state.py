from datetime import datetime

from fastapi import APIRouter

from app.deps import SimulatorDep

router = APIRouter(tags=["data"])


@router.get("/state")
async def get_state(ts: datetime, sim: SimulatorDep) -> dict:
    row = sim.get_state(ts).iloc[0]

    tags: dict[str, float] = {}
    lims_age_hours: dict[str, float] = {}
    lims_values: dict[str, float] = {}

    for col, val in row.items():
        if col == "date":
            continue
        if col.startswith("lims__"):
            if col.endswith("_age_h"):
                key = col[len("lims__"):-len("_age_h")]
                if val == val:
                    lims_age_hours[key] = float(val)
            else:
                key = col[len("lims__"):]
                if val == val:
                    lims_values[key] = float(val)
        else:
            if val == val:
                tags[col] = float(val)

    return {
        "timestamp": row["date"].isoformat(),
        "tags": tags,
        "lims_values": lims_values,
        "lims_age_hours": lims_age_hours,
    }


@router.get("/state/range")
async def get_range(sim: SimulatorDep) -> dict:
    return {
        "min_date": sim.min_date.isoformat(),
        "max_date": sim.max_date.isoformat(),
    }
