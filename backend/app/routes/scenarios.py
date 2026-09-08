from fastapi import APIRouter

router = APIRouter(tags=["scenarios"])


@router.get("/scenarios")
async def list_scenarios() -> list[dict]:
    return [
        {"id": "stable", "name": "Стабильный режим", "description": "система молчит"},
        {"id": "sulfur_risk", "name": "Риск роста серы", "description": "система советует"},
        {"id": "stale_data", "name": "Устаревшие данные", "description": "система отказывается"},
    ]


@router.post("/scenarios/{scenario_id}/run")
async def run_scenario(scenario_id: str) -> dict:
    return {"scenario_id": scenario_id, "decision_id": "stub"}
