from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["scenarios"])


SCENARIOS = [
    {
        "id": "stable",
        "name": "Стабильный режим",
        "description": "Свежий калибровочный ЛИМС серы, риски ниже порога. Система молчит.",
        "timestamp": "2024-08-28T02:00:00",
    },
    {
        "id": "sulfur_risk",
        "name": "Риск роста серы",
        "description": "Риск превышения спеки серы. Система советует управляющие воздействия.",
        "timestamp": "2023-04-29T07:50:00",
    },
    {
        "id": "stale_data",
        "name": "Аномальные данные",
        "description": "Аномальное состояние процесса (score≈1.0). Система отказывается от рекомендации.",
        "timestamp": "2024-03-18T10:10:00",
    },
]


@router.get("/scenarios")
async def list_scenarios() -> list[dict]:
    return SCENARIOS


@router.post("/scenarios/{scenario_id}/run")
async def run_scenario(scenario_id: str) -> dict:
    scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
    if not scenario:
        return {"error": "unknown scenario"}
    return {
        "scenario_id": scenario_id,
        "timestamp": scenario["timestamp"],
        "decision_id": "stub",
    }
