from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["scenarios"])


SCENARIOS = [
    {
        "id": "stable",
        "name": "Стабильный режим",
        "description": "Сера 6.5 ppm, за 6 часов диапазон 4.7-6.8. ЛИМС свежий (17 ч). Система молчит.",
        "timestamp": "2025-01-13T03:00:00",
    },
    {
        "id": "sulfur_risk",
        "name": "Риск роста серы",
        "description": "Сера 10.95 ppm, только что пересекла лимит 10. Волатильный режим. Система советует.",
        "timestamp": "2023-01-06T05:10:00",
    },
    {
        "id": "stale_data",
        "name": "Устаревшие данные",
        "description": "ЛИМС старше 48 часов, ПАК серы застыл (одно значение 6 часов). Система отказывается.",
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
