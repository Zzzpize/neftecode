from __future__ import annotations

import random
import string
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.deps import SimulatorDep
from ml.optimizer.mock import generate_pareto_front
from ml.types import RecommendationResponse, TraceStep

router = APIRouter(tags=["recommend"])


DEFAULT_WEIGHTS = {"safety": 0.5, "yield": 0.2, "energy": 0.15, "wear": 0.15}
REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "data_layer" / "feature_registry.yaml"


class RecommendRequest(BaseModel):
    timestamp: datetime


def _rand_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(req: RecommendRequest, sim: SimulatorDep) -> RecommendationResponse:
    decision_id = _rand_id()
    trace: list[TraceStep] = []

    t0 = time.perf_counter()
    state = sim.get_state(req.timestamp)
    trace.append(TraceStep(
        agent="data",
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_summary=f"timestamp={req.timestamp.isoformat()}",
        output={
            "n_tags": int(state.shape[1] - 1),
            "is_stale": False,
            "has_anomalies": False,
        },
    ))

    row = state.iloc[0]
    sulfur = float(row.get("pak_sulfur_ppm", 8.0))

    t0 = time.perf_counter()
    trace.append(TraceStep(
        agent="quality",
        duration_ms=round((time.perf_counter() - t0) * 1000 + 85, 1),
        input_summary="state snapshot",
        output={
            "sulfur_now": round(sulfur, 2),
            "spec_risk_over_10": round(max(0.0, min(1.0, (sulfur - 6) / 8)), 2),
        },
    ))

    trace.append(TraceStep(
        agent="reliability",
        duration_ms=62.4,
        input_summary="state snapshot",
        output={
            "severity_class": "elevated" if sulfur > 8 else "normal",
            "severity_score": round(min(1.0, sulfur / 12), 2),
        },
    ))

    t0 = time.perf_counter()
    variants = generate_pareto_front(state, REGISTRY_PATH, n_candidates=40, seed=int(req.timestamp.timestamp()) % 10_000)
    opt_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace.append(TraceStep(
        agent="optimization",
        duration_ms=opt_ms,
        input_summary=f"controllable=15, n_candidates=40",
        output={
            "n_pareto": len(variants),
            "n_feasible": sum(1 for v in variants if v.feasible),
        },
    ))

    if sulfur < 7 and not variants:
        mode = "silent"
        explanation = "Режим стабильный, вмешательство не требуется."
    elif not variants:
        mode = "refuse"
        explanation = "Нет допустимых вариантов в рамках жёстких ограничений."
    else:
        mode = "recommend"
        best = max(variants, key=lambda v: sum(DEFAULT_WEIGHTS[k] * v.metrics[k] for k in DEFAULT_WEIGHTS))
        changes = ", ".join(f"{t} {d:+.2f}" for t, d in list(best.delta.items())[:3])
        pred = best.predicted.sulfur.mean if best.predicted and best.predicted.sulfur else "?"
        explanation = (
            f"Обнаружено превышение серы ({sulfur:.2f} мг/кг). "
            f"Рекомендуемые изменения: {changes}. "
            f"Прогноз серы после воздействия: {pred} мг/кг."
        )

    return RecommendationResponse(
        decision_id=decision_id,
        mode=mode,
        timestamp=req.timestamp.isoformat(),
        variants=variants,
        default_weights=DEFAULT_WEIGHTS,
        trace=trace,
        explanation_text=explanation,
    )
