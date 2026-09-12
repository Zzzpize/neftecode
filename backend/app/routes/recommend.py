from __future__ import annotations

import logging
import random
import string
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.ai_adapter import to_recommendation_response
from app.deps import OrchestratorDep, SimulatorDep, TracerDep
from app.ml_adapter import artifacts_ready, real_recommend
from app.schemas import RecommendationResponse, TraceStep
from ml.optimizer.mock import generate_pareto_front

router = APIRouter(tags=["recommend"])
log = logging.getLogger(__name__)


DEFAULT_WEIGHTS = {"safety": 0.5, "yield": 0.2, "energy": 0.15, "wear": 0.15}
REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "data_layer" / "feature_registry.yaml"


class RecommendRequest(BaseModel):
    timestamp: datetime
    weights: dict[str, float] | None = None


def _rand_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    req: RecommendRequest,
    sim: SimulatorDep,
    orch: OrchestratorDep,
    tracer: TracerDep,
) -> RecommendationResponse:
    try:
        out = await orch.decide(req.timestamp, weights=req.weights)
        return to_recommendation_response(out, req.timestamp, sim, tracer)
    except Exception:
        log.exception("orchestrator.decide failed, falling back to direct ML/mock path")
        if artifacts_ready():
            return real_recommend(req.timestamp, sim, weights=req.weights)
        return _mock_recommend(req.timestamp, sim, weights=req.weights)


def _mock_recommend(timestamp: datetime, sim, weights: dict[str, float] | None = None) -> RecommendationResponse:
    decision_id = _rand_id()
    weights = dict(weights or DEFAULT_WEIGHTS)
    trace: list[TraceStep] = []

    t0 = time.perf_counter()
    state = sim.get_state(timestamp)
    trace.append(TraceStep(
        agent="data",
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_summary=f"timestamp={timestamp.isoformat()}",
        output={"n_tags": int(state.shape[1] - 1), "is_stale": False, "has_anomalies": False},
    ))

    row = state.iloc[0]
    sulfur = float(row.get("pak_sulfur_ppm", 8.0))

    trace.append(TraceStep(
        agent="quality",
        duration_ms=85.0,
        input_summary="state snapshot",
        output={"sulfur_now": round(sulfur, 2)},
    ))
    trace.append(TraceStep(
        agent="reliability",
        duration_ms=62.4,
        input_summary="state snapshot",
        output={"severity_class": "elevated" if sulfur > 8 else "normal"},
    ))

    t0 = time.perf_counter()
    variants = generate_pareto_front(state, REGISTRY_PATH, seed=int(timestamp.timestamp()) % 10_000)
    trace.append(TraceStep(
        agent="optimization",
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_summary="mock",
        output={"n_pareto": len(variants)},
    ))

    if sulfur < 7 and not variants:
        return RecommendationResponse(
            decision_id=decision_id, mode="silent", timestamp=timestamp.isoformat(),
            trace=trace, default_weights=weights,
            explanation_text="Режим стабильный, вмешательство не требуется.",
        )
    if not variants:
        return RecommendationResponse(
            decision_id=decision_id, mode="refuse", timestamp=timestamp.isoformat(),
            trace=trace, default_weights=weights,
            explanation_text="Нет допустимых вариантов в рамках жёстких ограничений.",
        )

    best = max(variants, key=lambda v: sum(w * v.metrics.get(k, 0.0) for k, w in weights.items()))
    changes = ", ".join(f"{t} {d:+.2f}" for t, d in list(best.delta.items())[:3])
    pred = best.predicted.sulfur.mean if best.predicted and best.predicted.sulfur else "?"
    explanation = (
        f"[MOCK] Сера {sulfur:.2f} мг/кг. Изменения: {changes}. "
        f"Прогноз серы: {pred} мг/кг."
    )
    return RecommendationResponse(
        decision_id=decision_id, mode="recommend", timestamp=timestamp.isoformat(),
        variants=variants, default_weights=weights, trace=trace,
        explanation_text=explanation,
    )
