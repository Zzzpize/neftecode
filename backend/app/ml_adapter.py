from __future__ import annotations

import logging
import math
import random
import string
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from app.config import settings
from app.schemas import (
    Interval as ApiInterval,
    QualityPrediction as ApiQualityPrediction,
    RecommendationResponse,
    TraceStep,
    Variant as ApiVariant,
)

log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "ml" / "artifacts"
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data_layer" / "feature_registry.yaml"

DEFAULT_WEIGHTS = {"safety": 0.5, "yield": 0.2, "energy": 0.15, "wear": 0.15}


def artifacts_ready() -> bool:
    for name in ("quality_avt.pkl", "quality_hydro.pkl", "anomaly.pkl"):
        if not (ARTIFACTS_DIR / name).exists():
            return False
    return True


@lru_cache(maxsize=1)
def _load_bundle():
    from ml.models import (
        AnomalyDetector,
        BlendingModel,
        QualityAVTModel,
        QualityHydroModel,
    )
    from ml.optimizer import ParetoOptimizer

    log.info("loading ML artifacts from %s", ARTIFACTS_DIR)
    avt = QualityAVTModel.load(str(ARTIFACTS_DIR / "quality_avt.pkl"))
    hydro = QualityHydroModel.load(str(ARTIFACTS_DIR / "quality_hydro.pkl"))
    anom = AnomalyDetector.load(str(ARTIFACTS_DIR / "anomaly.pkl"))
    blending = BlendingModel()
    optimizer = ParetoOptimizer(avt, hydro, blending)
    return avt, hydro, anom, blending, optimizer


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _controllable_ranges_from_state(state: pd.DataFrame, envelope_bounds: dict) -> dict[str, tuple[float, float]]:
    registry = _load_registry()
    row = state.iloc[0]
    ranges: dict[str, tuple[float, float]] = {}
    excluded: list[str] = []
    for tag, cfg in registry.items():
        if not cfg.get("controllable"):
            continue
        if tag not in row.index or pd.isna(row[tag]):
            continue
        current = float(row[tag])
        low_env, high_env = envelope_bounds.get(tag, (None, None))
        if low_env is not None and (current < low_env or current > high_env):
            excluded.append(tag)
            continue
        rmin = cfg.get("range_min", current * 0.9)
        rmax = cfg.get("range_max", current * 1.1)
        ranges[tag] = (float(rmin), float(rmax))
    if excluded:
        log.info("excluding out-of-envelope tags from optimizer: %s", excluded)
    return ranges


def _envelope_bounds(avt_model, hydro_model) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for model in (avt_model, hydro_model):
        q01 = model.artifact.get("feature_q01", {})
        q99 = model.artifact.get("feature_q99", {})
        for tag in q01:
            if tag in q99:
                low = float(q01[tag])
                high = float(q99[tag])
                if math.isfinite(low) and math.isfinite(high):
                    bounds.setdefault(tag, (low, high))
    return bounds


def _rand_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def _decision_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


def _safety_from_sulfur(sulfur_ppm: float, limit: float = 10.0) -> float:
    if not math.isfinite(sulfur_ppm):
        return 0.5
    return max(0.05, min(1.0, 1.0 - sulfur_ppm / (limit + 2.0)))


def _wear_from_severity(severity: float) -> float:
    if not math.isfinite(severity):
        return 0.5
    return max(0.05, min(1.0, 1.0 - severity))


def _clip_unit(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _to_api_variant(ml_variant, current_state: pd.DataFrame) -> ApiVariant:
    row = current_state.iloc[0]

    delta: dict[str, float] = {}
    action_out: dict[str, float] = {}
    for tag, new_val in ml_variant.action.items():
        if tag in row.index and not pd.isna(row[tag]):
            d = float(new_val) - float(row[tag])
            delta[tag] = round(d, 3)
        action_out[tag] = round(float(new_val), 3)

    sulfur_iv = ml_variant.expected.predictions.get("sulfur_ppm") if ml_variant.expected else None
    t50_iv = ml_variant.expected.predictions.get("T50") if ml_variant.expected else None
    t90_iv = ml_variant.expected.predictions.get("T90") if ml_variant.expected else None
    d15_iv = ml_variant.expected.predictions.get("D15") if ml_variant.expected else None

    predicted = ApiQualityPrediction(
        sulfur=_iv(sulfur_iv),
        t50=_iv(t50_iv),
        t90=_iv(t90_iv),
        d15=_iv(d15_iv),
        confidence=(ml_variant.expected.confidence if ml_variant.expected else "medium"),
        spec_risk=(dict(ml_variant.expected.spec_risk) if ml_variant.expected else {}),
    )

    sulfur_mean = sulfur_iv.mean if sulfur_iv is not None else float(row.get("pak_sulfur_ppm", 8.0))
    metrics = {
        "safety": round(_safety_from_sulfur(sulfur_mean), 3),
        "yield": round(_clip_unit(ml_variant.metrics.get("yield", 0.0)), 3),
        "energy": round(_clip_unit(1.0 - ml_variant.metrics.get("energy", 0.0)), 3),
        "wear": round(_wear_from_severity(ml_variant.metrics.get("severity", 0.5)), 3),
    }

    return ApiVariant(
        id=_rand_id(),
        action=action_out,
        delta=delta,
        metrics=metrics,
        predicted=predicted,
        feasible=bool(ml_variant.feasible),
        infeasible_reason=ml_variant.infeasible_reason,
    )


def _iv(x) -> ApiInterval | None:
    if x is None:
        return None
    return ApiInterval(mean=float(x.mean), low=float(x.low), high=float(x.high), unit=x.unit)


def real_recommend(timestamp, simulator) -> RecommendationResponse:
    """Гоняет реальные модели и оптимизатор, приводит к API-формату."""
    from ml.types import OptimizationConstraints

    decision_id = _decision_id()
    trace: list[TraceStep] = []
    warnings: list[str] = []

    avt, hydro, anom, blending, optimizer = _load_bundle()

    t0 = time.perf_counter()
    state = simulator.get_state(timestamp)
    trace.append(TraceStep(
        agent="data",
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_summary=f"timestamp={timestamp.isoformat()}",
        output={"n_tags": int(state.shape[1] - 1)},
    ))

    t0 = time.perf_counter()
    anom_report = anom.score(state)
    trace.append(TraceStep(
        agent="data-anomaly",
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_summary="state snapshot",
        output={
            "is_anomaly": bool(anom_report.is_anomaly),
            "score": round(float(anom_report.anomaly_score), 3),
            "flagged": list(anom_report.flagged_tags)[:5],
            "stale": list(anom_report.stale_tags)[:5],
        },
    ))

    t0 = time.perf_counter()
    q_hydro = hydro.predict(state)
    sulfur_iv = q_hydro.predictions.get("sulfur_ppm")
    trace.append(TraceStep(
        agent="quality",
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        input_summary="state snapshot",
        output={
            "sulfur_pred": (round(float(sulfur_iv.mean), 2) if sulfur_iv else None),
            "confidence": q_hydro.confidence,
            "spec_risk_over_10": round(float(q_hydro.spec_risk.get("sulfur_over_10", 0.0)), 2),
        },
    ))

    trace.append(TraceStep(
        agent="reliability",
        duration_ms=0.5,
        input_summary="state snapshot",
        output={"note": "прокси через энергия+тяжесть в оптимизаторе"},
    ))

    if anom_report.is_anomaly and anom_report.anomaly_score > 0.85:
        warnings.append("данные аномальны, вмешательство не рекомендуется")
        return RecommendationResponse(
            decision_id=decision_id,
            mode="refuse",
            timestamp=timestamp.isoformat(),
            variants=[],
            default_weights=DEFAULT_WEIGHTS,
            trace=trace,
            explanation_text="Данные текущего момента аномальны (score {:.2f}). Рекомендация не выдаётся.".format(
                float(anom_report.anomaly_score)
            ),
            warnings=warnings,
        )

    sulfur_now = float(state.iloc[0].get("pak_sulfur_ppm", 0.0))
    if sulfur_now < 7.0 and (sulfur_iv is None or sulfur_iv.mean < 8.0):
        return RecommendationResponse(
            decision_id=decision_id,
            mode="silent",
            timestamp=timestamp.isoformat(),
            variants=[],
            default_weights=DEFAULT_WEIGHTS,
            trace=trace,
            explanation_text=f"Режим стабильный (сера {sulfur_now:.2f} мг/кг). Вмешательство не требуется.",
        )

    envelope = _envelope_bounds(avt, hydro)
    controllable_ranges = _controllable_ranges_from_state(state, envelope)
    if not controllable_ranges:
        warnings.append("нет управляемых тегов внутри исторического envelope")
        return RecommendationResponse(
            decision_id=decision_id,
            mode="refuse",
            timestamp=timestamp.isoformat(),
            variants=[],
            default_weights=DEFAULT_WEIGHTS,
            trace=trace,
            explanation_text="Текущий режим вне исторических данных по всем управляемым тегам.",
            warnings=warnings,
        )

    constraints = OptimizationConstraints(
        hard={"sulfur_ppm": (0.0, 10.0)},
        controllable_ranges=controllable_ranges,
        max_deviation_pct=3.0,
    )

    t0 = time.perf_counter()
    ml_variants = optimizer.find_pareto(state, constraints, n_variants=12)
    opt_ms = round((time.perf_counter() - t0) * 1000, 1)
    n_feasible = sum(1 for v in ml_variants if v.feasible)
    trace.append(TraceStep(
        agent="optimization",
        duration_ms=opt_ms,
        input_summary=f"n_controllable={len(controllable_ranges)}, n_variants=12",
        output={"n_returned": len(ml_variants), "n_feasible": n_feasible},
    ))

    api_variants = [_to_api_variant(v, state) for v in ml_variants]
    if n_feasible == 0 and api_variants:
        warnings.append("все варианты за историческим envelope, показаны приближённые")
        for v in api_variants:
            v.feasible = True

    if not api_variants:
        return RecommendationResponse(
            decision_id=decision_id,
            mode="refuse",
            timestamp=timestamp.isoformat(),
            variants=[],
            default_weights=DEFAULT_WEIGHTS,
            trace=trace,
            explanation_text="Оптимизатор не нашёл ни одного варианта.",
            warnings=warnings,
        )

    best = max(api_variants, key=lambda v: sum(DEFAULT_WEIGHTS[k] * v.metrics[k] for k in DEFAULT_WEIGHTS))
    changes = ", ".join(f"{t} {d:+.2f}" for t, d in list(best.delta.items())[:3])
    pred = best.predicted.sulfur.mean if best.predicted and best.predicted.sulfur else "?"
    explanation = (
        f"Сера сейчас {sulfur_now:.2f} мг/кг. "
        f"Рекомендуемые изменения: {changes}. "
        f"Прогноз серы после воздействия: {pred} мг/кг."
    )

    return RecommendationResponse(
        decision_id=decision_id,
        mode="recommend",
        timestamp=timestamp.isoformat(),
        variants=api_variants,
        default_weights=DEFAULT_WEIGHTS,
        trace=trace,
        explanation_text=explanation,
        warnings=warnings,
    )
