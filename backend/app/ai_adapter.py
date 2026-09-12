"""Адаптер OrchestratorOutput -> RecommendationResponse.

Оркестратор AI-слоя отдаёт ``agents.schemas.OrchestratorOutput``, а HTTP-роут
``/recommend`` обязан вернуть ``app.schemas.RecommendationResponse``. Этот модуль
переводит один формат в другой, переиспользуя метрики карточки из ``ml_adapter``,
чтобы агентный и неагентный пути давали одинаковый JSON для фронта.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from agents.schemas import OrchestratorOutput
from app.ml_adapter import (
    DEFAULT_WEIGHTS,
    _clip_unit,
    _rand_id,
    _safety_from_sulfur,
    _wear_from_severity,
)
from app.schemas import (
    Interval as ApiInterval,
    QualityPrediction as ApiQualityPrediction,
    RecommendationResponse,
    TraceStep,
    Variant as ApiVariant,
)


def _iv(value: Any) -> ApiInterval | None:
    """``ml.types.Interval`` (уже в виде dict) -> API Interval."""
    if not isinstance(value, dict):
        return None
    try:
        return ApiInterval(
            mean=float(value["mean"]),
            low=float(value["low"]),
            high=float(value["high"]),
            unit=str(value.get("unit", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _variant_to_api(variant: dict, row: pd.Series) -> ApiVariant:
    """``ml.types.Variant`` (сериализованный оркестратором) -> API Variant."""
    action_in = variant.get("action") or {}
    action_out: dict[str, float] = {}
    delta: dict[str, float] = {}
    for tag, new_val in action_in.items():
        try:
            new_f = float(new_val)
        except (TypeError, ValueError):
            continue
        action_out[tag] = round(new_f, 3)
        if tag in row.index and not pd.isna(row[tag]):
            delta[tag] = round(new_f - float(row[tag]), 3)

    expected = variant.get("expected") or {}
    predictions = expected.get("predictions") or {}
    predicted = ApiQualityPrediction(
        sulfur=_iv(predictions.get("sulfur_ppm")),
        t50=_iv(predictions.get("T50")),
        t90=_iv(predictions.get("T90")),
        d15=_iv(predictions.get("D15")),
        confidence=str(expected.get("confidence") or "medium"),
        spec_risk=dict(expected.get("spec_risk") or {}),
    )

    metrics = variant.get("metrics") or {}
    sulfur_pred = predictions.get("sulfur_ppm") or {}
    sulfur_mean = sulfur_pred.get("mean")
    if sulfur_mean is None:
        sulfur_mean = float(row.get("pak_sulfur_ppm", 8.0))

    severity = metrics.get("severity")
    yield_metric = metrics.get("yield")
    energy_metric = metrics.get("energy")

    api_metrics = {
        "safety": round(_safety_from_sulfur(float(sulfur_mean)), 3),
        "yield": round(_clip_unit(float(yield_metric) if yield_metric is not None else 0.0), 3),
        "energy": round(_clip_unit(1.0 - (float(energy_metric) if energy_metric is not None else 0.0)), 3),
        "wear": round(_wear_from_severity(float(severity) if severity is not None else 0.5), 3),
    }

    return ApiVariant(
        id=_rand_id(),
        action=action_out,
        delta=delta,
        metrics=api_metrics,
        predicted=predicted,
        feasible=bool(variant.get("feasible", True)),
        infeasible_reason=variant.get("infeasible_reason"),
    )


def trace_to_steps(tracer, decision_id: str) -> list[TraceStep]:
    """``tracing.logger.TraceEntry`` -> список ``TraceStep`` для фронта."""
    steps: list[TraceStep] = []
    try:
        entries = tracer.get_trace(decision_id)
    except Exception:
        return steps
    for entry in entries:
        output: dict = {}
        try:
            parsed = json.loads(entry.output_summary)
            if isinstance(parsed, dict):
                output = parsed
        except (json.JSONDecodeError, TypeError):
            output = {"raw": entry.output_summary}
        steps.append(TraceStep(
            agent=entry.agent,
            duration_ms=entry.duration_ms,
            input_summary=(entry.input_hash or "")[:16],
            output=output,
        ))
    return steps


def to_recommendation_response(
    out: OrchestratorOutput,
    timestamp: datetime,
    simulator,
    tracer,
) -> RecommendationResponse:
    warnings: list[str] = []
    variants: list[ApiVariant] = []

    if out.mode == "recommend":
        state = simulator.get_state(timestamp)
        row = state.iloc[0]
        ordered: list[dict] = []
        best = out.payload.get("best_variant")
        if best:
            ordered.append(best)
        ordered.extend(out.payload.get("alternatives") or [])
        variants = [_variant_to_api(v, row) for v in ordered]
    elif out.mode == "refuse":
        reason = out.payload.get("reason")
        if reason:
            warnings.append(str(reason))

    weights = out.payload.get("weights") or DEFAULT_WEIGHTS
    return RecommendationResponse(
        decision_id=out.decision_id,
        mode=out.mode,
        timestamp=timestamp.isoformat(),
        variants=variants,
        default_weights=dict(weights),
        trace=trace_to_steps(tracer, out.decision_id),
        explanation_text=out.explanation_text,
        warnings=warnings,
    )
