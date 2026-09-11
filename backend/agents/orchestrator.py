"""Orchestrator: связывает агентов в единый цикл принятия решения.

Порядок: data -> (quality + reliability) -> optimizer -> ранжирование -> ответ.
Приоритет разрешения конфликтов: безопасность > качество > экономика.
"""
from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from agents.schemas import DataAgentInput, OrchestratorOutput
from ml.types import OptimizationConstraints

DEFAULT_WEIGHTS = {"safety": 0.5, "quality": 0.3, "yield": 0.15, "energy": 0.05}
SILENT_RISK_THRESHOLD = 0.15
ANOMALY_SCORE_THRESHOLD = 0.9


def _to_dict(value: Any) -> Any:
    """Превращает pydantic-модели и dataclass-типы во вложенные plain dict."""
    if isinstance(value, dict):
        return {k: _to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(v) for v in value]
    if is_dataclass(value):
        return {k: _to_dict(v) for k, v in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return _to_dict(value.model_dump())
    return value


class Orchestrator:
    def __init__(self, data, quality, reliability, optimization, formatter=None, tracer=None):
        self.data = data
        self.quality = quality
        self.reliability = reliability
        self.optimization = optimization
        self.formatter = formatter
        self.tracer = tracer

    async def decide(self, timestamp: datetime) -> OrchestratorOutput:
        # 1. Новый decision_id.
        decision_id = (
            self.tracer.start_decision()
            if self.tracer is not None
            else uuid.uuid4().hex
        )
        self._timestamp_iso: str = timestamp.isoformat()

        # 2. Проверка данных.
        started = time.perf_counter()
        data_out = await self.data.check(DataAgentInput(timestamp=timestamp))
        self._record(decision_id, "data", {"timestamp": timestamp.isoformat()},
                     {
                         "is_stale": data_out.is_stale,
                         "has_anomalies": data_out.has_anomalies,
                         "anomaly_score": round(float(data_out.anomaly_score), 3),
                         "n_flagged": len(data_out.flagged_tags),
                         "n_stale": len(data_out.stale_tags),
                     },
                     time.perf_counter() - started)

        if data_out.is_stale or (
            data_out.has_anomalies and data_out.anomaly_score > ANOMALY_SCORE_THRESHOLD
        ):
            reason = " ".join(data_out.warnings) or "данные непригодны для рекомендации"
            return self._finish(decision_id, "refuse", {"reason": reason}, reason)

        state = self.data.get_state(timestamp)

        # 3. Quality и Reliability — параллельно.
        started = time.perf_counter()
        quality_out, reliability_out = await asyncio.gather(
            self.quality.forecast(state),
            self.reliability.assess(state),
        )
        duration_s = time.perf_counter() - started
        sulfur_pred = quality_out.hydro.predictions.get("sulfur_ppm")
        self._record(decision_id, "quality", {},
                     {
                         "sulfur_pred": round(float(sulfur_pred.mean), 2) if sulfur_pred else None,
                         "confidence": quality_out.hydro.confidence,
                         "spec_risk_over_10": round(float(quality_out.combined_spec_risk.get("sulfur_over_10", 0.0)), 3),
                     },
                     duration_s)
        self._record(decision_id, "reliability", {},
                     {
                         "severity_class": reliability_out.severity_class,
                         "severity_score": round(float(reliability_out.severity_score), 3),
                         "n_risk_factors": len(reliability_out.risk_factors),
                         "n_controllable": len(reliability_out.limits),
                     },
                     duration_s)

        # 4. Вмешательство не требуется?
        risks = quality_out.combined_spec_risk.values()
        if reliability_out.severity_class == "normal" and (
            not risks or all(risk < SILENT_RISK_THRESHOLD for risk in risks)
        ):
            return self._finish(
                decision_id, "silent", {},
                "Режим стабилен, вмешательство не требуется.",
            )

        # 5. Сборка ограничений оптимизатора.
        constraints = OptimizationConstraints(
            hard={"sulfur_ppm": (0.0, 10.0)},
            controllable_ranges=dict(reliability_out.limits),
            max_deviation_pct=10.0,
        )

        # 6. Оптимизация.
        started = time.perf_counter()
        optimization_out = await self.optimization.find_variants(state, constraints)
        n_feasible = sum(1 for v in optimization_out.variants if v.feasible)
        self._record(decision_id, "optimization",
                     {"n_controllable": len(constraints.controllable_ranges)},
                     {
                         "n_variants": len(optimization_out.variants),
                         "n_feasible": n_feasible,
                     },
                     time.perf_counter() - started)

        feasible = [v for v in optimization_out.variants if v.feasible]

        # 7. Нет допустимых вариантов.
        if not feasible:
            if not optimization_out.variants:
                reason = "Оптимизатор не нашёл ни одного варианта."
                return self._finish(decision_id, "refuse", {"reason": reason}, reason)
            feasible = list(optimization_out.variants)
            for variant in feasible:
                variant.feasible = True

        # 8. Ранжирование по взвешенной сумме метрик.
        # Маппинг весов ТЗ на ML-метрики:
        #   безопасность -> severity, качество -> sulfur, выход -> yield, энергия -> energy.
        ranked = sorted(feasible, key=self._rank_score, reverse=True)

        # 9. Лучший вариант + 2-3 альтернативы.
        best = ranked[0]
        alternatives = ranked[1:4]
        payload = {
            "best_variant": _to_dict(best),
            "alternatives": [_to_dict(v) for v in alternatives],
            "checks": {
                "combined_spec_risk": quality_out.combined_spec_risk,
                "severity_class": reliability_out.severity_class,
                "sulfur_limit_ppm": 10.0,
            },
            "weights": dict(DEFAULT_WEIGHTS),
        }

        decision = OrchestratorOutput(
            decision_id=decision_id,
            mode="recommend",
            payload=payload,
            trace_id=decision_id,
            explanation_text="",
        )

        # 10-11. Текст рекомендации (LLM или fallback) и ответ.
        explanation_text = await self._format(decision)
        return self._finish(decision_id, "recommend", payload, explanation_text)

    def _finish(self, decision_id: str, mode: str, payload: dict, explanation_text: str) -> OrchestratorOutput:
        out = OrchestratorOutput(
            decision_id=decision_id,
            mode=mode,
            payload=payload,
            trace_id=decision_id,
            explanation_text=explanation_text,
        )
        summary: dict = {"mode": mode, "timestamp": getattr(self, "_timestamp_iso", "")}
        if mode == "recommend":
            best = payload.get("best_variant", {}) or {}
            action = best.get("action") or {}
            expected = best.get("expected") or {}
            predictions = expected.get("predictions") or {}
            sulfur = predictions.get("sulfur_ppm") or {}
            top_action = dict(list(action.items())[:5])
            checks = payload.get("checks") or {}
            summary.update({
                "n_alternatives": len(payload.get("alternatives") or []),
                "n_actions": len(action),
                "top_action": top_action,
                "predicted_sulfur_ppm": sulfur.get("mean"),
                "sulfur_over_10_risk": (checks.get("combined_spec_risk") or {}).get("sulfur_over_10"),
                "severity_class": checks.get("severity_class"),
                "explanation": explanation_text[:280],
            })
        elif mode == "refuse":
            summary["reason"] = str(payload.get("reason", ""))[:180]
        elif mode == "silent":
            summary["explanation"] = explanation_text[:180]
        self._record(decision_id, "orchestrator", {}, summary, 0.0)
        return out

    async def _format(self, decision: OrchestratorOutput) -> str:
        if self.formatter is None:
            return self._fallback(decision)
        try:
            result = self.formatter.format_recommendation(decision)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, str) else self._fallback(decision)
        except Exception:
            return self._fallback(decision)

    @staticmethod
    def _fallback(decision: OrchestratorOutput) -> str:
        best = decision.payload.get("best_variant") or {}
        action = best.get("action") or {}
        if action:
            changes = ", ".join(
                f"{tag} {value:+.2f}" for tag, value in list(action.items())[:3]
            )
            return f"Рекомендуемые изменения: {changes}."
        return "Рекомендация сформирована."

    @staticmethod
    def _rank_score(variant) -> float:
        """Взвешенная оценка варианта; больше — лучше.

        ML-метрики ``{sulfur, yield, energy, severity}`` приводятся к единой
        шкале «больше = лучше» и взвешиваются ``DEFAULT_WEIGHTS``.
        """
        metrics = variant.metrics
        severity = float(metrics.get("severity", 0.0))
        sulfur = float(metrics.get("sulfur", 0.0))
        yield_score = float(metrics.get("yield", 0.0))
        energy = float(metrics.get("energy", 0.0))

        safety = 1.0 - min(1.0, severity)          # severity 0..1
        quality = max(0.0, 1.0 - sulfur / 10.0)    # sulfur ppm, спека 10
        energy_score = 1.0 - min(1.0, energy)      # energy 0..1

        return (
            DEFAULT_WEIGHTS["safety"] * safety
            + DEFAULT_WEIGHTS["quality"] * quality
            + DEFAULT_WEIGHTS["yield"] * yield_score
            + DEFAULT_WEIGHTS["energy"] * energy_score
        )

    def _record(self, decision_id: str, agent: str, input_data: dict, output: dict, duration_s: float) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.record(
                decision_id=decision_id,
                agent=agent,
                input_data=input_data,
                output=output,
                duration_ms=round(duration_s * 1000, 1),
            )
        except Exception:
            pass
