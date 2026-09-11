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
SILENT_RISK_THRESHOLD = 0.1
ANOMALY_SCORE_THRESHOLD = 0.7


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

        # 2. Проверка данных.
        started = time.perf_counter()
        data_out = await self.data.check(DataAgentInput(timestamp=timestamp))
        self._record(decision_id, "data", {"timestamp": timestamp.isoformat()},
                     _to_dict(data_out), time.perf_counter() - started)

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
        self._record(decision_id, "quality", {}, _to_dict(quality_out), duration_s)
        self._record(decision_id, "reliability", {}, _to_dict(reliability_out), duration_s)

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
        self._record(decision_id, "optimization", _to_dict(constraints),
                     _to_dict(optimization_out), time.perf_counter() - started)

        feasible = [v for v in optimization_out.variants if v.feasible]

        # 7. Нет допустимых вариантов.
        if not feasible:
            reason = "Нет допустимых вариантов в рамках ограничений."
            return self._finish(decision_id, "refuse", {"reason": reason}, reason)

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
        self._record(decision_id, "orchestrator", {}, _to_dict(out), 0.0)
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
