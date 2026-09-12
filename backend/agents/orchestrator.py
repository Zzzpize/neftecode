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

DEFAULT_WEIGHTS = {"safety": 0.5, "yield": 0.2, "energy": 0.15, "wear": 0.15}
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

    async def decide(
        self,
        timestamp: datetime,
        weights: dict[str, float] | None = None,
    ) -> OrchestratorOutput:
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
            return self._finish(decision_id, "refuse", {"reason": reason}, reason, timestamp=timestamp)

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
                timestamp=timestamp,
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
            return self._finish(decision_id, "refuse", {"reason": reason}, reason, timestamp=timestamp)

        # 8. Ранжирование по взвешенной сумме метрик.
        # Маппинг весов (safety/yield/energy/wear) на ML-метрики:
        #   safety -> sulfur, wear -> severity, yield -> yield, energy -> energy.
        weights = self._normalize_weights(weights)
        ranked = sorted(feasible, key=lambda v: self._rank_score(v, weights), reverse=True)

        # 9. Лучший вариант + 2-3 альтернативы.
        best = ranked[0]
        alternatives = ranked[1:4]
        payload = {
            "timestamp": timestamp.isoformat(),
            "best_variant": _to_dict(best),
            "alternatives": [_to_dict(v) for v in alternatives],
            "checks": {
                "combined_spec_risk": quality_out.combined_spec_risk,
                "severity_class": reliability_out.severity_class,
                "sulfur_limit_ppm": 10.0,
            },
            "weights": dict(weights),
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
        return self._finish(decision_id, "recommend", payload, explanation_text, timestamp=timestamp)

    def _finish(
        self,
        decision_id: str,
        mode: str,
        payload: dict,
        explanation_text: str,
        timestamp: datetime | None = None,
    ) -> OrchestratorOutput:
        payload = dict(payload)
        if timestamp is not None:
            payload.setdefault("timestamp", timestamp.isoformat())
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
    def _rank_score(variant, weights: dict[str, float] | None = None) -> float:
        """Взвешенная оценка варианта; больше — лучше.

        ML-метрики ``{sulfur, yield, energy, severity}`` приводятся к той же
        шкале ``{safety, yield, energy, wear}``, что и на фронте/в ``ai_adapter``,
        чтобы backend-ранжирование совпадало с выбором оператора.
        """
        weights = weights or DEFAULT_WEIGHTS
        metrics = variant.metrics
        severity = float(metrics.get("severity", 0.0))
        sulfur = float(metrics.get("sulfur", 0.0))
        yield_score = float(metrics.get("yield", 0.0))
        energy = float(metrics.get("energy", 0.0))

        # Предпочитаем прогноз из expected (как это делает ai_adapter).
        expected = getattr(variant, "expected", None)
        if expected is not None:
            predictions = getattr(expected, "predictions", None) or {}
            sulfur_iv = predictions.get("sulfur_ppm")
            if sulfur_iv is not None:
                sulfur = float(getattr(sulfur_iv, "mean", sulfur))

        safety = max(0.05, min(1.0, 1.0 - sulfur / 12.0))   # safety <- sulfur
        yield_norm = max(0.0, min(1.0, yield_score))
        energy_norm = max(0.0, min(1.0, 1.0 - energy))       # energy -> больше=лучше
        wear = max(0.05, min(1.0, 1.0 - severity))           # wear <- severity

        return (
            weights.get("safety", 0.0) * safety
            + weights.get("yield", 0.0) * yield_norm
            + weights.get("energy", 0.0) * energy_norm
            + weights.get("wear", 0.0) * wear
        )

    @staticmethod
    def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
        """Объединяет пользовательские веса с дефолтными, оставляя только известные ключи."""
        merged = dict(DEFAULT_WEIGHTS)
        if weights:
            for key in DEFAULT_WEIGHTS:
                value = weights.get(key)
                if isinstance(value, (int, float)):
                    merged[key] = float(value)
        return merged

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
