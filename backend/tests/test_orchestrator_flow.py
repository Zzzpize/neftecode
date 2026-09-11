"""End-to-end прогон оркестратора на mock-агентах (Фаза 1).

Проверяет полный цикл ``Orchestrator.decide`` для трёх режимов
(silent / recommend / refuse) без реальных ML-моделей и LLM.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

from agents.data_agent import DataAgent
from agents.optimization_agent import OptimizationAgent
from agents.orchestrator import Orchestrator
from agents.quality_agent import QualityAgent
from agents.reliability_agent import ReliabilityAgent


class MockTracer:
    def __init__(self):
        self._decisions: dict[str, list[dict]] = {}
        self._counter = 0

    def start_decision(self) -> str:
        self._counter += 1
        return f"decision-{self._counter}"

    def record(self, decision_id, agent, input_data, output, duration_ms):
        self._decisions.setdefault(decision_id, []).append({
            "agent": agent,
            "input": input_data,
            "output": output,
            "duration_ms": duration_ms,
        })

    def get_trace(self, decision_id) -> list[dict]:
        return self._decisions.get(decision_id, [])


class MockFormatter:
    async def format_recommendation(self, decision) -> str:
        best = decision.payload.get("best_variant") or {}
        action = best.get("action") or {}
        changes = ", ".join(f"{tag}={value}" for tag, value in action.items())
        return f"Рекомендация: {changes}" if changes else "Рекомендация."


def make_orchestrator(*, stale=False, anomaly=False, anomaly_score=0.0,
                      severity="normal", spec_risk=0.0, variants=None):
    data = DataAgent()
    data.mock_is_stale = stale
    data.mock_has_anomalies = anomaly
    data.mock_anomaly_score = anomaly_score
    data.mock_warnings = ["данные устарели"] if stale else []

    quality = QualityAgent()
    quality.mock_combined_spec_risk = {"sulfur_over_10": spec_risk}

    reliability = ReliabilityAgent()
    reliability.mock_severity_class = severity

    optimization = OptimizationAgent()
    if variants is not None:
        optimization.mock_variants = variants

    tracer = MockTracer()
    formatter = MockFormatter()
    orch = Orchestrator(data, quality, reliability, optimization, formatter, tracer)
    return orch, tracer


def _run(coro):
    return asyncio.run(coro)


def test_decide_returns_recommend():
    orch, tracer = make_orchestrator(severity="elevated", spec_risk=0.3)

    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))

    assert out.mode == "recommend"
    assert out.payload["best_variant"]
    assert out.payload["alternatives"]
    assert out.explanation_text
    agents = {step["agent"] for step in tracer.get_trace(out.decision_id)}
    assert {"data", "quality", "reliability", "optimization", "orchestrator"} <= agents


def test_decide_returns_silent():
    orch, _ = make_orchestrator(severity="normal", spec_risk=0.0)

    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))

    assert out.mode == "silent"
    assert out.explanation_text


def test_decide_refuses_on_stale_data():
    orch, _ = make_orchestrator(stale=True)

    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))

    assert out.mode == "refuse"
    assert "устарели" in out.explanation_text


def test_decide_refuses_on_high_anomaly_score():
    orch, _ = make_orchestrator(anomaly=True, anomaly_score=0.9)

    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))

    assert out.mode == "refuse"


def test_decide_refuses_when_no_feasible_variants():
    orch, _ = make_orchestrator(severity="elevated", spec_risk=0.3, variants=[])

    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))

    assert out.mode == "refuse"
    assert "допустимых вариантов" in out.explanation_text


def test_decide_falls_back_without_formatter():
    orch, _ = make_orchestrator(severity="elevated", spec_risk=0.3)
    orch.formatter = None

    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))

    assert out.mode == "recommend"
    assert "Рекомендуемые изменения" in out.explanation_text


def test_decide_completes_under_3_seconds_without_llm():
    """Критерий приёмки #5: цикл ``decide`` без LLM быстрее 3 секунд."""
    orch, _ = make_orchestrator(severity="elevated", spec_risk=0.3)

    started = time.perf_counter()
    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))
    elapsed = time.perf_counter() - started

    assert out.mode == "recommend"
    assert elapsed < 3.0
