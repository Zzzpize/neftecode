"""Тесты реальных обёрток агентов над ML-компонентами (Фаза 3)."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd
import yaml

from agents.data_agent import DataAgent
from agents.optimization_agent import OptimizationAgent
from agents.orchestrator import Orchestrator
from agents.quality_agent import QualityAgent
from agents.reliability_agent import ReliabilityAgent
from agents.schemas import DataAgentInput
from ml.types import (
    AnomalyReport,
    Interval,
    OptimizationConstraints,
    QualityPrediction,
    Variant,
)
from tracing.logger import AgentTracer


def _run(coro):
    return asyncio.run(coro)


def _state(**overrides) -> pd.DataFrame:
    row = {
        "date": pd.Timestamp("2025-08-01 12:00:00"),
        "pak_sulfur_ppm": 8.0,
        "avt_T55": 348.0,
        "hydro_T5": 300.0,
        "lims__гидроочистка__pt2__mg_sulfur_age_h": 2.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _write_registry(tmp_path, data: dict):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# ---------- DataAgent ----------

class _Simulator:
    def __init__(self, state):
        self.state = state

    def get_state(self, ts):
        return self.state


class _Anomaly:
    def __init__(self, report):
        self.report = report

    def score(self, state):
        return self.report


def test_data_agent_delegates_to_anomaly_and_simulator():
    report = AnomalyReport(
        is_anomaly=True,
        anomaly_score=0.8,
        flagged_tags=["avt_T55"],
        is_out_of_envelope=False,
        stale_tags=[],
    )
    agent = DataAgent(anomaly=_Anomaly(report), simulator=_Simulator(_state()))

    out = _run(agent.check(DataAgentInput(timestamp=datetime(2025, 8, 1, 12, 0, 0))))

    assert out.has_anomalies is True
    assert out.anomaly_score == 0.8
    assert out.flagged_tags == ["avt_T55"]
    assert out.is_stale is False


def test_data_agent_detects_stale_lims():
    report = AnomalyReport(
        is_anomaly=False,
        anomaly_score=0.0,
        flagged_tags=[],
        is_out_of_envelope=False,
        stale_tags=[],
    )
    state = _state(**{"lims__гидроочистка__pt2__mg_sulfur_age_h": 30.0})
    agent = DataAgent(anomaly=_Anomaly(report), simulator=_Simulator(state))

    out = _run(agent.check(DataAgentInput(timestamp=datetime(2025, 8, 1, 12, 0, 0))))

    assert out.is_stale is True
    assert any("Устаревшие" in w for w in out.warnings)


# ---------- QualityAgent ----------

class _AVT:
    def predict(self, state):
        return QualityPrediction(
            predictions={"T50": Interval(280.0, 275.0, 285.0, "C")},
            spec_risk={"avt_risk": 0.1},
            confidence="high",
            warnings=[],
        )

    def predict_after_action(self, state, action):
        return self.predict(state)


class _Hydro:
    def predict(self, state):
        return QualityPrediction(
            predictions={"sulfur_ppm": Interval(7.0, 6.5, 7.5, "ppm")},
            spec_risk={"sulfur_over_10": 0.4},
            confidence="high",
            warnings=[],
        )

    def predict_after_action(self, state, action):
        return self.predict(state)


def test_quality_agent_merges_spec_risk():
    agent = QualityAgent(avt=_AVT(), hydro=_Hydro())

    out = _run(agent.forecast(_state()))

    assert out.avt.spec_risk["avt_risk"] == 0.1
    assert out.hydro.spec_risk["sulfur_over_10"] == 0.4
    assert out.combined_spec_risk["sulfur_over_10"] == 0.4
    assert out.combined_spec_risk["avt_risk"] == 0.1




# ---------- ReliabilityAgent ----------

def test_reliability_agent_returns_limits_for_controllable_tags(tmp_path):
    registry = {
        "hydro_T5": {"controllable": True, "range_min": 300.0, "range_max": 340.0},
        "avt_T55": {"controllable": True, "range_min": 340.0, "range_max": 370.0},
    }
    agent = ReliabilityAgent(registry_path=_write_registry(tmp_path, registry))

    out = _run(agent.assess(_state()))

    assert out.severity_class == "normal"
    assert out.limits["hydro_T5"] == (300.0, 340.0)
    assert out.limits["avt_T55"] == (340.0, 370.0)


def test_reliability_agent_flags_out_of_range(tmp_path):
    # hydro_T5 = 300.0 при норме 310..340 -> ниже нормы.
    registry = {
        "hydro_T5": {"controllable": True, "range_min": 310.0, "range_max": 340.0},
    }
    agent = ReliabilityAgent(registry_path=_write_registry(tmp_path, registry))

    out = _run(agent.assess(_state()))

    assert out.severity_class == "elevated"
    assert out.severity_score > 0.0
    assert any("hydro_T5" in f for f in out.risk_factors)


# ---------- OptimizationAgent ----------

class _Optimizer:
    def find_pareto(self, state, constraints, n_variants=10):
        return [
            Variant(
                action={"hydro_T5": 310.0},
                expected=QualityPrediction(
                    predictions={"sulfur_ppm": Interval(6.0, 5.5, 6.5, "ppm")},
                    spec_risk={"sulfur_over_10": 0.1},
                    confidence="high",
                    warnings=[],
                ),
                metrics={"sulfur": 6.0, "yield": 0.85, "energy": 0.5, "severity": 0.2},
                feasible=True,
                infeasible_reason=None,
            ),
            Variant(
                action={"hydro_T5": 315.0},
                expected=QualityPrediction(
                    predictions={"sulfur_ppm": Interval(7.0, 6.5, 7.5, "ppm")},
                    spec_risk={"sulfur_over_10": 0.2},
                    confidence="high",
                    warnings=[],
                ),
                metrics={"sulfur": 7.0, "yield": 0.80, "energy": 0.6, "severity": 0.3},
                feasible=True,
                infeasible_reason=None,
            ),
        ]


def test_optimization_agent_delegates_to_optimizer():
    agent = OptimizationAgent(optimizer=_Optimizer())

    out = _run(agent.find_variants(_state(), OptimizationConstraints(
        hard={"sulfur_ppm": (0.0, 10.0)},
        controllable_ranges={"hydro_T5": (300.0, 340.0)},
        max_deviation_pct=10.0,
    )))

    assert len(out.variants) == 2
    assert all(v.feasible for v in out.variants)
    assert out.infeasible_reasons == []


# ---------- Оркестратор end-to-end на реальных обёртках ----------

def test_orchestrator_end_to_end_with_real_agents(tmp_path):
    state = _state()
    registry = {
        "hydro_T5": {"controllable": True, "range_min": 290.0, "range_max": 340.0},
    }
    report = AnomalyReport(
        is_anomaly=False,
        anomaly_score=0.0,
        flagged_tags=[],
        is_out_of_envelope=False,
        stale_tags=[],
    )

    data = DataAgent(anomaly=_Anomaly(report), simulator=_Simulator(state))
    quality = QualityAgent(avt=_AVT(), hydro=_Hydro())
    reliability = ReliabilityAgent(registry_path=_write_registry(tmp_path, registry))
    optimization = OptimizationAgent(optimizer=_Optimizer())
    tracer = AgentTracer(tmp_path / "tracing.db")

    orch = Orchestrator(data, quality, reliability, optimization, formatter=None, tracer=tracer)

    out = _run(orch.decide(datetime(2025, 8, 1, 12, 0, 0)))

    assert out.mode == "recommend"
    assert out.payload["best_variant"]
    assert out.payload["alternatives"]

    agents = {e.agent for e in tracer.get_trace(out.decision_id)}
    assert {"data", "quality", "reliability", "optimization", "orchestrator"} <= agents

def test_quality_agent_forecast_after_action():
    agent = QualityAgent(avt=_AVT(), hydro=_Hydro())

    out = _run(agent.forecast_after_action(_state(), {"hydro_T5": 310.0}))

    assert out.hydro.spec_risk["sulfur_over_10"] == 0.4
