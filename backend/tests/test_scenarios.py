"""Демо-сценарии оркестратора (раздел 10 ``spec.md``).

Три сценария, каждый возвращает ожидаемый ``mode``:

1. ``stable_period`` — нет риска и аномалий, ожидается ``silent``.
2. ``sulfur_risk`` — прогноз серы с риском превышения спеки, ожидается
   ``recommend`` с содержательным изменением управляемых параметров.
3. ``stale_data`` — ЛИМС старше 48 часов и аномальная телеметрия, ожидается
   ``refuse``.

Сценарии гоняются на реальных обёртках агентов, реальном оркестраторе и
SQLite-трейсере. ML-компоненты подменены тест-двойниками на границе, потому что
в этом окружении отсутствуют ``master.parquet`` и обученные артефакты моделей.

Примечание по Фазе 7: конкретные timestamp сценариев подбираются совместно с ML
после появления реальных данных — здесь они зафиксированы как явные константы,
семантика которых описана в ``docs/spec.md`` §10.
"""
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
from ml.types import AnomalyReport, Interval, QualityPrediction, Variant
from tracing.logger import AgentTracer

# Место временных меток для согласования с ML (docs/spec.md §10).
STABLE_TS = datetime(2025, 7, 1, 12, 0, 0)
SULFUR_TS = datetime(2025, 7, 10, 14, 0, 0)
STALE_TS = datetime(2025, 7, 20, 3, 0, 0)


def _run(coro):
    return asyncio.run(coro)


def _state(**overrides) -> pd.DataFrame:
    row = {
        "date": pd.Timestamp("2025-07-10 14:00:00"),
        "pak_sulfur_ppm": 8.5,
        "avt_T55": 380.0,
        "hydro_T5": 320.0,
        "lims__гидроочистка__pt2__mg_sulfur_age_h": 2.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


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


def _report(*, is_anomaly=False, score=0.0, flagged=None,
            out_of_envelope=False, stale=None) -> AnomalyReport:
    return AnomalyReport(
        is_anomaly=is_anomaly,
        anomaly_score=score,
        flagged_tags=flagged or [],
        is_out_of_envelope=out_of_envelope,
        stale_tags=stale or [],
    )


class _QualityModel:
    def __init__(self, spec_risk, mean=7.0, low=6.5, high=7.5):
        self.spec_risk = dict(spec_risk)
        self.mean = mean
        self.low = low
        self.high = high

    def predict(self, state) -> QualityPrediction:
        return QualityPrediction(
            predictions={
                "sulfur_ppm": Interval(self.mean, self.low, self.high, "ppm")
            },
            spec_risk=dict(self.spec_risk),
            confidence="high",
            warnings=[],
        )

    def predict_after_action(self, state, action) -> QualityPrediction:
        return self.predict(state)


def _variant(action, *, sulfur, yield_score, energy, severity) -> Variant:
    return Variant(
        action=action,
        expected=QualityPrediction(
            predictions={
                "sulfur_ppm": Interval(sulfur, sulfur - 0.4, sulfur + 0.4, "ppm")
            },
            spec_risk={
                "sulfur_over_10": round(max(0.0, (sulfur - 6.0) / 8.0), 2)
            },
            confidence="high",
            warnings=[],
        ),
        metrics={"sulfur": sulfur, "yield": yield_score,
                 "energy": energy, "severity": severity},
        feasible=True,
        infeasible_reason=None,
    )


class _Optimizer:
    def __init__(self, variants):
        self.variants = variants

    def find_pareto(self, state, constraints, n_variants=10):
        return list(self.variants)


def _write_registry(tmp_path) -> str:
    registry = {
        "avt_T55": {"controllable": True, "range_min": 375.8, "range_max": 385.01},
        "hydro_T5": {"controllable": True, "range_min": 293.32, "range_max": 384.39},
    }
    path = tmp_path / "feature_registry.yaml"
    path.write_text(yaml.safe_dump(registry, allow_unicode=True), encoding="utf-8")
    return str(path)


def make_orchestrator(tmp_path, *, state, report, avt, hydro, optimizer):
    data = DataAgent(
        anomaly=_Anomaly(report),
        simulator=_Simulator(state),
        stale_threshold_hours=24.0,
    )
    quality = QualityAgent(avt=avt, hydro=hydro)
    reliability = ReliabilityAgent(registry_path=_write_registry(tmp_path))
    optimization = OptimizationAgent(optimizer=optimizer)
    tracer = AgentTracer(tmp_path / "tracing.db")
    orch = Orchestrator(data, quality, reliability, optimization,
                        formatter=None, tracer=tracer)
    return orch, tracer


# ---------- 1. stable_period ----------

def test_stable_period_returns_silent(tmp_path):
    state = _state(date=pd.Timestamp(STABLE_TS), pak_sulfur_ppm=6.5)
    report = _report()
    avt = _QualityModel({"sulfur_over_10": 0.02}, mean=6.3, low=6.0, high=6.6)
    hydro = _QualityModel({"sulfur_over_10": 0.03}, mean=6.4, low=6.1, high=6.7)
    orch, tracer = make_orchestrator(
        tmp_path, state=state, report=report, avt=avt, hydro=hydro,
        optimizer=_Optimizer([]),
    )

    out = _run(orch.decide(STABLE_TS))

    assert out.mode == "silent"
    assert "стабилен" in out.explanation_text.lower()

    agents = {entry.agent for entry in tracer.get_trace(out.decision_id)}
    assert {"data", "quality", "reliability", "orchestrator"} <= agents
    assert "optimization" not in agents  # до оптимизатора не дошли


# ---------- 2. sulfur_risk ----------

def test_sulfur_risk_returns_recommend(tmp_path):
    state = _state(date=pd.Timestamp(SULFUR_TS), pak_sulfur_ppm=8.5)
    report = _report()  # аномалий нет, ЛИМС свежий
    avt = _QualityModel({"sulfur_over_10": 0.05}, mean=8.2, low=7.8, high=8.6)
    hydro = _QualityModel({"sulfur_over_10": 0.40}, mean=8.6, low=8.0, high=9.2)
    optimizer = _Optimizer([
        _variant({"hydro_T5": 310.0}, sulfur=6.2, yield_score=0.85,
                 energy=0.50, severity=0.20),
        _variant({"hydro_T5": 315.0}, sulfur=6.8, yield_score=0.82,
                 energy=0.55, severity=0.25),
        _variant({"avt_T55": 383.0}, sulfur=7.2, yield_score=0.88,
                 energy=0.60, severity=0.30),
    ])
    orch, tracer = make_orchestrator(
        tmp_path, state=state, report=report, avt=avt, hydro=hydro,
        optimizer=optimizer,
    )

    out = _run(orch.decide(SULFUR_TS))

    assert out.mode == "recommend"
    best = out.payload["best_variant"]
    assert best["action"], "ожидается содержательное изменение параметров"

    # Изменение реально сдвигает управляемый тег от текущего значения.
    current = state.iloc[0]["hydro_T5"]
    recommended = best["action"]["hydro_T5"]
    assert abs(recommended - current) > 1e-6

    # Прогнозная сера после действия ниже текущего риска (0.40).
    assert best["expected"]["spec_risk"]["sulfur_over_10"] < 0.40

    agents = {entry.agent for entry in tracer.get_trace(out.decision_id)}
    assert {"data", "quality", "reliability", "optimization", "orchestrator"} <= agents


# ---------- 3. stale_data ----------

def test_stale_data_returns_refuse(tmp_path):
    state = _state(
        date=pd.Timestamp(STALE_TS),
        lims__гидроочистка__pt2__mg_sulfur_age_h=50.0,  # ЛИМС старше 24ч
    )
    report = _report(is_anomaly=True, score=0.9, flagged=["avt_T55"],
                     out_of_envelope=True, stale=["hydro_T5"])
    avt = _QualityModel({"sulfur_over_10": 0.05})
    hydro = _QualityModel({"sulfur_over_10": 0.40})
    orch, tracer = make_orchestrator(
        tmp_path, state=state, report=report, avt=avt, hydro=hydro,
        optimizer=_Optimizer([]),
    )

    out = _run(orch.decide(STALE_TS))

    assert out.mode == "refuse"
    assert "reason" in out.payload
    assert "Устаревшие" in out.explanation_text

    agents = {entry.agent for entry in tracer.get_trace(out.decision_id)}
    assert {"data", "orchestrator"} <= agents
    # Отказ происходит до прогноза и оптимизации.
    assert not {"quality", "reliability", "optimization"} & agents
