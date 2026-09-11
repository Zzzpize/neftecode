"""Тесты адаптера OrchestratorOutput -> RecommendationResponse (app-слой)."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from agents.schemas import OrchestratorOutput
from app.ai_adapter import to_recommendation_response
from ml.types import Interval, QualityPrediction, Variant


def _variant() -> Variant:
    return Variant(
        action={"hydro_T5": 305.0},
        expected=QualityPrediction(
            predictions={"sulfur_ppm": Interval(7.5, 7.0, 8.0, "ppm")},
            spec_risk={"sulfur_over_10": 0.15},
            confidence="high",
            warnings=[],
        ),
        metrics={"sulfur": 7.5, "yield": 0.85, "energy": 0.6, "severity": 0.2},
        feasible=True,
        infeasible_reason=None,
    )


class _Simulator:
    def get_state(self, ts: datetime) -> pd.DataFrame:
        return pd.DataFrame([{
            "date": pd.Timestamp(ts),
            "pak_sulfur_ppm": 8.2,
            "hydro_T5": 300.0,
            "avt_T55": 348.0,
        }])


class _Tracer:
    def get_trace(self, decision_id: str):
        return [
            SimpleNamespace(
                agent="data",
                duration_ms=12.3,
                input_hash="abc",
                output_summary='{"is_stale": false}',
            )
        ]


def test_recommend_adapts_variants_and_trace():
    v = _variant()
    out = OrchestratorOutput(
        decision_id="d1",
        mode="recommend",
        payload={
            "best_variant": asdict(v),
            "alternatives": [],
            "checks": {
                "combined_spec_risk": {"sulfur_over_10": 0.15},
                "severity_class": "normal",
            },
            "weights": {"safety": 0.5, "quality": 0.3, "yield": 0.15, "energy": 0.05},
        },
        trace_id="d1",
        explanation_text="Рекомендация",
    )

    resp = to_recommendation_response(
        out, datetime(2025, 8, 1, 12, 0), _Simulator(), _Tracer()
    )

    assert resp.decision_id == "d1"
    assert resp.mode == "recommend"
    assert len(resp.variants) == 1

    api_v = resp.variants[0]
    assert api_v.action == {"hydro_T5": 305.0}
    assert api_v.delta == {"hydro_T5": 5.0}
    assert api_v.predicted is not None
    assert api_v.predicted.sulfur is not None
    assert api_v.predicted.sulfur.mean == 7.5
    assert set(api_v.metrics) == {"safety", "yield", "energy", "wear"}
    assert set(resp.default_weights) == {"safety", "yield", "energy", "wear"}
    assert resp.trace and resp.trace[0].agent == "data"
    assert resp.trace[0].output == {"is_stale": False}


def test_refuse_adapts_warning():
    out = OrchestratorOutput(
        decision_id="d2",
        mode="refuse",
        payload={"reason": "данные устарели"},
        trace_id="d2",
        explanation_text="Рекомендация не выдана",
    )

    resp = to_recommendation_response(
        out, datetime(2025, 8, 1, 12, 0), _Simulator(), _Tracer()
    )

    assert resp.mode == "refuse"
    assert resp.variants == []
    assert resp.warnings == ["данные устарели"]
