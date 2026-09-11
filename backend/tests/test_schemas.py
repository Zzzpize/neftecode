from datetime import datetime

import pytest
from pydantic import ValidationError

from agents.schemas import (
    AskInput,
    AskOutput,
    DataAgentInput,
    DataAgentOutput,
    OptimizationAgentOutput,
    OrchestratorOutput,
    QualityAgentOutput,
    ReliabilityAgentOutput,
)
from ml.types import Interval, QualityPrediction, Variant


def make_prediction(sulfur: float = 8.2) -> QualityPrediction:
    return QualityPrediction(
        predictions={
            "sulfur_ppm": Interval(
                mean=sulfur,
                low=sulfur - 0.5,
                high=sulfur + 0.5,
                unit="ppm",
            ),
        },
        spec_risk={"sulfur_over_10": 0.27},
        confidence="high",
        warnings=[],
    )


def make_variant(sulfur: float = 6.4, feasible: bool = True) -> Variant:
    return Variant(
        action={"T55": 348.0, "F30": 44.5},
        expected=make_prediction(sulfur),
        metrics={
            "sulfur": sulfur,
            "yield": 0.85,
            "energy": 1.11,
            "severity": 0.3,
        },
        feasible=feasible,
        infeasible_reason=None if feasible else "hard_constraint_violation",
    )


def test_data_agent_contract():
    ts = datetime(2025, 8, 1, 12, 0, 0)

    assert DataAgentInput(timestamp=ts).timestamp == ts

    out = DataAgentOutput(
        timestamp=ts,
        is_stale=False,
        has_anomalies=True,
        anomaly_score=0.9,
        flagged_tags=["hydro_T5"],
        is_out_of_envelope=True,
        stale_tags=["pak_sulfur_ppm"],
        warnings=["anomaly"],
    )
    assert out.has_anomalies is True
    assert "hydro_T5" in out.flagged_tags
    assert out.stale_tags == ["pak_sulfur_ppm"]


def test_quality_agent_output_accepts_ml_prediction():
    out = QualityAgentOutput(
        avt=make_prediction(8.0),
        hydro=make_prediction(6.0),
        blended=None,
        combined_spec_risk={"sulfur_over_10": 0.27},
    )

    assert isinstance(out.avt, QualityPrediction)
    assert out.hydro.predictions["sulfur_ppm"].mean == pytest.approx(6.0)
    assert out.combined_spec_risk["sulfur_over_10"] == pytest.approx(0.27)


def test_reliability_agent_output_severity_literal():
    out = ReliabilityAgentOutput(
        severity_class="elevated",
        severity_score=0.6,
        risk_factors=["high_temperature"],
        limits={"hydro_T5": (290.0, 330.0)},
    )

    assert out.severity_class == "elevated"
    assert out.limits["hydro_T5"] == (290.0, 330.0)

    with pytest.raises(ValidationError):
        ReliabilityAgentOutput(severity_class="critical")


def test_optimization_agent_output_uses_ml_variant():
    out = OptimizationAgentOutput(
        variants=[make_variant(), make_variant(sulfur=6.8)],
        infeasible_reasons=[],
    )

    assert len(out.variants) == 2
    assert isinstance(out.variants[0], Variant)
    assert out.variants[0].metrics["yield"] == pytest.approx(0.85)


def test_orchestrator_output_mode_literal():
    out = OrchestratorOutput(
        decision_id="d123",
        mode="recommend",
        payload={"best_variant": {}},
        trace_id="t456",
        explanation_text="Повысить температуру.",
    )

    assert out.mode == "recommend"
    assert out.explanation_text

    with pytest.raises(ValidationError):
        OrchestratorOutput(decision_id="d", mode="unknown", trace_id="t")


def test_ask_contract():
    assert AskInput(decision_id="d123", question="Почему?").decision_id == "d123"

    out = AskOutput(answer="Потому что сера выше нормы.", tool_calls_used=2)
    assert out.tool_calls_used == 2
