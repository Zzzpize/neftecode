# Pydantic-контракты обмена между агентами.

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ml.types import QualityPrediction, Variant


class DataAgentInput(BaseModel):
    timestamp: datetime


class DataAgentOutput(BaseModel):
    timestamp: datetime
    is_stale: bool = False
    has_anomalies: bool = False
    anomaly_score: float = 0.0
    flagged_tags: list[str] = Field(default_factory=list)
    is_out_of_envelope: bool = False
    stale_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QualityAgentOutput(BaseModel):
    avt: QualityPrediction
    hydro: QualityPrediction
    blended: QualityPrediction | None = None
    combined_spec_risk: dict[str, float] = Field(default_factory=dict)


class ReliabilityAgentOutput(BaseModel):
    severity_class: Literal["normal", "elevated", "heavy"] = "normal"
    severity_score: float = 0.0
    risk_factors: list[str] = Field(default_factory=list)
    limits: dict[str, tuple[float, float]] = Field(default_factory=dict)


class OptimizationAgentOutput(BaseModel):
    variants: list[Variant] = Field(default_factory=list)
    infeasible_reasons: list[str] = Field(default_factory=list)


class OrchestratorOutput(BaseModel):
    decision_id: str
    mode: Literal["recommend", "silent", "refuse"]
    payload: dict = Field(default_factory=dict)
    trace_id: str
    explanation_text: str = ""


class AskInput(BaseModel):
    decision_id: str
    question: str


class AskOutput(BaseModel):
    answer: str
    tool_calls_used: int = 0
