# Pydantic-контракты обмена между агентами.

from datetime import datetime

from pydantic import BaseModel, Field


class Interval(BaseModel):
    mean: float
    low: float
    high: float
    unit: str


class Prediction(BaseModel):
    predictions: dict[str, Interval] = Field(default_factory=dict)
    spec_risk: dict[str, float] = Field(default_factory=dict)
    confidence: str = "unknown"
    warnings: list[str] = Field(default_factory=list)


class DataAgentOutput(BaseModel):
    timestamp: datetime
    is_stale: bool = False
    has_anomalies: bool = False
    anomaly_score: float = 0.0
    flagged_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReliabilityAgentOutput(BaseModel):
    severity_class: str = "normal"
    severity_score: float = 0.0
    risk_factors: list[str] = Field(default_factory=list)
    limits: dict[str, tuple[float, float]] = Field(default_factory=dict)


class Action(BaseModel):
    changes: dict[str, float]
    expected: Prediction
    metrics: dict[str, float]


class OptimizationAgentOutput(BaseModel):
    variants: list[Action]
    infeasible_reasons: list[str] = Field(default_factory=list)


class OrchestratorOutput(BaseModel):
    decision_id: str
    mode: str
    payload: dict
    trace_id: str
