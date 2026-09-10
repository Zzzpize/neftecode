"""Pydantic-схемы для HTTP-ответов /recommend, /ask.

ML-модели используют dataclass-типы из ml.types. Тут — обёртки для API,
которые пойдут во фронт как JSON. При подключении реального оптимизатора
из ml.optimizer нужен адаптер dataclass -> pydantic.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Interval(BaseModel):
    mean: float
    low: float
    high: float
    unit: str


class QualityPrediction(BaseModel):
    sulfur: Interval | None = None
    t50: Interval | None = None
    t90: Interval | None = None
    d15: Interval | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    spec_risk: dict[str, float] = Field(default_factory=dict)


class Variant(BaseModel):
    id: str
    action: dict[str, float]
    delta: dict[str, float]
    metrics: dict[str, float]
    predicted: QualityPrediction | None = None
    feasible: bool = True
    infeasible_reason: str | None = None


class TraceStep(BaseModel):
    agent: str
    duration_ms: float
    input_summary: str
    output: dict


class RecommendationResponse(BaseModel):
    decision_id: str
    mode: Literal["recommend", "silent", "refuse"]
    timestamp: str
    variants: list[Variant] = Field(default_factory=list)
    default_weights: dict[str, float] = Field(default_factory=dict)
    trace: list[TraceStep] = Field(default_factory=list)
    explanation_text: str = ""
    warnings: list[str] = Field(default_factory=list)
