"""OptimizationAgent: поиск вариантов управления.

Реальная обёртка над ``ParetoOptimizer``. Без оптимизатора работает в
mock-режиме (для Фазы 1 и демо-сценариев).
"""
from __future__ import annotations

import pandas as pd

from agents.schemas import OptimizationAgentOutput
from ml.types import Interval, OptimizationConstraints, QualityPrediction, Variant


def _prediction(sulfur: float) -> QualityPrediction:
    return QualityPrediction(
        predictions={
            "sulfur_ppm": Interval(
                mean=sulfur,
                low=sulfur - 0.4,
                high=sulfur + 0.4,
                unit="ppm",
            ),
        },
        spec_risk={"sulfur_over_10": round(max(0.0, (sulfur - 6.0) / 8.0), 2)},
        confidence="medium",
        warnings=[],
    )


class OptimizationAgent:
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.mock_variants: list[Variant] | None = None

    async def find_variants(
        self,
        state: pd.DataFrame,
        constraints: OptimizationConstraints,
    ) -> OptimizationAgentOutput:
        if self.mock_variants is not None:
            return OptimizationAgentOutput(
                variants=list(self.mock_variants),
                infeasible_reasons=[],
            )

        # Реальный путь: Парето-оптимизатор.
        if self.optimizer is not None:
            variants = self.optimizer.find_pareto(state, constraints, n_variants=10)
            infeasible_reasons = [
                v.infeasible_reason
                for v in variants
                if not v.feasible and v.infeasible_reason
            ]
            return OptimizationAgentOutput(
                variants=variants,
                infeasible_reasons=infeasible_reasons,
            )

        # Mock-режим. Метрики в реальном формате ML (sulfur/yield/energy/severity).
        base = 8.0
        if state is not None and not state.empty and "pak_sulfur_ppm" in state.columns:
            value = state.iloc[0]["pak_sulfur_ppm"]
            if pd.notna(value):
                base = float(value)

        profiles = [
            ({"hydro_T5": 295.0}, base - 1.0, {"sulfur": base - 1.0, "yield": 0.85, "energy": 0.60, "severity": 0.25}),
            ({"hydro_T5": 300.0}, base - 0.5, {"sulfur": base - 0.5, "yield": 0.80, "energy": 0.55, "severity": 0.20}),
            ({"avt_T55": 350.0}, base - 0.2, {"sulfur": base - 0.2, "yield": 0.90, "energy": 0.50, "severity": 0.30}),
        ]
        variants = [
            Variant(
                action=action,
                expected=_prediction(sulfur),
                metrics=metrics,
                feasible=True,
                infeasible_reason=None,
            )
            for action, sulfur, metrics in profiles
        ]
        return OptimizationAgentOutput(variants=variants, infeasible_reasons=[])
