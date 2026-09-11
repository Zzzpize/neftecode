"""ReliabilityAgent: оценка надёжности режима (прокси-метрики).

Реальный режим опирается на ``feature_registry.yaml``: для управляемых тегов
сверяет текущее значение с историческим диапазоном ``[range_min, range_max]``
и возвращает ``limits`` — безопасные диапазоны управляемых тегов (они затем
попадают в ``OptimizationConstraints.controllable_ranges``).

Без ``registry_path`` работает в mock-режиме (для Фазы 1 и демо-сценариев).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agents.schemas import ReliabilityAgentOutput

NORMAL_THRESHOLD = 0.3
ELEVATED_THRESHOLD = 0.6


class ReliabilityAgent:
    def __init__(self, registry_path: Path | str | None = None):
        self.registry_path = registry_path
        self._registry: dict | None = None

        # Переопределения для mock-режима и сценариев.
        self.mock_severity_class = "normal"
        self.mock_severity_score = 0.0
        self.mock_risk_factors: list[str] = []
        self.mock_limits: dict[str, tuple[float, float]] = {}

    def _load_registry(self) -> dict:
        if self._registry is None and self.registry_path is not None:
            with Path(self.registry_path).open(encoding="utf-8") as f:
                self._registry = yaml.safe_load(f)
        return self._registry or {}

    def _compute(self, state: pd.DataFrame) -> ReliabilityAgentOutput:
        registry = self._load_registry()
        controllable = {
            tag: cfg
            for tag, cfg in registry.items()
            if isinstance(cfg, dict) and cfg.get("controllable")
        }

        row = state.iloc[0] if state is not None and not state.empty else None

        limits: dict[str, tuple[float, float]] = {}
        risk_factors: list[str] = []
        deviations: list[float] = []

        for tag, cfg in controllable.items():
            range_min = cfg.get("range_min")
            range_max = cfg.get("range_max")
            if range_min is None or range_max is None:
                continue

            low = float(range_min)
            high = float(range_max)

            if row is None or tag not in row.index:
                continue

            value = pd.to_numeric(row[tag], errors="coerce")
            if pd.isna(value):
                continue
            value = float(value)

            span = high - low
            if span <= 0:
                continue

            if value < low:
                deviations.append((low - value) / span)
                risk_factors.append(f"{tag} ниже нормы: {value:.2f} < {low:.2f}")
                # Текущее значение вне исторического диапазона — не отдаём его
                # жёсткий диапазон оптимизатору (иначе find_pareto упадёт
                # «no valid search range»).
                continue
            if value > high:
                deviations.append((value - high) / span)
                risk_factors.append(f"{tag} выше нормы: {value:.2f} > {high:.2f}")
                continue

            limits[tag] = (low, high)

        severity_score = (
            min(1.0, sum(deviations) / len(deviations))
            if deviations else 0.0
        )

        if severity_score < NORMAL_THRESHOLD:
            severity_class = "normal"
        elif severity_score < ELEVATED_THRESHOLD:
            severity_class = "elevated"
        else:
            severity_class = "heavy"

        return ReliabilityAgentOutput(
            severity_class=severity_class,
            severity_score=round(severity_score, 3),
            risk_factors=risk_factors,
            limits=limits,
        )

    async def assess(self, state: pd.DataFrame) -> ReliabilityAgentOutput:
        if self.registry_path is not None:
            return self._compute(state)
        return ReliabilityAgentOutput(
            severity_class=self.mock_severity_class,
            severity_score=self.mock_severity_score,
            risk_factors=list(self.mock_risk_factors),
            limits=dict(self.mock_limits),
        )
