# Парето-оптимизатор через Optuna. Метрики: сера, выход, энергия, тяжесть режима.

from __future__ import annotations

import pandas as pd

from ml.models.blending import BlendingModel
from ml.models.quality_avt import QualityAVTModel
from ml.models.quality_hydro import QualityHydroModel
from ml.types import OptimizationConstraints, Variant


class ParetoOptimizer:
    def __init__(
        self,
        quality_avt: QualityAVTModel,
        quality_hydro: QualityHydroModel,
        blending: BlendingModel,
    ):
        self.quality_avt = quality_avt
        self.quality_hydro = quality_hydro
        self.blending = blending

    def find_pareto(
        self,
        state: pd.DataFrame,
        constraints: OptimizationConstraints,
        n_variants: int = 10,
    ) -> list[Variant]:
        self._validate_state(state)

        # Временная заглушка: правильный тип, но без оптимизации.
        return []

    @staticmethod
    def _validate_state(state: pd.DataFrame) -> None:
        if not isinstance(state, pd.DataFrame):
            raise TypeError("state must be a pandas DataFrame")
        if len(state) != 1:
            raise ValueError("state must contain exactly one row")