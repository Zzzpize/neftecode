"""QualityAgent: прогноз качества АВТ / гидроочистки / блендинга.

Реальная обёртка над ``QualityAVTModel`` + ``QualityHydroModel``. Без моделей
работает в mock-режиме (для Фазы 1 и демо-сценариев).
"""
from __future__ import annotations

import pandas as pd

from agents.schemas import QualityAgentOutput
from ml.types import Interval, QualityPrediction


class QualityAgent:
    def __init__(self, avt=None, hydro=None, blending=None):
        self.avt = avt
        self.hydro = hydro
        self.blending = blending

        # Переопределение суммарного риска для mock-режима и сценариев.
        self.mock_combined_spec_risk: dict[str, float] = {}

    @staticmethod
    def _merge_risks(*risk_maps: dict[str, float]) -> dict[str, float]:
        """Объединяет риски АВТ и гидроочистки, беря максимум по каждому ключу."""
        merged: dict[str, float] = {}
        for risk_map in risk_maps:
            for key, value in risk_map.items():
                merged[key] = max(merged.get(key, 0.0), float(value))
        return merged

    @staticmethod
    def _read_sulfur(state: pd.DataFrame) -> float:
        if state is not None and not state.empty and "pak_sulfur_ppm" in state.columns:
            value = state.iloc[0]["pak_sulfur_ppm"]
            if pd.notna(value):
                return float(value)
        return 8.0

    @staticmethod
    def _mock_prediction(sulfur: float) -> QualityPrediction:
        return QualityPrediction(
            predictions={
                "sulfur_ppm": Interval(
                    mean=sulfur,
                    low=sulfur - 0.5,
                    high=sulfur + 0.5,
                    unit="ppm",
                ),
            },
            spec_risk={
                "sulfur_over_10": round(max(0.0, (sulfur - 6.0) / 8.0), 2),
            },
            confidence="high",
            warnings=[],
        )

    async def forecast(self, state: pd.DataFrame) -> QualityAgentOutput:
        # Реальный путь: модели АВТ и гидроочистки.
        if self.avt is not None and self.hydro is not None:
            avt = self.avt.predict(state)
            hydro = self.hydro.predict(state)
            return QualityAgentOutput(
                avt=avt,
                hydro=hydro,
                blended=None,
                combined_spec_risk=self._merge_risks(avt.spec_risk, hydro.spec_risk),
            )

        # Mock-режим.
        sulfur = self._read_sulfur(state)
        combined = dict(self.mock_combined_spec_risk) or {
            "sulfur_over_10": round(max(0.0, (sulfur - 6.0) / 8.0), 2),
        }
        return QualityAgentOutput(
            avt=self._mock_prediction(sulfur),
            hydro=self._mock_prediction(sulfur),
            blended=None,
            combined_spec_risk=combined,
        )

    async def forecast_after_action(
        self,
        state: pd.DataFrame,
        action: dict[str, float],
    ) -> QualityAgentOutput:
        # Реальный путь.
        if self.avt is not None and self.hydro is not None:
            avt = self.avt.predict_after_action(state, action)
            hydro = self.hydro.predict_after_action(state, action)
            return QualityAgentOutput(
                avt=avt,
                hydro=hydro,
                blended=None,
                combined_spec_risk=self._merge_risks(avt.spec_risk, hydro.spec_risk),
            )

        # Mock-режим.
        sulfur = max(0.0, self._read_sulfur(state) - 0.5)
        return QualityAgentOutput(
            avt=self._mock_prediction(sulfur),
            hydro=self._mock_prediction(sulfur),
            blended=None,
            combined_spec_risk={
                "sulfur_over_10": round(max(0.0, (sulfur - 6.0) / 8.0), 2),
            },
        )
