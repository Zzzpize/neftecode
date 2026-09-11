"""DataAgent: проверка свежести данных и аномалий.

Реальная обёртка над ``AnomalyDetector`` + ``Simulator``. Без зависимостей
работает в mock-режиме (для Фазы 1 и демо-сценариев).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from agents.schemas import DataAgentInput, DataAgentOutput


def _mock_state(timestamp: datetime) -> pd.DataFrame:
    """Минимальный срез процесса для прогона на заглушках."""
    return pd.DataFrame([{
        "date": pd.Timestamp(timestamp),
        "pak_sulfur_ppm": 8.0,
        "avt_T55": 348.0,
        "hydro_T5": 300.0,
    }])


class DataAgent:
    def __init__(self, anomaly=None, simulator=None, stale_threshold_hours: float = 24.0):
        self.anomaly = anomaly
        self.simulator = simulator
        self.stale_threshold_hours = stale_threshold_hours

        # Переопределения для mock-режима и сценариев (Фаза 1).
        self.mock_is_stale = False
        self.mock_has_anomalies = False
        self.mock_anomaly_score = 0.0
        self.mock_is_out_of_envelope = False
        self.mock_flagged_tags: list[str] = []
        self.mock_stale_tags: list[str] = []
        self.mock_warnings: list[str] = []

    def get_state(self, timestamp: datetime) -> pd.DataFrame:
        """Срез процесса на момент времени (реальный — из Simulator)."""
        if self.simulator is not None:
            return self.simulator.get_state(timestamp)
        return _mock_state(timestamp)

    def _find_stale_lims(self, state: pd.DataFrame) -> list[str]:
        """Возвращает серосодержащие ЛИМС старше ``stale_threshold_hours``.

        Проверяем только серосодержащие лабораторные показатели: именно они
        калибруют прогноз серы, вокруг которого строится рекомендация. Остальные
        ЛИМС (T50/T90/D15/CFPP и т.п.) редкие и не блокируют рекомендацию — их
        свежесть модели учитывают самостоятельно через ``confidence``.
        """
        stale: list[str] = []
        for column in state.columns:
            if not (column.startswith("lims__") and column.endswith("_age_h")):
                continue
            if "mg_sulfur" not in column:
                continue
            value = pd.to_numeric(state[column], errors="coerce").iloc[0]
            if pd.notna(value) and float(value) > self.stale_threshold_hours:
                stale.append(column)
        return stale

    async def check(self, inp: DataAgentInput) -> DataAgentOutput:
        # Реальный путь: AnomalyDetector + Simulator.
        if self.anomaly is not None and self.simulator is not None:
            state = self.simulator.get_state(inp.timestamp)
            report = self.anomaly.score(state)

            stale_lims = self._find_stale_lims(state)
            is_stale = bool(stale_lims)

            warnings: list[str] = []
            if report.is_anomaly:
                warnings.append(f"Обнаружена аномалия (score={report.anomaly_score:.2f})")
            if report.is_out_of_envelope:
                warnings.append("Состояние вне исторического конверта режимов")
            if report.stale_tags:
                warnings.append("Застывшие теги: " + ", ".join(report.stale_tags))
            if is_stale:
                warnings.append("Устаревшие лабораторные значения: " + ", ".join(stale_lims))

            return DataAgentOutput(
                timestamp=inp.timestamp,
                is_stale=is_stale,
                has_anomalies=report.is_anomaly,
                anomaly_score=report.anomaly_score,
                flagged_tags=list(report.flagged_tags),
                is_out_of_envelope=report.is_out_of_envelope,
                stale_tags=list(report.stale_tags),
                warnings=warnings,
            )

        # Mock-режим (Фаза 1 / демо-сценарии).
        return DataAgentOutput(
            timestamp=inp.timestamp,
            is_stale=self.mock_is_stale,
            has_anomalies=self.mock_has_anomalies,
            anomaly_score=self.mock_anomaly_score,
            flagged_tags=list(self.mock_flagged_tags),
            is_out_of_envelope=self.mock_is_out_of_envelope,
            stale_tags=list(self.mock_stale_tags),
            warnings=list(self.mock_warnings),
        )
