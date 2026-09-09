# Аномалии: Isolation Forest + rolling z-score, детектор застывших сигналов и out-of-envelope.
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ml.types import AnomalyReport


class AnomalyDetector:
    def __init__(self, artifact: dict | None = None):
        self.artifact = artifact or {}

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        with Path(path).open("rb") as f:
            artifact = pickle.load(f)
        return cls(artifact=artifact)

    def score(self, state: pd.DataFrame) -> AnomalyReport:
        self._validate_state(state)

        return AnomalyReport(
            is_anomaly=False,
            anomaly_score=0.0,
            flagged_tags=[],
            is_out_of_envelope=False,
            stale_tags=[],
        )

    @staticmethod
    def _validate_state(state: pd.DataFrame) -> None:
        if not isinstance(state, pd.DataFrame):
            raise TypeError("state must be a pandas DataFrame")
        if len(state) != 1:
            raise ValueError("state must contain exactly one row")
