# Прогноз T50, T90, D15, ПТФ для фракции 240-350 на выходе АВТ.
# База - формулы ВАК, поверх LightGBM на резидуалах, квантильная регрессия.
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ml.types import Explanation, Interval, QualityPrediction


class QualityAVTModel:
    def __init__(self, artifact: dict | None = None):
        self.artifact = artifact or {}
        self._last_explanation: Explanation | None = None

    @classmethod
    def load(cls, path: str) -> "QualityAVTModel":
        with Path(path).open("rb") as f:
            artifact = pickle.load(f)
        return cls(artifact=artifact)

    def predict(self, state: pd.DataFrame) -> QualityPrediction:
        self._validate_state(state)

        # Временный baseline-ответ, потом сюда встанут ВАК + LightGBM.
        prediction = QualityPrediction(
            predictions={
                "T50": Interval(mean=float("nan"), low=float("nan"), high=float("nan"), unit="C"),
                "T90": Interval(mean=float("nan"), low=float("nan"), high=float("nan"), unit="C"),
                "D15": Interval(mean=float("nan"), low=float("nan"), high=float("nan"), unit="kg/m3"),
                "CFPP": Interval(mean=float("nan"), low=float("nan"), high=float("nan"), unit="C"),
            },
            spec_risk={},
            confidence="low",
            warnings=["QualityAVTModel is not trained yet"],
        )
        self._last_explanation = Explanation(top_features=[], base_value=float("nan"))
        return prediction


    def predict_after_action(
        self,
        state: pd.DataFrame,
        action: dict[str, float],
    ) -> QualityPrediction:
        changed_state = state.copy()
        for tag, value in action.items():
            if tag in changed_state.columns:
                changed_state.loc[:, tag] = value
        return self.predict(changed_state)

    def explain(self, state: pd.DataFrame) -> Explanation:
        self._validate_state(state)
        if self._last_explanation is None:
            return Explanation(top_features=[], base_value=float("nan"))
        return self._last_explanation

    @staticmethod
    def _validate_state(state: pd.DataFrame) -> None:
        if not isinstance(state, pd.DataFrame):
            raise TypeError("state must be a pandas DataFrame")
        if len(state) != 1:
            raise ValueError("state must contain exactly one row")
