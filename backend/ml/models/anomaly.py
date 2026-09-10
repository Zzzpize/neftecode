# Аномалии: Isolation Forest + rolling z-score, детектор застывших сигналов и out-of-envelope.

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ml.feature_engineering import (
    ANOMALY_WINDOWS,
    build_anomaly_features,
)
from ml.types import AnomalyReport


log = logging.getLogger(__name__)

DEFAULT_CONTAMINATION = 0.01
DEFAULT_Z_THRESHOLD = 4.0
DEFAULT_STALE_THRESHOLD_HOURS = 1.0


class AnomalyDetector:
    def __init__(
        self,
        artifact: dict[str, Any] | None = None,
    ):
        self.artifact = artifact or {}

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        normal_mask: pd.Series | None = None,
        contamination: float = DEFAULT_CONTAMINATION,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        stale_threshold_hours: float = (
            DEFAULT_STALE_THRESHOLD_HOURS
        ),
    ) -> "AnomalyDetector":
        """
        Обучает Isolation Forest на train-периоде.

        normal_mask позволяет исключить известные инциденты.
        True означает нормальный период.
        """

        cls._validate_training_frame(frame)

        if not 0.0 < contamination < 0.5:
            raise ValueError(
                "contamination must be between 0 and 0.5"
            )

        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")

        if stale_threshold_hours <= 0:
            raise ValueError(
                "stale_threshold_hours must be positive"
            )

        working = frame.copy()

        if normal_mask is None:
            working["__normal_period__"] = True
            log.warning(
                "Known incident mask is not provided; "
                "all train rows are treated as normal"
            )
        else:
            aligned_mask = normal_mask.reindex(
                working.index,
                fill_value=False,
            )
            working["__normal_period__"] = (
                aligned_mask.astype(bool)
            )

        if feature_columns is None:
            feature_columns = cls._select_features(working)

        feature_columns = [
            column
            for column in feature_columns
            if column in working.columns
            and pd.api.types.is_numeric_dtype(working[column])
            and working[column].notna().any()
            and working[column].nunique(dropna=True) > 1
        ]

        if not feature_columns:
            raise ValueError(
                "No numeric AVT or hydro features found"
            )

        prepared = build_anomaly_features(
            working,
            feature_columns,
        )

        normal_rows = prepared["__normal_period__"].fillna(
            False
        )

        train = prepared.loc[
            normal_rows,
            feature_columns,
        ].apply(pd.to_numeric, errors="coerce")

        if len(train) < 100:
            raise ValueError(
                "At least 100 normal rows are required"
            )

        medians = train.median()
        train_filled = train.fillna(medians)

        scaler = StandardScaler()
        normalized = scaler.fit_transform(train_filled)

        model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(normalized)

        decision = model.decision_function(normalized)

        positive_scale = max(
            float(np.quantile(decision, 0.99)),
            1e-9,
        )
        negative_scale = max(
            abs(float(np.quantile(decision, 0.01))),
            1e-9,
        )

        q001 = train.quantile(0.001)
        q999 = train.quantile(0.999)
        envelope_width = (q999 - q001).replace(0.0, 1.0)

        normal_prepared = prepared.loc[normal_rows]
        change_rate = (
            train.diff().abs().gt(1e-9).mean()
        )
        stale_feature_columns = (
            change_rate[change_rate >= 0.01]
            .index
            .tolist()
        )
        stale_thresholds: dict[str, float] = {}

        for feature in stale_feature_columns:
            stale_column = f"anomaly_stale_h__{feature}"
            values = pd.to_numeric(
                normal_prepared[stale_column],
                errors="coerce",
            ).dropna()
            historical_limit = (
                float(values.quantile(0.9999))
                if not values.empty
                else 0.0
            )
            stale_thresholds[feature] = max(
                float(stale_threshold_hours),
                historical_limit,
            )

        rolling_z_thresholds: dict[str, float] = {}
        for column in normal_prepared.columns:
            if not column.startswith("anomaly_z__"):
                continue
            values = pd.to_numeric(
                normal_prepared[column],
                errors="coerce",
            ).abs().dropna()
            historical_limit = (
                float(values.quantile(0.9999))
                if not values.empty
                else 0.0
            )
            rolling_z_thresholds[column] = max(
                float(z_threshold),
                historical_limit,
            )

        artifact: dict[str, Any] = {
            "version": 1,
            "trained": True,
            "feature_columns": feature_columns,
            "medians": medians.astype(float).to_dict(),
            "envelope_low": (
                q001 - 0.1 * envelope_width
            ).astype(float).to_dict(),
            "envelope_high": (
                q999 + 0.1 * envelope_width
            ).astype(float).to_dict(),
            "scaler": scaler,
            "model": model,
            "positive_score_scale": positive_scale,
            "negative_score_scale": negative_scale,
            "z_threshold": float(z_threshold),
            "stale_threshold_hours": float(
                stale_threshold_hours
            ),
            "stale_feature_columns": stale_feature_columns,
            "stale_thresholds": stale_thresholds,
            "rolling_z_thresholds": rolling_z_thresholds,
            "contamination": float(contamination),
            "n_samples": int(len(train)),
        }

        return cls(artifact=artifact)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        with Path(path).open("rb") as file:
            artifact = pickle.load(file)

        return cls(artifact=artifact)

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open("wb") as file:
            pickle.dump(self.artifact, file)

    def score(
        self,
        state: pd.DataFrame,
    ) -> AnomalyReport:
        self._validate_state(state)

        # необученная модель возвращает нейтральный результат.
        if not self.artifact.get("trained"):
            return AnomalyReport(
                is_anomaly=False,
                anomaly_score=0.0,
                flagged_tags=[],
                is_out_of_envelope=False,
                stale_tags=[],
            )

        feature_columns: list[str] = self.artifact[
            "feature_columns"
        ]

        medians: dict[str, float] = self.artifact[
            "medians"
        ]

        current = pd.DataFrame(
            {
                column: [
                    pd.to_numeric(
                        state[column],
                        errors="coerce",
                    ).iloc[0]
                    if column in state.columns
                    else np.nan
                ]
                for column in feature_columns
            }
        )

        missing_tags = [
            column
            for column in feature_columns
            if pd.isna(current.at[0, column])
        ]

        filled = current.fillna(medians)

        scaler: StandardScaler = self.artifact["scaler"]
        model: IsolationForest = self.artifact["model"]

        normalized = scaler.transform(filled)
        decision = float(
            model.decision_function(normalized)[0]
        )
        isolation_flag = (
            int(model.predict(normalized)[0]) == -1
        )

        anomaly_score = self._normalize_score(decision)

        envelope_tags = self._find_envelope_violations(
            current
        )
        rolling_tags = self._find_rolling_anomalies(
            state
        )
        stale_tags = self._find_stale_tags(state)

        flagged_tags = sorted(
            set(
                missing_tags
                + envelope_tags
                + rolling_tags
            )
        )

        is_out_of_envelope = bool(envelope_tags)

        is_anomaly = bool(
            isolation_flag
            or rolling_tags
            or envelope_tags
            or stale_tags
            or missing_tags
        )

        # Явные локальные проблемы должны иметь заметный score,
        # даже если многомерная Isolation Forest их сгладила.
        if rolling_tags or envelope_tags:
            anomaly_score = max(anomaly_score, 0.75)

        if stale_tags:
            anomaly_score = max(anomaly_score, 0.7)

        if missing_tags:
            anomaly_score = max(anomaly_score, 0.6)

        report = AnomalyReport(
            is_anomaly=is_anomaly,
            anomaly_score=float(
                np.clip(anomaly_score, 0.0, 1.0)
            ),
            flagged_tags=flagged_tags,
            is_out_of_envelope=is_out_of_envelope,
            stale_tags=stale_tags,
        )

        log.info(
            "AnomalyDetector.score input=%s result=%s",
            current.to_dict(orient="records")[0],
            report,
        )

        return report

    def _normalize_score(
        self,
        decision: float,
    ) -> float:
        """
        Преобразует decision_function Isolation Forest в 0..1.

        Граница решения decision=0 соответствует score=0.5.
        """

        if decision >= 0:
            scale = self.artifact[
                "positive_score_scale"
            ]
            result = 0.5 * (
                1.0 - min(decision / scale, 1.0)
            )
        else:
            scale = self.artifact[
                "negative_score_scale"
            ]
            result = 0.5 + 0.5 * min(
                abs(decision) / scale,
                1.0,
            )

        return float(np.clip(result, 0.0, 1.0))

    def _find_envelope_violations(
        self,
        current: pd.DataFrame,
    ) -> list[str]:
        low = self.artifact["envelope_low"]
        high = self.artifact["envelope_high"]

        flagged: list[str] = []

        for column in self.artifact["feature_columns"]:
            value = current.at[0, column]

            if pd.isna(value):
                continue

            if value < low[column] or value > high[column]:
                flagged.append(column)

        return flagged

    def _find_rolling_anomalies(
        self,
        state: pd.DataFrame,
    ) -> list[str]:
        flagged: set[str] = set()

        for feature in self.artifact["feature_columns"]:
            for window_name in ANOMALY_WINDOWS:
                column = (
                    f"anomaly_z__{window_name}__{feature}"
                )

                if column not in state.columns:
                    continue

                threshold = float(
                    self.artifact.get(
                        "rolling_z_thresholds",
                        {},
                    ).get(
                        column,
                        self.artifact["z_threshold"],
                    )
                )

                value = pd.to_numeric(
                    state[column],
                    errors="coerce",
                ).iloc[0]

                if (
                    pd.notna(value)
                    and math.isfinite(float(value))
                    and abs(float(value)) >= threshold
                ):
                    flagged.add(feature)

        return sorted(flagged)

    def _find_stale_tags(
        self,
        state: pd.DataFrame,
    ) -> list[str]:
        stale: list[str] = []

        for feature in self.artifact.get(
            "stale_feature_columns",
            self.artifact["feature_columns"],
        ):
            column = f"anomaly_stale_h__{feature}"

            if column not in state.columns:
                continue

            threshold = float(
                self.artifact.get(
                    "stale_thresholds",
                    {},
                ).get(
                    feature,
                    self.artifact["stale_threshold_hours"],
                )
            )

            value = pd.to_numeric(
                state[column],
                errors="coerce",
            ).iloc[0]

            if (
                pd.notna(value)
                and math.isfinite(float(value))
                and float(value) >= threshold
            ):
                stale.append(feature)

        return sorted(stale)

    @staticmethod
    def _select_features(
        frame: pd.DataFrame,
    ) -> list[str]:
        return [
            column
            for column in frame.columns
            if (
                column.startswith("avt_")
                or column.startswith("hydro_")
            )
            and pd.api.types.is_numeric_dtype(frame[column])
            and not column.endswith("_age_h")
        ]

    @staticmethod
    def _validate_training_frame(
        frame: pd.DataFrame,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "frame must be a pandas DataFrame"
            )

        if frame.empty:
            raise ValueError("frame must not be empty")

        if "date" not in frame.columns:
            raise ValueError(
                "frame must contain the date column"
            )

    @staticmethod
    def _validate_state(
        state: pd.DataFrame,
    ) -> None:
        if not isinstance(state, pd.DataFrame):
            raise TypeError(
                "state must be a pandas DataFrame"
            )

        if len(state) != 1:
            raise ValueError(
                "state must contain exactly one row"
            )
