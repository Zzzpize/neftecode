"""Модель качества дизельной фракции 240–350 на выходе АВТ."""

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ml.types import Explanation, Interval, QualityPrediction
from ml.vak import VAKCatalog, VAKFormula, VAKMissingFeatureError


log = logging.getLogger(__name__)


TARGET_CONFIG: dict[str, dict[str, str | None]] = {
    "T50": {
        "target_column": "lims__авт__pt1__50_t",
        "age_column": "lims__авт__pt1__50_t_age_h",
        "vak_name": "AVT6:240-350:T50",
        "unit": "C",
    },
    "T90": {
        "target_column": "lims__авт__pt1__90_t",
        "age_column": "lims__авт__pt1__90_t_age_h",
        # В справочнике ВАК отдельной формулы T90 нет.
        "vak_name": None,
        "unit": "C",
    },
    "D15": {
        "target_column": "lims__авт__pt1__d15",
        "age_column": "lims__авт__pt1__d15_age_h",
        "vak_name": "AVT6:240-350:D15",
        "unit": "kg/m3",
    },
    "CFPP": {
        "target_column": "lims__авт__pt1__cfpp",
        "age_column": "lims__авт__pt1__cfpp_age_h",
        "vak_name": "AVT6:240-350:CFPP",
        "unit": "C",
    },
}

QUANTILES = (0.1, 0.5, 0.9)
MIN_TRAIN_SAMPLES = 100

# Лабораторное значение считается новой меткой только рядом
# с моментом отбора пробы. Это не позволяет размножать одно значение
# ЛИМС на тысячи последующих 10-минутных строк.
MAX_LABEL_AGE_HOURS = 0.25


class QualityAVTModel:
    def __init__(self, artifact: dict[str, Any] | None = None):
        self.artifact = artifact or {}
        self._last_explanation: Explanation | None = None

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        vak_catalog: VAKCatalog,
    ) -> "QualityAVTModel":
        """Обучает модель на заранее выделенном train-периоде."""

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("frame must not be empty")
        if "date" not in frame.columns:
            raise ValueError("frame must contain the date column")

        train = frame.sort_values("date").reset_index(drop=True).copy()

        # В baseline-версии используем только телеметрию АВТ.
        # ЛИМС-колонки сюда не входят, иначе текущий таргет попадёт
        # во вход модели и возникнет утечка.
        feature_columns = [
            column
            for column in train.columns
            if column.startswith("avt_")
            and pd.api.types.is_numeric_dtype(train[column])
            and train[column].notna().any()
            and train[column].nunique(dropna=True) > 1
        ]

        if not feature_columns:
            raise ValueError("No numeric AVT features found")

        feature_q01 = (
            train[feature_columns]
            .quantile(0.01)
            .astype(float)
            .to_dict()
        )
        feature_q99 = (
            train[feature_columns]
            .quantile(0.99)
            .astype(float)
            .to_dict()
        )

        artifact: dict[str, Any] = {
            "version": 1,
            "trained": True,
            "feature_columns": feature_columns,
            "feature_q01": feature_q01,
            "feature_q99": feature_q99,
            "models": {},
            "vak_formulas": {},
            "fallback_baselines": {},
            "residual_quantiles": {},
            "n_samples": {},
        }

        for target, config in TARGET_CONFIG.items():
            target_column = str(config["target_column"])
            age_column = str(config["age_column"])
            vak_name = config["vak_name"]

            if target_column not in train.columns:
                log.warning(
                    "Target %s skipped: column %s is missing",
                    target,
                    target_column,
                )
                artifact["n_samples"][target] = 0
                continue

            y = pd.to_numeric(train[target_column], errors="coerce")

            if age_column not in train.columns:
                raise ValueError(
                    f"Age column is missing for {target}: {age_column}"
                )

            label_age = pd.to_numeric(train[age_column], errors="coerce")

            # time_join протягивает последнее значение ЛИМС вперёд.
            # Поэтому берём только строки, в которых возраст анализа
            # сбросился: именно там появился новый лабораторный результат.
            is_new_lab_sample = (
                y.notna()
                & label_age.notna()
                & label_age.le(MAX_LABEL_AGE_HOURS)
                & (
                    label_age.shift().isna()
                    | label_age.diff().lt(0)
                    | label_age.eq(0)
                )
            )

            fallback = float(y.loc[is_new_lab_sample].median())
            artifact["fallback_baselines"][target] = fallback

            formula: VAKFormula | None = None
            if vak_name is not None:
                formula = vak_catalog.get(str(vak_name))

            artifact["vak_formulas"][target] = formula

            baseline = cls._evaluate_baseline_series(
                frame=train,
                formula=formula,
                fallback=fallback,
            )

            valid = (
                is_new_lab_sample
                & baseline.notna()
                & np.isfinite(y)
                & np.isfinite(baseline)
            )

            n_samples = int(valid.sum())
            artifact["n_samples"][target] = n_samples

            if n_samples == 0:
                artifact["residual_quantiles"][target] = {
                    quantile: 0.0 for quantile in QUANTILES
                }
                continue

            residual = y.loc[valid] - baseline.loc[valid]

            # Даже когда данных недостаточно для LightGBM,
            # сохраняем эмпирический интервал ошибки baseline.
            artifact["residual_quantiles"][target] = {
                quantile: float(residual.quantile(quantile))
                for quantile in QUANTILES
            }

            if n_samples < MIN_TRAIN_SAMPLES:
                log.warning(
                    "%s: only %d samples; LightGBM is not trained",
                    target,
                    n_samples,
                )
                continue

            x = train.loc[valid, feature_columns].copy()

            # ВАК добавляется как отдельная фича. Сама модель учится
            # предсказывать только residual = факт - ВАК.
            x["__vak_baseline__"] = baseline.loc[valid].to_numpy()

            target_models: dict[float, LGBMRegressor] = {}

            for quantile in QUANTILES:
                model = LGBMRegressor(
                    objective="quantile",
                    alpha=quantile,
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=-1,
                )
                model.fit(x, residual)
                target_models[quantile] = model

            artifact["models"][target] = target_models

            log.info(
                "Trained QualityAVT target=%s samples=%d features=%d",
                target,
                n_samples,
                len(feature_columns),
            )

        return cls(artifact=artifact)

    @classmethod
    def load(cls, path: str) -> "QualityAVTModel":
        with Path(path).open("rb") as file:
            artifact = pickle.load(file)

        return cls(artifact=artifact)

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as file:
            pickle.dump(self.artifact, file)

    def predict(self, state: pd.DataFrame) -> QualityPrediction:
        self._validate_state(state)

        if not self.artifact.get("trained"):
            return self._untrained_prediction()

        warnings: list[str] = []
        predictions: dict[str, Interval] = {}

        missing_ratio, outside_ratio = self._state_quality(state)

        if missing_ratio > 0:
            warnings.append(
                f"Missing {missing_ratio:.1%} of model features"
            )

        if outside_ratio > 0.1:
            warnings.append(
                f"{outside_ratio:.1%} of features are outside "
                "the historical range"
            )

        for target, config in TARGET_CONFIG.items():
            interval, target_warnings = self._predict_target(
                state=state,
                target=target,
                unit=str(config["unit"]),
            )
            predictions[target] = interval
            warnings.extend(target_warnings)

        confidence = self._calculate_confidence(
            missing_ratio=missing_ratio,
            outside_ratio=outside_ratio,
        )

        result = QualityPrediction(
            predictions=predictions,
            spec_risk={},
            confidence=confidence,      #type: ignore
            warnings=list(dict.fromkeys(warnings)),
        )

        self._last_explanation = self._explain_t50(state)

        log.info(
            "QualityAVTModel.predict input=%s result=%s",
            state.to_dict(orient="records")[0],
            asdict(result),
        )

        return result

    def predict_after_action(
        self,
        state: pd.DataFrame,
        action: dict[str, float],
    ) -> QualityPrediction:
        self._validate_state(state)

        changed_state = state.copy()

        for raw_tag, raw_value in action.items():
            if not isinstance(raw_value, (int, float)):
                raise TypeError(
                    f"Action value for {raw_tag!r} must be numeric"
                )

            # Поддерживаем оба варианта:
            # {"avt_T55": 350} и короткий {"T55": 350}.
            if raw_tag in changed_state.columns:
                column = raw_tag
            elif f"avt_{raw_tag}" in changed_state.columns:
                column = f"avt_{raw_tag}"
            else:
                raise KeyError(f"Unknown AVT action tag: {raw_tag}")

            changed_state.loc[:, column] = float(raw_value)

        return self.predict(changed_state)

    def explain(self, state: pd.DataFrame) -> Explanation:
        self._validate_state(state)

        if not self.artifact.get("trained"):
            return Explanation(
                top_features=[],
                base_value=float("nan"),
            )

        self._last_explanation = self._explain_t50(state)
        return self._last_explanation

    def _predict_target(
        self,
        state: pd.DataFrame,
        target: str,
        unit: str,
    ) -> tuple[Interval, list[str]]:
        warnings: list[str] = []

        fallback = self.artifact["fallback_baselines"].get(
            target,
            float("nan"),
        )
        formula = self.artifact["vak_formulas"].get(target)

        baseline, used_fallback = self._evaluate_baseline_row(
            state=state,
            formula=formula,
            fallback=fallback,
        )

        if formula is None:
            warnings.append(
                f"{target}: VAK formula is absent; train median is used "
                "as baseline"
            )
        elif used_fallback:
            warnings.append(
                f"{target}: VAK could not be calculated; fallback is used"
            )

        target_models = self.artifact["models"].get(target)

        if target_models:
            x = self._prepare_features(state, baseline)

            residual_predictions = {
                quantile: float(model.predict(x)[0])
                for quantile, model in target_models.items()
            }
        else:
            residual_predictions = self.artifact[
                "residual_quantiles"
            ].get(
                target,
                {quantile: 0.0 for quantile in QUANTILES},
            )

            n_samples = self.artifact["n_samples"].get(target, 0)
            warnings.append(
                f"{target}: LightGBM is unavailable "
                f"({n_samples} laboratory samples)"
            )

        values = sorted(
            baseline + residual_predictions[quantile]
            for quantile in QUANTILES
        )

        return (
            Interval(
                mean=float(values[1]),
                low=float(values[0]),
                high=float(values[2]),
                unit=unit,
            ),
            warnings,
        )

    def _prepare_features(
        self,
        state: pd.DataFrame,
        baseline: float,
    ) -> pd.DataFrame:
        feature_columns = self.artifact["feature_columns"]

        # reindex добавляет отсутствующие колонки как NaN.
        # LightGBM умеет работать с пропущенными значениями.
        x = state.reindex(columns=feature_columns).copy()
        x = x.apply(pd.to_numeric, errors="coerce")
        x["__vak_baseline__"] = baseline

        return x

    def _explain_t50(self, state: pd.DataFrame) -> Explanation:
        """Объясняет центральный прогноз T50.

        Текущий API Explanation не содержит имени таргета, поэтому
        в baseline-версии explain формально относится к T50.
        """

        target = "T50"
        models = self.artifact["models"].get(target)

        fallback = self.artifact["fallback_baselines"].get(
            target,
            float("nan"),
        )
        formula = self.artifact["vak_formulas"].get(target)
        baseline, _ = self._evaluate_baseline_row(
            state,
            formula,
            fallback,
        )

        if not models:
            return Explanation(
                top_features=[],
                base_value=float(baseline),
            )

        model = models[0.5]
        x = self._prepare_features(state, baseline)

        # LightGBM возвращает SHAP contributions:
        # по одному значению на фичу и последний элемент — expected value.
        contributions = model.booster_.predict(
            x,
            pred_contrib=True,
        )[0]

        feature_names = list(x.columns)
        pairs = list(zip(feature_names, contributions[:-1]))
        pairs.sort(key=lambda item: abs(float(item[1])), reverse=True)

        return Explanation(
            top_features=[
                (feature, float(value))
                for feature, value in pairs[:10]
            ],
            base_value=float(baseline + contributions[-1]),
        )

    def _state_quality(
        self,
        state: pd.DataFrame,
    ) -> tuple[float, float]:
        features = self.artifact["feature_columns"]
        row = state.reindex(columns=features).iloc[0]
        numeric = pd.to_numeric(row, errors="coerce")

        missing_ratio = float(numeric.isna().mean())

        observed = numeric.notna()
        if not observed.any():
            return missing_ratio, 1.0

        q01 = pd.Series(self.artifact["feature_q01"])
        q99 = pd.Series(self.artifact["feature_q99"])

        outside = (
            (numeric[observed] < q01[observed])
            | (numeric[observed] > q99[observed])
        )

        return missing_ratio, float(outside.mean())

    def _calculate_confidence(
        self,
        missing_ratio: float,
        outside_ratio: float,
    ) -> str:
        sample_counts = self.artifact.get("n_samples", {})

        # Если хотя бы один таргет не набрал 100 анализов,
        # общий ответ честно получает низкую уверенность.
        insufficient_targets = any(
            sample_counts.get(target, 0) < MIN_TRAIN_SAMPLES
            for target in TARGET_CONFIG
        )

        if (
            not insufficient_targets
            and missing_ratio <= 0.05
            and outside_ratio <= 0.10
        ):
            return "high"

        if (
            missing_ratio <= 0.20
            and outside_ratio <= 0.25
            and not insufficient_targets
        ):
            return "medium"

        return "low"

    @staticmethod
    def _evaluate_baseline_series(
        frame: pd.DataFrame,
        formula: VAKFormula | None,
        fallback: float,
    ) -> pd.Series:
        if formula is None:
            return pd.Series(
                fallback,
                index=frame.index,
                dtype="float64",
            )

        try:
            result = formula.evaluate(frame)
        except VAKMissingFeatureError:
            return pd.Series(
                fallback,
                index=frame.index,
                dtype="float64",
            )

        return result.fillna(fallback)

    @staticmethod
    def _evaluate_baseline_row(
        state: pd.DataFrame,
        formula: VAKFormula | None,
        fallback: float,
    ) -> tuple[float, bool]:
        if formula is None:
            return float(fallback), True

        try:
            value = float(formula.evaluate(state).iloc[0])
        except (VAKMissingFeatureError, KeyError, ValueError):
            return float(fallback), True

        if not np.isfinite(value):
            return float(fallback), True

        return value, False

    @staticmethod
    def _validate_state(state: pd.DataFrame) -> None:
        if not isinstance(state, pd.DataFrame):
            raise TypeError("state must be a pandas DataFrame")
        if len(state) != 1:
            raise ValueError("state must contain exactly one row")

    def _untrained_prediction(self) -> QualityPrediction:
        result = QualityPrediction(
            predictions={
                target: Interval(
                    mean=float("nan"),
                    low=float("nan"),
                    high=float("nan"),
                    unit=str(config["unit"]),
                )
                for target, config in TARGET_CONFIG.items()
            },
            spec_risk={},
            confidence="low",
            warnings=["QualityAVTModel is not trained yet"],
        )

        self._last_explanation = Explanation(
            top_features=[],
            base_value=float("nan"),
        )
        return result