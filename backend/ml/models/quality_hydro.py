# Прогноз серы, T50, T90, D15 после гидроочистки.
# Таргет - ПАК Mg.Sulfur.Q (10-мин), ЛИМС используется как калибратор.

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ml.feature_engineering import build_hydro_features
from ml.types import Explanation, Interval, QualityPrediction
from ml.vak import (
    VAKCatalog,
    VAKFormula,
    VAKMissingFeatureError,
)


log = logging.getLogger(__name__)

QUANTILES = (0.1, 0.5, 0.9)
MIN_TRAIN_SAMPLES = 100
MAX_LABEL_AGE_HOURS = 0.25
STALE_LIMS_HOURS = 24.0
SULFUR_LIMIT_PPM = 10.0
TARGET_ENVELOPE_MARGIN = 3.0


TARGET_CONFIG: dict[str, dict[str, Any]] = {
    "sulfur_ppm": {
        # ПАК — главный частый таргет согласно data_reference.md.
        "target_column": "pak_sulfur_ppm",
        "age_column": None,
        "calibration_column":
            "lims__гидроочистка__pt2__mg_sulfur",
        "calibration_age_column":
            "lims__гидроочистка__pt2__mg_sulfur_age_h",
        # Формулы серы в справочнике ВАК нет.
        "vak_name": None,
        "unit": "ppm",
        "absolute_range": (0.0, 1000.0),
    },
    "T50": {
        "target_column":
            "lims__гидроочистка__pt2__50_t",
        "age_column":
            "lims__гидроочистка__pt2__50_t_age_h",
        "vak_name": "24-2000:GODT:T50",
        "unit": "C",
        "absolute_range": (0.0, 600.0),
    },
    "T90": {
        "target_column":
            "lims__гидроочистка__pt2__90_t",
        "age_column":
            "lims__гидроочистка__pt2__90_t_age_h",
        "vak_name": "24-2000:GODT:T90",
        "unit": "C",
        "absolute_range": (0.0, 700.0),
    },
    "D15": {
        # ПАК D15 начинается только в марте 2025 года,
        # поэтому на train 2023–2024 используем ЛИМС.
        "target_column":
            "lims__гидроочистка__pt2__d15",
        "age_column":
            "lims__гидроочистка__pt2__d15_age_h",
        "vak_name": "24-2000:GODT:D15",
        "unit": "kg/m3",
        "absolute_range": (500.0, 1200.0),
    },
}

RELEVANT_LIMS_AGE_COLUMNS = frozenset(
    str(config[column])
    for config in TARGET_CONFIG.values()
    for column in ("age_column", "calibration_age_column")
    if config.get(column) is not None
)


class QualityHydroModel:
    def __init__(
        self,
        artifact: dict[str, Any] | None = None,
    ):
        self.artifact = artifact or {}
        self._last_explanation: Explanation | None = None

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        vak_catalog: VAKCatalog,
        *,
        avt_delay_steps: int,
    ) -> "QualityHydroModel":
        """
        Обучает модели квантилей остатка:

            target = VAK baseline + LightGBM residual.

        frame должен содержать только train-период.
        """

        cls._validate_training_frame(frame)

        train = build_hydro_features(
            frame,
            avt_delay_steps=avt_delay_steps,
        )

        feature_columns = cls._select_features(
            train,
            avt_delay_steps=avt_delay_steps,
        )

        if not feature_columns:
            raise ValueError(
                "No numeric hydro features found"
            )

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
            "version": 2,
            "trained": True,
            "avt_delay_steps": avt_delay_steps,
            "feature_columns": feature_columns,
            "feature_q01": feature_q01,
            "feature_q99": feature_q99,
            "models": {},
            "vak_formulas": {},
            "fallback_baselines": {},
            "target_envelopes": {},
            "residual_quantiles": {},
            "n_samples": {},
            "sulfur_calibration_samples": 0,
        }

        for target, config in TARGET_CONFIG.items():
            y, label_mask = cls._make_target(
                train=train,
                target=target,
                config=config,
                artifact=artifact,
            )

            finite_target = y[
                label_mask & np.isfinite(y)
            ]

            if finite_target.empty:
                artifact["n_samples"][target] = 0
                artifact["fallback_baselines"][target] = (
                    float("nan")
                )
                artifact["residual_quantiles"][target] = {
                    quantile: 0.0
                    for quantile in QUANTILES
                }
                continue

            target_envelope = cls._robust_target_envelope(
                finite_target,
                absolute_range=config.get("absolute_range"),
            )
            artifact["target_envelopes"][target] = target_envelope

            # Очевидно ошибочные лабораторные значения (например, нулевые
            # температуры кипения) не должны становиться обучающими метками.
            valid_target = y.between(*target_envelope)
            clean_target = y[label_mask & valid_target & np.isfinite(y)]

            if clean_target.empty:
                artifact["n_samples"][target] = 0
                artifact["fallback_baselines"][target] = float("nan")
                artifact["residual_quantiles"][target] = {
                    quantile: 0.0 for quantile in QUANTILES
                }
                continue

            fallback = float(clean_target.median())
            artifact["fallback_baselines"][target] = fallback

            vak_name = config["vak_name"]
            formula: VAKFormula | None = None

            if vak_name is not None:
                formula = vak_catalog.get(str(vak_name))

            artifact["vak_formulas"][target] = formula

            baseline = cls._evaluate_baseline_series(
                frame=train,
                formula=formula,
                fallback=fallback,
                target_column=str(config["target_column"]),
                valid_range=target_envelope,
            )

            valid = (
                label_mask
                & valid_target
                & y.notna()
                & baseline.notna()
                & np.isfinite(y)
                & np.isfinite(baseline)
            )

            n_samples = int(valid.sum())
            artifact["n_samples"][target] = n_samples

            if n_samples == 0:
                artifact["residual_quantiles"][target] = {
                    quantile: 0.0
                    for quantile in QUANTILES
                }
                continue

            residual = (
                y.loc[valid] - baseline.loc[valid]
            )

            artifact["residual_quantiles"][target] = {
                quantile: float(
                    residual.quantile(quantile)
                )
                for quantile in QUANTILES
            }

            if n_samples < MIN_TRAIN_SAMPLES:
                log.warning(
                    "%s: only %d samples; "
                    "LightGBM is not trained",
                    target,
                    n_samples,
                )
                continue

            x = train.loc[
                valid,
                feature_columns,
            ].copy()

            x["__vak_baseline__"] = (
                baseline.loc[valid].to_numpy()
            )

            models: dict[float, LGBMRegressor] = {}

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
                models[quantile] = model

            artifact["models"][target] = models

            log.info(
                "Trained QualityHydro target=%s "
                "samples=%d features=%d",
                target,
                n_samples,
                len(feature_columns),
            )

        return cls(artifact=artifact)

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "QualityHydroModel":
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

    def predict(
        self,
        state: pd.DataFrame,
    ) -> QualityPrediction:
        self._validate_state(state)

        if not self.artifact.get("trained"):
            result = self._untrained_prediction()
            log.info(
                "QualityHydroModel.predict input=%s result=%s",
                state.to_dict(orient="records")[0],
                asdict(result),
            )
            return result

        predictions: dict[str, Interval] = {}
        warnings: list[str] = []

        missing_ratio, outside_ratio = (
            self._state_quality(state)
        )
        stale_lims = self._stale_lims_columns(state)

        if missing_ratio > 0:
            warnings.append(
                f"Missing {missing_ratio:.1%} "
                "of model features"
            )

        if outside_ratio > 0.1:
            warnings.append(
                f"{outside_ratio:.1%} of features "
                "are outside the historical range"
            )

        if stale_lims:
            warnings.append(
                "LIMS values older than 24 hours: "
                + ", ".join(stale_lims)
            )

        for target, config in TARGET_CONFIG.items():
            interval, target_warnings = (
                self._predict_target(
                    state=state,
                    target=target,
                    unit=str(config["unit"]),
                )
            )
            predictions[target] = interval
            warnings.extend(target_warnings)

        sulfur_risk = self._sulfur_spec_risk(
            predictions["sulfur_ppm"]
        )

        confidence = self._calculate_confidence(
            missing_ratio=missing_ratio,
            outside_ratio=outside_ratio,
            has_stale_lims=bool(stale_lims),
        )

        result = QualityPrediction(
            predictions=predictions,
            spec_risk={
                "sulfur_over_10": sulfur_risk,
            },
            confidence=confidence,  # type: ignore[arg-type]
            warnings=list(dict.fromkeys(warnings)),
        )

        self._last_explanation = (
            self._explain_sulfur(state)
        )

        log.info(
            "QualityHydroModel.predict input=%s result=%s",
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
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
            ):
                raise TypeError(
                    f"Action value for {raw_tag!r} must be numeric"
                )

            value = float(raw_value)

            if not math.isfinite(value):
                raise ValueError(
                    f"Action value for {raw_tag!r} must be finite"
                )

            candidates = (
                raw_tag,
                f"hydro_{raw_tag}",
                f"avt_{raw_tag}",
            )

            column = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate in changed_state.columns
                ),
                None,
            )

            if column is None:
                raise KeyError(
                    f"Unknown process action tag: {raw_tag}"
                )

            if not column.startswith(("hydro_", "avt_")):
                raise KeyError(
                    f"Tag {raw_tag!r} does not belong "
                    "to AVT or hydro installation"
                )

            changed_state.loc[:, column] = value

        result = self.predict(changed_state)
        log.info(
            "QualityHydroModel.predict_after_action "
            "input=%s action=%s result=%s",
            state.to_dict(orient="records")[0],
            action,
            asdict(result),
        )
        return result

    def explain(
        self,
        state: pd.DataFrame,
    ) -> Explanation:
        self._validate_state(state)

        if not self.artifact.get("trained"):
            result = Explanation(
                top_features=[],
                base_value=float("nan"),
            )
            log.info(
                "QualityHydroModel.explain input=%s result=%s",
                state.to_dict(orient="records")[0],
                asdict(result),
            )
            return result

        self._last_explanation = (
            self._explain_sulfur(state)
        )
        log.info(
            "QualityHydroModel.explain input=%s result=%s",
            state.to_dict(orient="records")[0],
            asdict(self._last_explanation),
        )
        return self._last_explanation

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

    @staticmethod
    def _select_features(
        frame: pd.DataFrame,
        *,
        avt_delay_steps: int,
    ) -> list[str]:
        result: list[str] = []

        avt_delay_prefix = (
            f"avt_lag_{avt_delay_steps}__"
        )

        for column in frame.columns:
            allowed = (
                column.startswith("hydro_")
                or column.startswith(avt_delay_prefix)
                or column in RELEVANT_LIMS_AGE_COLUMNS
            )

            if (
                allowed
                and pd.api.types.is_numeric_dtype(frame[column])
                and frame[column].notna().any()
                and frame[column].nunique(dropna=True) > 1
            ):
                result.append(column)

        return result

    @classmethod
    def _make_target(
        cls,
        *,
        train: pd.DataFrame,
        target: str,
        config: dict[str, Any],
        artifact: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        target_column = str(config["target_column"])

        if target_column not in train.columns:
            empty = pd.Series(
                np.nan,
                index=train.index,
                dtype="float64",
            )
            return empty, empty.notna()

        if target == "sulfur_ppm":
            return cls._calibrated_sulfur_target(
                train=train,
                config=config,
                artifact=artifact,
            )

        y = pd.to_numeric(
            train[target_column],
            errors="coerce",
        )

        age_column = str(config["age_column"])

        if age_column not in train.columns:
            raise ValueError(
                f"Age column is missing for {target}: "
                f"{age_column}"
            )

        age = pd.to_numeric(
            train[age_column],
            errors="coerce",
        )

        fresh_sample = (
            y.notna()
            & age.notna()
            & age.le(MAX_LABEL_AGE_HOURS)
            & (
                age.shift().isna()
                | age.diff().lt(0)
                | age.eq(0)
            )
        )

        return y, fresh_sample

    @staticmethod
    def _calibrated_sulfur_target(
        *,
        train: pd.DataFrame,
        config: dict[str, Any],
        artifact: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        """
        Калибрует потоковый ПАК по последнему доступному анализу ЛИМС.

        Поправка появляется только в момент получения ЛИМС и затем
        распространяется вперёд. Будущий анализ не используется.
        """

        pak = pd.to_numeric(
            train[str(config["target_column"])],
            errors="coerce",
        )

        lab_column = str(
            config["calibration_column"]
        )
        age_column = str(
            config["calibration_age_column"]
        )

        if (
            lab_column not in train.columns
            or age_column not in train.columns
        ):
            artifact["sulfur_calibration_samples"] = 0
            return pak, pak.notna()

        lab = pd.to_numeric(
            train[lab_column],
            errors="coerce",
        )
        age = pd.to_numeric(
            train[age_column],
            errors="coerce",
        )

        fresh_lab = (
            lab.notna()
            & pak.notna()
            & age.notna()
            & age.le(MAX_LABEL_AGE_HOURS)
            & (
                age.shift().isna()
                | age.diff().lt(0)
                | age.eq(0)
            )
        )

        correction = pd.Series(
            np.nan,
            index=train.index,
            dtype="float64",
        )
        correction.loc[fresh_lab] = (
            lab.loc[fresh_lab] - pak.loc[fresh_lab]
        )

        # ЛИМС имеет приоритет над ПАК, пока результат актуален. После 24
        # часов поправка считается устаревшей и больше не применяется.
        correction = correction.ffill()
        correction = correction.where(
            age.le(STALE_LIMS_HOURS),
            0.0,
        ).fillna(0.0)

        artifact["sulfur_calibration_samples"] = int(
            fresh_lab.sum()
        )

        calibrated = pak + correction
        return calibrated, calibrated.notna()

    @staticmethod
    def _evaluate_baseline_series(
        *,
        frame: pd.DataFrame,
        formula: VAKFormula | None,
        fallback: float,
        target_column: str,
        valid_range: tuple[float, float] | None = None,
    ) -> pd.Series:
        if formula is None:
            return pd.Series(
                fallback,
                index=frame.index,
                dtype="float64",
            )

        baseline_frame = frame.copy()

        # Формула D15 содержит ссылку на ЛИМС D15.
        # При обучении на свежем лабораторном результате используем
        # предыдущее известное значение, иначе таргет попадёт в baseline.
        if target_column in formula.required_columns:
            baseline_frame[target_column] = (
                baseline_frame[target_column].shift(1)
            )

        try:
            baseline = formula.evaluate(baseline_frame)
        except VAKMissingFeatureError:
            return pd.Series(
                fallback,
                index=frame.index,
                dtype="float64",
            )

        baseline = baseline.replace([np.inf, -np.inf], np.nan)

        if valid_range is not None:
            baseline = baseline.where(
                baseline.between(*valid_range)
            )

        return baseline.fillna(fallback)

    @staticmethod
    def _evaluate_baseline_row(
        *,
        state: pd.DataFrame,
        formula: VAKFormula | None,
        fallback: float,
        valid_range: tuple[float, float] | None = None,
    ) -> tuple[float, bool]:
        if formula is None:
            return float(fallback), True

        try:
            value = float(
                formula.evaluate(state).iloc[0]
            )
        except (
            VAKMissingFeatureError,
            KeyError,
            ValueError,
        ):
            return float(fallback), True

        if not np.isfinite(value):
            return float(fallback), True

        if valid_range is not None:
            minimum, maximum = valid_range
            if not minimum <= value <= maximum:
                return float(fallback), True

        return value, False

    def _predict_target(
        self,
        *,
        state: pd.DataFrame,
        target: str,
        unit: str,
    ) -> tuple[Interval, list[str]]:
        warnings: list[str] = []

        fallback = self.artifact[
            "fallback_baselines"
        ].get(target, float("nan"))

        formula = self.artifact[
            "vak_formulas"
        ].get(target)
        valid_range = self.artifact.get(
            "target_envelopes", {}
        ).get(target)

        baseline, used_fallback = (
            self._evaluate_baseline_row(
                state=state,
                formula=formula,
                fallback=fallback,
                valid_range=valid_range,
            )
        )

        if formula is None:
            warnings.append(
                f"{target}: VAK formula is absent; "
                "train median is used as baseline"
            )
        elif used_fallback:
            warnings.append(
                f"{target}: VAK could not be calculated; "
                "fallback is used"
            )

        models = self.artifact["models"].get(target)

        if models:
            x = self._prepare_features(
                state,
                baseline,
            )
            residuals = {
                quantile: float(
                    models[quantile].predict(x)[0]
                )
                for quantile in QUANTILES
            }
        else:
            residuals = self.artifact[
                "residual_quantiles"
            ].get(
                target,
                {
                    quantile: 0.0
                    for quantile in QUANTILES
                },
            )

            warnings.append(
                f"{target}: LightGBM is unavailable "
                f"({self.artifact['n_samples'].get(target, 0)} "
                "samples)"
            )

        values = sorted(
            baseline + residuals[quantile]
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
        x = state.reindex(
            columns=self.artifact["feature_columns"]
        ).copy()

        x = x.apply(
            pd.to_numeric,
            errors="coerce",
        )
        x["__vak_baseline__"] = baseline
        return x

    def _explain_sulfur(
        self,
        state: pd.DataFrame,
    ) -> Explanation:
        target = "sulfur_ppm"
        models = self.artifact["models"].get(target)

        fallback = self.artifact[
            "fallback_baselines"
        ].get(target, float("nan"))
        formula = self.artifact[
            "vak_formulas"
        ].get(target)
        valid_range = self.artifact.get(
            "target_envelopes", {}
        ).get(target)

        baseline, _ = self._evaluate_baseline_row(
            state=state,
            formula=formula,
            fallback=fallback,
            valid_range=valid_range,
        )

        if not models:
            return Explanation(
                top_features=[],
                base_value=float(baseline),
            )

        model = models[0.5]
        x = self._prepare_features(state, baseline)

        contributions = model.booster_.predict(
            x,
            pred_contrib=True,
        )[0]

        pairs = list(
            zip(
                x.columns,
                contributions[:-1],
            )
        )
        pairs.sort(
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )

        return Explanation(
            top_features=[
                (name, float(value))
                for name, value in pairs[:10]
            ],
            base_value=float(
                baseline + contributions[-1]
            ),
        )

    @staticmethod
    def _sulfur_spec_risk(
        interval: Interval,
    ) -> float:
        """
        Оценивает вероятность sulfur > 10 ppm по q10/q50/q90.

        Межквантильный диапазон переводится в стандартное
        отклонение приближённого нормального распределения.
        """

        values = [
            interval.low,
            interval.mean,
            interval.high,
        ]

        if not all(np.isfinite(values)):
            return 1.0

        sigma = (
            interval.high - interval.low
        ) / (2.0 * 1.2815515655446004)

        if sigma <= 1e-9:
            return float(
                interval.mean > SULFUR_LIMIT_PPM
            )

        z = (
            SULFUR_LIMIT_PPM - interval.mean
        ) / sigma

        probability_below = 0.5 * (
            1.0 + math.erf(z / math.sqrt(2.0))
        )

        return float(
            np.clip(
                1.0 - probability_below,
                0.0,
                1.0,
            )
        )

    def _state_quality(
        self,
        state: pd.DataFrame,
    ) -> tuple[float, float]:
        columns = self.artifact["feature_columns"]

        row = state.reindex(
            columns=columns
        ).iloc[0]

        numeric = pd.to_numeric(
            row,
            errors="coerce",
        )

        missing_ratio = float(
            numeric.isna().mean()
        )

        observed = numeric.notna()

        if not observed.any():
            return missing_ratio, 1.0

        q01 = pd.Series(
            self.artifact["feature_q01"]
        )
        q99 = pd.Series(
            self.artifact["feature_q99"]
        )

        outside = (
            (numeric[observed] < q01[observed])
            | (numeric[observed] > q99[observed])
        )

        return missing_ratio, float(outside.mean())

    @staticmethod
    def _stale_lims_columns(
        state: pd.DataFrame,
    ) -> list[str]:
        stale: list[str] = []

        for column in sorted(RELEVANT_LIMS_AGE_COLUMNS):
            if column not in state.columns:
                continue

            value = pd.to_numeric(
                state[column],
                errors="coerce",
            ).iloc[0]

            if (
                pd.notna(value)
                and float(value) > STALE_LIMS_HOURS
            ):
                stale.append(column)

        return stale

    @staticmethod
    def _robust_target_envelope(
        values: pd.Series,
        absolute_range: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Строит допустимый диапазон таргета только по train-данным.

        Границы нужны не для подрезания итогового прогноза, а для защиты
        от некорректных лабораторных меток и неприменимых формул ВАК.
        """

        numeric = pd.to_numeric(values, errors="coerce")
        numeric = numeric[np.isfinite(numeric)]

        if numeric.empty:
            if absolute_range is not None:
                return absolute_range
            return float("-inf"), float("inf")

        low = float(numeric.quantile(0.01))
        high = float(numeric.quantile(0.99))
        spread = high - low

        if not np.isfinite(spread) or spread <= 0:
            center = float(numeric.median())
            margin = max(abs(center) * 0.05, 1.0)
            lower, upper = center - margin, center + margin
        else:
            margin = TARGET_ENVELOPE_MARGIN * spread
            lower, upper = low - margin, high + margin

        if absolute_range is not None:
            absolute_minimum, absolute_maximum = absolute_range
            lower = max(lower, absolute_minimum)
            upper = min(upper, absolute_maximum)

        if lower >= upper:
            raise ValueError("Cannot build a valid target envelope")

        return lower, upper

    def _calculate_confidence(
        self,
        *,
        missing_ratio: float,
        outside_ratio: float,
        has_stale_lims: bool,
    ) -> str:
        counts = self.artifact.get(
            "n_samples",
            {},
        )

        insufficient = any(
            counts.get(target, 0)
            < MIN_TRAIN_SAMPLES
            for target in TARGET_CONFIG
        )

        if (
            not insufficient
            and not has_stale_lims
            and missing_ratio <= 0.05
            and outside_ratio <= 0.10
        ):
            return "high"

        if (
            not insufficient
            and missing_ratio <= 0.20
            and outside_ratio <= 0.25
        ):
            return "medium"

        return "low"

    def _untrained_prediction(
        self,
    ) -> QualityPrediction:
        result = QualityPrediction(
            predictions={
                target: Interval(
                    mean=float("nan"),
                    low=float("nan"),
                    high=float("nan"),
                    unit=str(config["unit"]),
                )
                for target, config
                in TARGET_CONFIG.items()
            },
            spec_risk={
                "sulfur_over_10": 1.0,
            },
            confidence="low",
            warnings=[
                "QualityHydroModel is not trained yet"
            ],
        )

        self._last_explanation = Explanation(
            top_features=[],
            base_value=float("nan"),
        )
        return result
