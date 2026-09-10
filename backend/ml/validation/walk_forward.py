# Walk-forward без утечки времени.
# Train 2023-01..2024-12, Val 2025-01..2025-06, Test 2025-07..2026-08.

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ml.feature_engineering import build_anomaly_features
from ml.models.anomaly import AnomalyDetector
from ml.models.quality_avt import (
    MAX_LABEL_AGE_HOURS,
    TARGET_CONFIG as AVT_TARGET_CONFIG,
    QualityAVTModel,
)
from ml.models.quality_hydro import (
    TARGET_CONFIG as HYDRO_TARGET_CONFIG,
    QualityHydroModel,
)
from ml.types import Interval


TRAIN_START = pd.Timestamp("2023-01-01")
VALIDATION_START = pd.Timestamp("2025-01-01")
TEST_START = pd.Timestamp("2025-07-01")
TEST_END_EXCLUSIVE = pd.Timestamp("2026-08-08")

SULFUR_LIMIT_PPM = 10.0
SULFUR_RISK_THRESHOLD = 0.5
TARGET_INTERVAL_COVERAGE = 0.90
ONLINE_CALIBRATION_WINDOW = 30
SULFUR_VALIDATION_RECALL_TARGET = 0.95
MIN_TRAIN_SAMPLES = 100


QualityModel = QualityAVTModel | QualityHydroModel


def split_by_time(
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Делит данные строго по времени.

    Соответствует разделу «Валидация» tz_ml.md.
    Строки не перемешиваются.
    """

    if "date" not in frame.columns:
        raise ValueError("frame must contain the date column")

    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(
        prepared["date"],
        errors="raise",
    )
    prepared = prepared.sort_values("date").reset_index(drop=True)

    return {
        "train": prepared.loc[
            prepared["date"].ge(TRAIN_START)
            & prepared["date"].lt(VALIDATION_START)
        ].copy(),
        "validation": prepared.loc[
            prepared["date"].ge(VALIDATION_START)
            & prepared["date"].lt(TEST_START)
        ].copy(),
        "test": prepared.loc[
            prepared["date"].ge(TEST_START)
            & prepared["date"].lt(TEST_END_EXCLUSIVE)
        ].copy(),
    }


def _fresh_label_mask(
    frame: pd.DataFrame,
    *,
    target_column: str,
    age_column: str | None,
) -> pd.Series:
    """
    Для ПАК берёт все доступные измерения.

    Для ЛИМС берёт только момент появления нового анализа.
    Значения, протянутые merge_asof, повторно не считаются.
    """

    if target_column not in frame.columns:
        return pd.Series(False, index=frame.index)

    target = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    )

    if age_column is None:
        return target.notna() & np.isfinite(target)

    if age_column not in frame.columns:
        return pd.Series(False, index=frame.index)

    age = pd.to_numeric(
        frame[age_column],
        errors="coerce",
    )

    new_sample = (
        age.shift().isna()
        | age.diff().lt(0)
        | age.eq(0)
    )

    return (
        target.notna()
        & np.isfinite(target)
        & age.notna()
        & age.le(MAX_LABEL_AGE_HOURS)
        & new_sample
    )


def _baseline_series(
    model: QualityModel,
    frame: pd.DataFrame,
    *,
    target: str,
    config: dict[str, Any],
) -> pd.Series:
    """
    Получает прогноз чистой ВАК-формулы без LightGBM.

    Это нужно для обязательного сравнения baseline_mae из tz_ml.md.
    """

    fallback = float(
        model.artifact["fallback_baselines"].get(
            target,
            float("nan"),
        )
    )
    formula = model.artifact["vak_formulas"].get(target)

    if isinstance(model, QualityAVTModel):
        return model._evaluate_baseline_series(
            frame=frame,
            formula=formula,
            fallback=fallback,
        )

    valid_range = model.artifact.get(
        "target_envelopes",
        {},
    ).get(target)

    return model._evaluate_baseline_series(
        frame=frame,
        formula=formula,
        fallback=fallback,
        target_column=str(config["target_column"]),
        valid_range=valid_range,
    )

def _batch_prediction_arrays(
    model: QualityModel,
    frame: pd.DataFrame,
    *,
    target: str,
    config: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Пакетно рассчитывает baseline и три границы прогноза.

    В отличие от публичного predict, не рассчитывает SHAP
    для каждой строки валидации.
    """

    baseline = (
        pd.to_numeric(
            _baseline_series(
                model,
                frame,
                target=target,
                config=config,
            ),
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    feature_columns = model.artifact["feature_columns"]

    features = (
        frame.reindex(columns=feature_columns)
        .apply(pd.to_numeric, errors="coerce")
    )
    features["__vak_baseline__"] = baseline

    target_models = model.artifact["models"].get(target)

    if target_models:
        quantiles = sorted(target_models)

        residual_predictions = np.column_stack(
            [
                np.asarray(
                    target_models[quantile].predict(features),
                    dtype=float,
                )
                for quantile in quantiles
            ]
        )
    else:
        quantiles = (0.1, 0.5, 0.9)
        stored_quantiles = model.artifact[
            "residual_quantiles"
        ].get(target, {})

        residual_predictions = np.column_stack(
            [
                np.full(
                    len(frame),
                    float(stored_quantiles.get(quantile, 0.0)),
                )
                for quantile in quantiles
            ]
        )

    predicted_values = (
        residual_predictions
        + baseline[:, np.newaxis]
    )

    # Повторяет поведение predict(): low <= mean <= high.
    predicted_values.sort(axis=1)

    low = predicted_values[:, 0]
    predicted = predicted_values[:, 1]
    high = predicted_values[:, 2]

    calibrated_half_width = model.artifact.get(
        "interval_half_width",
        {},
    ).get(target)

    if calibrated_half_width is not None:
        half_width = float(calibrated_half_width)
        low = predicted - half_width
        high = predicted + half_width

    return baseline, predicted, low, high

def _empty_metrics() -> dict[str, Any]:
    return {
        "n_samples": 0,
        "mae": None,
        "rmse": None,
        "coverage_90": None,
        "baseline_mae": None,
        "ml_beats_baseline": None,
    }


def _physical_target_mask(
    actual: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    valid = np.isfinite(actual)
    absolute_range = config.get("absolute_range")

    if absolute_range is not None:
        minimum, maximum = absolute_range
        valid &= actual >= float(minimum)
        valid &= actual <= float(maximum)

    return valid


def _conformal_radius(
    errors: np.ndarray,
    *,
    coverage: float = TARGET_INTERVAL_COVERAGE,
) -> float:
    finite = np.asarray(errors, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return 0.0

    level = min(
        1.0,
        np.ceil((finite.size + 1) * coverage)
        / finite.size,
    )
    return float(
        np.quantile(finite, level, method="higher")
    )


def _apply_online_interval_calibration(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    history: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Строит интервал только по ошибкам, известным до текущей метки."""

    working_history = list(history)
    low = np.empty_like(predicted, dtype=float)
    high = np.empty_like(predicted, dtype=float)

    for index, (actual_value, predicted_value) in enumerate(
        zip(actual, predicted, strict=True)
    ):
        recent = np.asarray(
            working_history[-ONLINE_CALIBRATION_WINDOW:],
            dtype=float,
        )
        radius = _conformal_radius(recent)
        low[index] = predicted_value - radius
        high[index] = predicted_value + radius

        # Обновление происходит после построения прогноза для текущей точки.
        working_history.append(
            abs(float(actual_value) - float(predicted_value))
        )

    return low, high


def _regression_metrics(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & np.isfinite(low)
        & np.isfinite(high)
    )

    if not valid.any():
        return _empty_metrics()

    y_true = actual[valid]
    y_pred = predicted[valid]
    y_low = low[valid]
    y_high = high[valid]

    baseline_valid = valid & np.isfinite(baseline)

    baseline_mae: float | None = None
    if baseline_valid.any():
        baseline_mae = float(
            mean_absolute_error(
                actual[baseline_valid],
                baseline[baseline_valid],
            )
        )

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(
        mean_squared_error(y_true, y_pred) ** 0.5
    )
    coverage = float(
        np.mean(
            (y_true >= y_low)
            & (y_true <= y_high)
        )
    )

    return {
        "n_samples": int(valid.sum()),
        "mae": mae,
        "rmse": rmse,
        "coverage_90": coverage,
        "baseline_mae": baseline_mae,
        "ml_beats_baseline": (
            mae < baseline_mae
            if baseline_mae is not None
            else None
        ),
    }


def evaluate_quality_model(
    model: QualityModel,
    frame: pd.DataFrame,
    *,
    target_config: dict[str, dict[str, Any]],
    online_calibration_history: dict[str, list[float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Считает метрики на полном временном периоде пакетно.
    """

    evaluation = (
        frame.sort_values("date")
        .reset_index(drop=True)
    )

    result: dict[str, dict[str, Any]] = {}

    for target, config in target_config.items():
        age_column = config.get("age_column")

        label_mask = _fresh_label_mask(
            evaluation,
            target_column=str(config["target_column"]),
            age_column=(
                str(age_column)
                if age_column is not None
                else None
            ),
        )

        positions = np.flatnonzero(
            label_mask.to_numpy()
        )

        if len(positions) == 0:
            result[target] = _empty_metrics()
            continue

        target_column = str(config["target_column"])
        actual = (
            pd.to_numeric(
                evaluation.iloc[positions][target_column],
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )
        physically_valid = _physical_target_mask(
            actual,
            config,
        )

        training_samples = int(
            model.artifact.get("n_samples", {}).get(target, 0)
        )
        if training_samples < MIN_TRAIN_SAMPLES:
            metrics = _empty_metrics()
            metrics.update(
                {
                    "n_samples": int(physically_valid.sum()),
                    "training_samples": training_samples,
                    "model_status": "insufficient_training_data",
                    "ml_enabled": False,
                }
            )
            result[target] = metrics
            continue

        (
            baseline_all,
            predicted_all,
            low_all,
            high_all,
        ) = _batch_prediction_arrays(
            model,
            evaluation,
            target=target,
            config=config,
        )

        baseline = baseline_all[positions]
        predicted = predicted_all[positions]
        low = low_all[positions]
        high = high_all[positions]

        physically_valid = _physical_target_mask(
            actual,
            config,
        )
        actual = actual[physically_valid]
        baseline = baseline[physically_valid]
        predicted = predicted[physically_valid]
        low = low[physically_valid]
        high = high[physically_valid]

        if (
            online_calibration_history is not None
            and target in online_calibration_history
            and len(actual) > 0
        ):
            low, high = _apply_online_interval_calibration(
                actual=actual,
                predicted=predicted,
                history=online_calibration_history[target],
            )

        metrics = _regression_metrics(
            actual=actual,
            predicted=predicted,
            low=low,
            high=high,
            baseline=baseline,
        )

        if target == "sulfur_ppm":
            valid = (
                np.isfinite(actual)
                & np.isfinite(predicted)
                & np.isfinite(low)
                & np.isfinite(high)
            )

            actual_over_spec = (
                actual[valid] > SULFUR_LIMIT_PPM
            )

            risks = np.asarray(
                [
                    QualityHydroModel._sulfur_spec_risk(
                        Interval(
                            mean=float(mean),
                            low=float(lower),
                            high=float(upper),
                            unit="ppm",
                        )
                    )
                    for mean, lower, upper in zip(
                        predicted[valid],
                        low[valid],
                        high[valid],
                        strict=True,
                    )
                ],
                dtype=float,
            )

            risk_threshold = float(
                model.artifact.get(
                    "sulfur_risk_threshold",
                    SULFUR_RISK_THRESHOLD,
                )
            )
            predicted_over_spec = (
                risks >= risk_threshold
            )

            precision, recall, f1, _ = (
                precision_recall_fscore_support(
                    actual_over_spec,
                    predicted_over_spec,
                    average="binary",
                    zero_division=0,
                )
            )

            metrics.update(
                {
                    "spec_precision": float(precision),
                    "spec_recall": float(recall),
                    "spec_f1": float(f1),
                    "spec_threshold_ppm":
                        SULFUR_LIMIT_PPM,
                    "risk_threshold":
                        risk_threshold,
                }
            )

        metrics["ml_enabled"] = bool(
            model.artifact.get("models", {}).get(target)
        )
        metrics["selected_predictor"] = (
            "baseline_plus_residual_ml"
            if metrics["ml_enabled"]
            else "baseline"
        )
        disabled_reason = model.artifact.get(
            "disabled_ml_targets",
            {},
        ).get(target)
        if disabled_reason is not None:
            metrics["disabled_ml_reason"] = disabled_reason

        result[target] = metrics

    return result


def _calibration_observations(
    model: QualityModel,
    frame: pd.DataFrame,
    *,
    target: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evaluation = frame.sort_values("date").reset_index(drop=True)
    label_mask = _fresh_label_mask(
        evaluation,
        target_column=str(config["target_column"]),
        age_column=(
            str(config["age_column"])
            if config.get("age_column") is not None
            else None
        ),
    )
    positions = np.flatnonzero(label_mask.to_numpy())

    if len(positions) == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    _, predicted_all, low_all, high_all = _batch_prediction_arrays(
        model,
        evaluation,
        target=target,
        config=config,
    )
    actual = pd.to_numeric(
        evaluation.iloc[positions][str(config["target_column"])],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = _physical_target_mask(actual, config)

    return (
        actual[valid],
        predicted_all[positions][valid],
        low_all[positions][valid],
        high_all[positions][valid],
    )


def calibrate_quality_model(
    model: QualityModel,
    validation: pd.DataFrame,
    *,
    target_config: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Выбирает predictor и калибрует его только на validation."""

    model.artifact.pop("interval_half_width", None)
    model.artifact.pop("sulfur_risk_threshold", None)

    candidate_metrics = evaluate_quality_model(
        model,
        validation,
        target_config=target_config,
    )
    disabled: dict[str, str] = {}

    for target, metrics in candidate_metrics.items():
        if not model.artifact.get("models", {}).get(target):
            continue

        if metrics.get("ml_beats_baseline") is False:
            model.artifact["models"].pop(target, None)
            model.artifact["residual_quantiles"][target] = {
                0.1: 0.0,
                0.5: 0.0,
                0.9: 0.0,
            }
            disabled[target] = (
                "validation_mae_not_better_than_baseline"
            )

    model.artifact["disabled_ml_targets"] = disabled
    interval_half_width: dict[str, float] = {}
    histories: dict[str, list[float]] = {}

    for target, config in target_config.items():
        actual, predicted, _, _ = _calibration_observations(
            model,
            validation,
            target=target,
            config=config,
        )

        valid = np.isfinite(actual) & np.isfinite(predicted)
        errors = np.abs(actual[valid] - predicted[valid])

        if errors.size == 0:
            continue

        histories[target] = errors.astype(float).tolist()
        interval_half_width[target] = _conformal_radius(errors)

    model.artifact["interval_half_width"] = interval_half_width

    if isinstance(model, QualityHydroModel):
        config = target_config["sulfur_ppm"]
        actual, predicted, low, high = _calibration_observations(
            model,
            validation,
            target="sulfur_ppm",
            config=config,
        )
        valid = (
            np.isfinite(actual)
            & np.isfinite(predicted)
            & np.isfinite(low)
            & np.isfinite(high)
        )
        actual = actual[valid]
        predicted = predicted[valid]
        low = low[valid]
        high = high[valid]
        over_spec = actual > SULFUR_LIMIT_PPM

        if over_spec.any():
            risks = np.asarray(
                [
                    model._sulfur_spec_risk(
                        Interval(
                            mean=float(mean),
                            low=float(lower),
                            high=float(upper),
                            unit="ppm",
                        )
                    )
                    for mean, lower, upper in zip(
                        predicted,
                        low,
                        high,
                        strict=True,
                    )
                ],
                dtype=float,
            )
            threshold = float(
                np.quantile(
                    risks[over_spec],
                    1.0 - SULFUR_VALIDATION_RECALL_TARGET,
                    method="lower",
                )
            )
            model.artifact["sulfur_risk_threshold"] = threshold

    return {
        "candidate_metrics": candidate_metrics,
        "disabled_ml_targets": disabled,
        "calibration_history": histories,
    }


def build_calibration_history(
    model: QualityModel,
    validation: pd.DataFrame,
    *,
    target_config: dict[str, dict[str, Any]],
) -> dict[str, list[float]]:
    histories: dict[str, list[float]] = {}

    for target, config in target_config.items():
        actual, predicted, _, _ = _calibration_observations(
            model,
            validation,
            target=target,
            config=config,
        )
        valid = np.isfinite(actual) & np.isfinite(predicted)
        errors = np.abs(actual[valid] - predicted[valid])
        if errors.size:
            histories[target] = errors.astype(float).tolist()

    return histories


def evaluate_anomaly_detector(
    detector: AnomalyDetector,
    frame: pd.DataFrame,
    *,
    evaluation_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Пакетно оценивает аномалии на всём test-периоде.
    """

    history = (
        frame.sort_values("date")
        .reset_index(drop=True)
    )

    feature_columns = list(
        detector.artifact["feature_columns"]
    )

    prepared = build_anomaly_features(
        history,
        feature_columns,
    )

    if evaluation_start is not None:
        evaluation_mask = history["date"].ge(evaluation_start)
        evaluation = history.loc[evaluation_mask].reset_index(drop=True)
        prepared = prepared.loc[evaluation_mask].reset_index(drop=True)
    else:
        evaluation = history

    current = (
        prepared.reindex(columns=feature_columns)
        .apply(pd.to_numeric, errors="coerce")
    )

    missing_flags = (
        current.isna()
        .any(axis=1)
        .to_numpy()
    )

    medians = pd.Series(
        detector.artifact["medians"],
        dtype=float,
    )
    filled = current.fillna(medians)

    scaler = detector.artifact["scaler"]
    isolation_forest = detector.artifact["model"]

    normalized = scaler.transform(filled)

    decision = isolation_forest.decision_function(
        normalized
    )
    isolation_flags = (
        isolation_forest.predict(normalized) == -1
    )

    positive_scale = float(
        detector.artifact["positive_score_scale"]
    )
    negative_scale = float(
        detector.artifact["negative_score_scale"]
    )

    scores = np.where(
        decision >= 0,
        0.5 - 0.5 * np.minimum(
            decision / positive_scale,
            1.0,
        ),
        0.5 + 0.5 * np.minimum(
            np.abs(decision) / negative_scale,
            1.0,
        ),
    )

    envelope_low = pd.Series(
        detector.artifact["envelope_low"],
        dtype=float,
    )
    envelope_high = pd.Series(
        detector.artifact["envelope_high"],
        dtype=float,
    )

    envelope_flags = (
        current.lt(envelope_low, axis="columns")
        | current.gt(envelope_high, axis="columns")
    ).any(axis=1).to_numpy()

    rolling_columns = [
        column
        for column in prepared.columns
        if column.startswith("anomaly_z__")
    ]

    if rolling_columns:
        rolling_thresholds = pd.Series(
            {
                column: float(
                    detector.artifact.get(
                        "rolling_z_thresholds",
                        {},
                    ).get(
                        column,
                        detector.artifact["z_threshold"],
                    )
                )
                for column in rolling_columns
            }
        )
        rolling_flags = (
            prepared[rolling_columns]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .ge(rolling_thresholds, axis="columns")
            .any(axis=1)
            .to_numpy()
        )
    else:
        rolling_flags = np.zeros(
            len(prepared),
            dtype=bool,
        )

    stale_features = detector.artifact.get(
        "stale_feature_columns",
        detector.artifact["feature_columns"],
    )
    stale_columns = [
        f"anomaly_stale_h__{feature}"
        for feature in stale_features
        if f"anomaly_stale_h__{feature}" in prepared.columns
    ]

    if stale_columns:
        stale_thresholds = pd.Series(
            {
                f"anomaly_stale_h__{feature}": float(
                    detector.artifact.get(
                        "stale_thresholds",
                        {},
                    ).get(
                        feature,
                        detector.artifact[
                            "stale_threshold_hours"
                        ],
                    )
                )
                for feature in stale_features
                if f"anomaly_stale_h__{feature}"
                in stale_columns
            }
        )
        stale_flags = (
            prepared[stale_columns]
            .apply(pd.to_numeric, errors="coerce")
            .ge(stale_thresholds, axis="columns")
            .any(axis=1)
            .to_numpy()
        )
    else:
        stale_flags = np.zeros(
            len(prepared),
            dtype=bool,
        )

    flags = (
        isolation_flags
        | envelope_flags
        | rolling_flags
        | stale_flags
        | missing_flags
    )

    scores = np.where(
        envelope_flags | rolling_flags,
        np.maximum(scores, 0.75),
        scores,
    )
    scores = np.where(
        stale_flags,
        np.maximum(scores, 0.7),
        scores,
    )
    scores = np.where(
        missing_flags,
        np.maximum(scores, 0.6),
        scores,
    )

    scores = np.clip(scores, 0.0, 1.0)

    auc: float | None = None

    if "is_known_incident" in evaluation.columns:
        labels = (
            evaluation["is_known_incident"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        if np.unique(labels).size == 2:
            auc = float(
                roc_auc_score(labels, scores)
            )

    return {
        "auc": auc,
        "n_test_rows": int(len(evaluation)),
        "n_flagged_in_test": int(flags.sum()),
        "flagged_share": float(flags.mean()),
        "n_isolation_forest": int(isolation_flags.sum()),
        "n_out_of_envelope": int(envelope_flags.sum()),
        "n_rolling_z_score": int(rolling_flags.sum()),
        "n_stale": int(stale_flags.sum()),
        "n_missing": int(missing_flags.sum()),
        "has_incident_labels": (
            "is_known_incident" in evaluation.columns
        ),
    }


def measure_prediction_latency(
    model: QualityModel,
    frame: pd.DataFrame,
    *,
    sample_size: int = 100,
) -> dict[str, float]:
    """
    Проверяет требование predict < 100 мс.

    Измеряется публичный predict вместе с подготовкой ответа и SHAP.
    """

    if frame.empty:
        return {
            "median_ms": float("nan"),
            "p95_ms": float("nan"),
            "max_ms": float("nan"),
        }

    positions = np.linspace(
        0,
        len(frame) - 1,
        num=min(sample_size, len(frame)),
        dtype=int,
    )

    durations: list[float] = []

    for position in positions:
        started = perf_counter()
        model.predict(frame.iloc[[int(position)]])
        durations.append(
            (perf_counter() - started) * 1000.0
        )

    values = np.asarray(durations, dtype=float)

    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
    }


def run_walk_forward_validation(
    *,
    frame: pd.DataFrame,
    quality_avt: QualityAVTModel,
    quality_hydro: QualityHydroModel,
    anomaly: AnomalyDetector,
) -> dict[str, Any]:
    periods = split_by_time(frame)

    avt_calibration_history = build_calibration_history(
        quality_avt,
        periods["validation"],
        target_config=AVT_TARGET_CONFIG,
    )
    hydro_calibration_history = build_calibration_history(
        quality_hydro,
        periods["validation"],
        target_config=HYDRO_TARGET_CONFIG,
    )

    validation_metrics = {
        "quality_avt": evaluate_quality_model(
            quality_avt,
            periods["validation"],
            target_config=AVT_TARGET_CONFIG,
        ),
        "quality_hydro": evaluate_quality_model(
            quality_hydro,
            periods["validation"],
            target_config=HYDRO_TARGET_CONFIG,
        ),
    }

    test_metrics = {
        "quality_avt": evaluate_quality_model(
            quality_avt,
            periods["test"],
            target_config=AVT_TARGET_CONFIG,
            online_calibration_history=avt_calibration_history,
        ),
        "quality_hydro": evaluate_quality_model(
            quality_hydro,
            periods["test"],
            target_config=HYDRO_TARGET_CONFIG,
            online_calibration_history=hydro_calibration_history,
        ),
        "anomaly": evaluate_anomaly_detector(
            anomaly,
            frame.loc[
                pd.to_datetime(frame["date"]).lt(
                    TEST_END_EXCLUSIVE
                )
            ],
            evaluation_start=TEST_START,
        ),
    }

    return {
        "split": {
            "train": [
                "2023-01-01",
                "2024-12-31",
            ],
            "validation": [
                "2025-01-01",
                "2025-06-30",
            ],
            "test": [
                "2025-07-01",
                "2026-08-07",
            ],
        },
        "validation": validation_metrics,
        **test_metrics,
        "runtime": {
            "quality_avt": measure_prediction_latency(
                quality_avt,
                periods["test"],
            ),
            "quality_hydro": measure_prediction_latency(
                quality_hydro,
                periods["test"],
            ),
        },
    }
