# Скользящие статистики, лаги, возраст ЛИМС, кумулятивные признаки.

from __future__ import annotations

import numpy as np
import pandas as pd

from collections.abc import Sequence
from data_layer.feature_registry import engineered_time_feature_sources


LAG_STEPS: dict[str, int] = {
    "10m": 1,
    "30m": 3,
    "60m": 6,
}

ROLLING_STEPS: dict[str, int] = {
    "30m": 3,
    "60m": 6,
    "120m": 12,
}

def _key_features(prefix: str) -> tuple[str, ...]:
    """Registry is authoritative for lag/rolling source selection."""

    return tuple(
        name
        for name in engineered_time_feature_sources()
        if name.startswith(prefix)
    )


def _prepare_time_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    if frame.empty:
        raise ValueError("frame must not be empty")

    if "date" not in frame.columns:
        raise ValueError("frame must contain the date column")

    result = frame.copy()
    result["date"] = pd.to_datetime(
        result["date"],
        errors="raise",
    )

    return (
        result.sort_values("date")
        .reset_index(drop=True)
    )


def add_causal_time_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Добавляет лаги и rolling-признаки без утечки будущего.

    Для строки T статистики строятся по значениям до T.
    Текущее значение T остаётся отдельной исходной фичей.
    """

    result = _prepare_time_frame(frame)
    generated: dict[str, pd.Series] = {}

    for column in feature_columns:
        if column not in result.columns:
            continue

        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        # Только история до текущего момента.
        history = values.shift(1)

        installation, short_name = column.split(
            "_",
            maxsplit=1,
        )

        for lag_name, lag_steps in LAG_STEPS.items():
            generated[
                f"{installation}_lag_{lag_name}__{short_name}"
            ] = values.shift(lag_steps)

        for window_name, window_steps in ROLLING_STEPS.items():
            rolling = history.rolling(
                window=window_steps,
                min_periods=2,
            )

            generated[
                f"{installation}_roll_mean_{window_name}__{short_name}"
            ] = rolling.mean()

            generated[
                f"{installation}_roll_std_{window_name}__{short_name}"
            ] = rolling.std()

            # Среднее изменение за один 10-минутный шаг.
            generated[
                f"{installation}_roll_slope_{window_name}__{short_name}"
            ] = (
                history.diff()
                .rolling(
                    window=max(window_steps - 1, 2),
                    min_periods=2,
                )
                .mean()
            )

    if not generated:
        return result

    generated_frame = pd.DataFrame(
        generated,
        index=result.index,
    )

    return pd.concat(
        [result, generated_frame],
        axis=1,
    )


def add_runtime_proxies(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет кумулятивные признаки наработки.

    При больших пропусках времени один переход ограничивается часом,
    чтобы простой источника не считался полной наработкой установки.
    """

    result = _prepare_time_frame(frame)

    step_hours = (
        result["date"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )

    cumulative_hours = step_hours.cumsum()

    result["avt_runtime_proxy_h"] = cumulative_hours
    result["hydro_catalyst_runtime_proxy_h"] = cumulative_hours

    return result


def build_quality_features(
    frame: pd.DataFrame,
    *,
    avt_delay_steps: int,
) -> pd.DataFrame:
    """
    Единый feature pipeline для обучения и инференса.

    Порядок важен:
    1. причинные лаги и rolling;
    2. кумулятивные признаки;
    3. физическая задержка АВТ -> гидроочистка.
    """

    result = add_causal_time_features(
        frame,
        feature_columns=(
            *_key_features("avt_"),
            *_key_features("hydro_"),
        ),
    )

    result = add_runtime_proxies(result)

    result = build_hydro_features(
        result,
        avt_delay_steps=avt_delay_steps,
    )

    return result


def estimate_avt_hydro_delay(
    frame: pd.DataFrame,
    *,
    target_column: str = "pak_sulfur_ppm",
    min_delay_steps: int = 1,
    max_delay_steps: int = 72,
) -> int:
    """
    Определяет задержку между АВТ и гидроочисткой по кросс-корреляции.

    Один шаг master frame равен 10 минутам. Например:
        18 шагов = 3 часа.

    Функцию необходимо вызывать только на train-периоде, чтобы выбор
    задержки не использовал validation/test и не создавал утечку.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    if target_column not in frame.columns:
        raise ValueError(
            f"Target column is missing: {target_column}"
        )

    if min_delay_steps < 1:
        raise ValueError("min_delay_steps must be positive")

    if max_delay_steps < min_delay_steps:
        raise ValueError(
            "max_delay_steps must be greater than or equal "
            "to min_delay_steps"
        )

    avt_columns = [
        column
        for column in frame.columns
        if column.startswith("avt_")
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].nunique(dropna=True) > 1
    ]

    if not avt_columns:
        raise ValueError("No numeric AVT columns found")

    target = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    )

    best_delay: int | None = None
    best_score = float("-inf")

    for delay_steps in range(
        min_delay_steps,
        max_delay_steps + 1,
    ):
        shifted = (
            frame[avt_columns]
            .apply(pd.to_numeric, errors="coerce")
            .shift(delay_steps)
        )

        correlations = shifted.corrwith(target).abs()
        correlations = correlations.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if correlations.empty:
            continue

        # Используем медиану пяти наиболее связанных сигналов.
        # Это устойчивее, чем выбирать один случайно коррелирующий тег.
        score = float(
            correlations.nlargest(
                min(5, len(correlations))
            ).median()
        )

        if score > best_score:
            best_score = score
            best_delay = delay_steps

    if best_delay is None:
        raise ValueError(
            "Cannot estimate AVT-to-hydro delay: "
            "not enough overlapping observations"
        )

    return best_delay


def build_hydro_features(
    frame: pd.DataFrame,
    *,
    avt_delay_steps: int,
) -> pd.DataFrame:
    """
    Добавляет причинно корректные признаки АВТ для гидроочистки.

    Для строки времени T колонка avt_lag_* содержит состояние АВТ
    в момент T - avt_delay_steps.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    if "date" not in frame.columns:
        raise ValueError("frame must contain the date column")

    if avt_delay_steps < 1:
        raise ValueError("avt_delay_steps must be positive")

    result = (
        frame.sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    avt_columns = [
        column
        for column in _key_features("avt_")
        if column in result.columns
        and pd.api.types.is_numeric_dtype(result[column])
    ]

    for column in avt_columns:
        short_name = column.removeprefix("avt_")
        lag_column = (
            f"avt_lag_{avt_delay_steps}__{short_name}"
        )
        result[lag_column] = result[column].shift(
            avt_delay_steps
        )

    return result

ANOMALY_WINDOWS: dict[str, int] = {
    "30m": 3,
    "60m": 6,
    "120m": 12,
}


def build_anomaly_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    if "date" not in frame.columns:
        raise ValueError("frame must contain the date column")

    missing = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "Missing anomaly feature columns: "
            + ", ".join(missing)
        )

    result = (
        frame.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    result["date"] = pd.to_datetime(result["date"])

    generated: dict[str, pd.Series] = {}

    for column in feature_columns:
        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        history = values.shift(1)

        for window_name, window_steps in ANOMALY_WINDOWS.items():
            min_periods = max(2, window_steps // 2)

            rolling_mean = history.rolling(
                window_steps,
                min_periods=min_periods,
            ).mean()

            rolling_std = history.rolling(
                window_steps,
                min_periods=min_periods,
            ).std()

            z_score = (
                (values - rolling_mean)
                / rolling_std.replace(0.0, np.nan)
            )

            generated[
                f"anomaly_z__{window_name}__{column}"
            ] = z_score.replace(
                [np.inf, -np.inf],
                np.nan,
            )

        previous = values.shift(1)
        changed = (
            values.ne(previous)
            | values.isna()
            | previous.isna()
        )

        last_change_time = (
            result["date"]
            .where(changed)
            .ffill()
        )

        stale_hours = (
            result["date"] - last_change_time
        ).dt.total_seconds() / 3600.0

        generated[
            f"anomaly_stale_h__{column}"
        ] = stale_hours.where(values.notna())

    generated_frame = pd.DataFrame(
        generated,
        index=result.index,
    )

    return pd.concat(
        [result, generated_frame],
        axis=1,
    )
