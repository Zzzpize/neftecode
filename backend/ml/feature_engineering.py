# Скользящие статистики, лаги, возраст ЛИМС, кумулятивные признаки.

from __future__ import annotations

import numpy as np
import pandas as pd


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
        for column in result.columns
        if column.startswith("avt_")
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