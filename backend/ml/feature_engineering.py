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