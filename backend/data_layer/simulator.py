"""Симулятор реалтайма: срез процесса на момент времени без утечки будущего."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_layer.feature_registry import model_feature_names

log = logging.getLogger(__name__)


class Simulator:
    """Держит inference feature frame в памяти, отдаёт срезы по timestamp.

    Единственный источник состояния для агентов. Гарантирует: любой возвращённый
    DataFrame содержит только строки с date <= запрошенного timestamp.
    """

    def __init__(self, feature_path: Path):
        self.feature_path = Path(feature_path)
        log.info("loading inference feature frame: %s", self.feature_path)
        self._df = pd.read_parquet(self.feature_path)
        self._df = self._df.sort_values("date").reset_index(drop=True)
        self._dates = self._df["date"].to_numpy()
        self._anomaly_features = [
            name
            for name in model_feature_names("anomaly")
            if name in self._df.columns
        ]
        self._anomaly_values = {
            name: pd.to_numeric(self._df[name], errors="coerce").to_numpy(
                dtype=float,
                copy=False,
            )
            for name in self._anomaly_features
        }
        log.info("simulator ready: %d timestamps, %d columns",
                 len(self._df), self._df.shape[1])

    @property
    def min_date(self) -> datetime:
        return pd.Timestamp(self._dates[0]).to_pydatetime()

    @property
    def max_date(self) -> datetime:
        return pd.Timestamp(self._dates[-1]).to_pydatetime()

    def get_state(self, ts: datetime) -> pd.DataFrame:
        """Одна строка: последний известный срез на момент ts (date <= ts)."""
        ts64 = pd.Timestamp(ts).to_datetime64()
        idx = self._df["date"].searchsorted(ts64, side="right") - 1
        if idx < 0:
            raise ValueError(f"нет данных до {ts}")
        state = self._df.iloc[[idx]].reset_index(drop=True).copy()
        return self._add_anomaly_context(state, idx)

    def _add_anomaly_context(self, state: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Calculates causal z-score/staleness fields for a single snapshot."""

        current_date = pd.Timestamp(self._dates[idx])
        windows = {"30m": 3, "60m": 6, "120m": 12}
        context: dict[str, float] = {}

        for column in self._anomaly_features:
            values = self._anomaly_values[column]
            current = values[idx]

            for window_name, window_size in windows.items():
                history = values[max(0, idx - window_size):idx]
                finite = history[np.isfinite(history)]
                minimum = max(2, window_size // 2)
                z_score = np.nan
                if np.isfinite(current) and finite.size >= minimum:
                    deviation = float(np.std(finite, ddof=1))
                    if deviation > 0.0:
                        z_score = (current - float(np.mean(finite))) / deviation
                context[f"anomaly_z__{window_name}__{column}"] = z_score

            stale_hours = np.nan
            if np.isfinite(current):
                first_equal = idx
                while first_equal > 0:
                    previous = values[first_equal - 1]
                    if not np.isfinite(previous) or previous != current:
                        break
                    first_equal -= 1
                stale_hours = (
                    current_date - pd.Timestamp(self._dates[first_equal])
                ).total_seconds() / 3600.0
            context[f"anomaly_stale_h__{column}"] = stale_hours

        if context:
            context_frame = pd.DataFrame([context], index=state.index)
            return pd.concat([state, context_frame], axis=1)
        return state

    def get_window(self, ts: datetime, minutes_back: int) -> pd.DataFrame:
        """Строки за окно [ts - minutes_back, ts]."""
        ts_end = pd.Timestamp(ts)
        ts_start = ts_end - timedelta(minutes=minutes_back)
        mask = (self._df["date"] >= ts_start) & (self._df["date"] <= ts_end)
        return self._df.loc[mask].reset_index(drop=True)

    def get_history(self, tag: str, ts_from: datetime, ts_to: datetime) -> pd.DataFrame:
        """Временной ряд одного тега для графика."""
        if tag not in self._df.columns:
            raise KeyError(f"неизвестный тег: {tag}")
        mask = (self._df["date"] >= pd.Timestamp(ts_from)) & (self._df["date"] <= pd.Timestamp(ts_to))
        return self._df.loc[mask, ["date", tag]].reset_index(drop=True)

    def available_timestamps(self, ts_from: datetime | None = None,
                             ts_to: datetime | None = None,
                             step_minutes: int = 60) -> list[datetime]:
        """Прореженный список доступных timestamp для TimeMachine на фронте."""
        df = self._df["date"]
        if ts_from is not None:
            df = df[df >= pd.Timestamp(ts_from)]
        if ts_to is not None:
            df = df[df <= pd.Timestamp(ts_to)]
        step = max(1, step_minutes // 10)
        return [pd.Timestamp(x).to_pydatetime() for x in df.iloc[::step]]

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)


def get_simulator(data_dir: Path) -> Simulator:
    """Фабрика для DI. Возвращает готовый Simulator."""
    return Simulator(Path(data_dir) / "ml_features.parquet")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from app.config import settings
    sim = get_simulator(settings.data_dir)
    print(f"range: {sim.min_date} .. {sim.max_date}")
    ts = pd.Timestamp("2024-06-15 14:20:00").to_pydatetime()
    state = sim.get_state(ts)
    print(f"state at {ts}: {state.shape}")
    print(state[["date", "avt_T55", "hydro_T5", "pak_sulfur_ppm"]].to_string())


if __name__ == "__main__":
    main()
