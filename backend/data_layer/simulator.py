"""Симулятор реалтайма: срез процесса на момент времени без утечки будущего."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class Simulator:
    """Держит master frame в памяти, отдаёт срезы по timestamp.

    Единственный источник состояния для агентов. Гарантирует: любой возвращённый
    DataFrame содержит только строки с date <= запрошенного timestamp.
    """

    def __init__(self, master_path: Path):
        self.master_path = Path(master_path)
        log.info("loading master frame: %s", self.master_path)
        self._df = pd.read_parquet(self.master_path)
        self._df = self._df.sort_values("date").reset_index(drop=True)
        self._dates = self._df["date"].to_numpy()
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
        return self._df.iloc[[idx]].reset_index(drop=True)

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
    return Simulator(Path(data_dir) / "master.parquet")


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
