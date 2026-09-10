from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_layer.feature_registry import sync_feature_registry
from ml.feature_engineering import (
    build_quality_features,
    estimate_avt_hydro_delay,
)


DATA_DIR = Path("../data")

MASTER_PATH = DATA_DIR / "master.parquet"
FEATURES_PATH = DATA_DIR / "ml_features.parquet"
CONFIG_PATH = DATA_DIR / "ml_features_config.json"

TRAIN_START = pd.Timestamp("2023-01-01")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2025-01-01")


def prepare_features(
    *,
    save: bool = True,
) -> tuple[pd.DataFrame, int]:
    frame = pd.read_parquet(MASTER_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    train = frame.loc[
        frame["date"].ge(TRAIN_START)
        & frame["date"].lt(TRAIN_END_EXCLUSIVE)
    ]

    # Задержку определяем только на train-периоде.
    # Validation и test в расчёте задержки не участвуют.
    avt_delay_steps = estimate_avt_hydro_delay(
        train,
        target_column="pak_sulfur_ppm",
        min_delay_steps=1,
        max_delay_steps=72,
    )

    featured = build_quality_features(
        frame,
        avt_delay_steps=avt_delay_steps,
    )

    sync_feature_registry(
        featured,
        avt_delay_steps=avt_delay_steps,
    )

    if save:
        featured.to_parquet(
            FEATURES_PATH,
            index=False,
        )

        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "avt_delay_steps": avt_delay_steps,
                    "step_minutes": 10,
                    "generated_at": pd.Timestamp.now().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return featured, avt_delay_steps


def main() -> None:
    featured, avt_delay_steps = prepare_features()

    print(f"Features saved to {FEATURES_PATH}")
    print(f"Rows: {len(featured)}")
    print(f"Columns: {featured.shape[1]}")
    print(f"AVT-to-hydro delay: {avt_delay_steps} steps")


if __name__ == "__main__":
    main()
