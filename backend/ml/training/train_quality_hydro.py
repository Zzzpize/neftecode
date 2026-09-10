from pathlib import Path

import pandas as pd

from ml.feature_engineering import (
    build_hydro_features,
    estimate_avt_hydro_delay,
)
from ml.models.quality_hydro import QualityHydroModel
from ml.vak import VAKCatalog


DATA_DIR = Path("../data")
ARTIFACT_PATH = Path(
    "ml/artifacts/quality_hydro.pkl"
)

TRAIN_START = pd.Timestamp("2023-01-01")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2025-01-01")


def main() -> None:
    frame = pd.read_parquet(
        DATA_DIR / "master.parquet"
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    train = frame.loc[
        frame["date"].ge(TRAIN_START)
        & frame["date"].lt(TRAIN_END_EXCLUSIVE)
    ].copy()

    # Задержка определяется исключительно на train.
    avt_delay_steps = estimate_avt_hydro_delay(
        train,
        target_column="pak_sulfur_ppm",
        min_delay_steps=1,
        max_delay_steps=72,
    )

    vak_catalog = VAKCatalog.load(
        DATA_DIR / "vac_formulas.parquet"
    )

    model = QualityHydroModel.fit(
        frame=train,
        vak_catalog=vak_catalog,
        avt_delay_steps=avt_delay_steps,
    )

    model.save(str(ARTIFACT_PATH))

    print(f"Model saved to {ARTIFACT_PATH}")
    print(
        "AVT-to-hydro delay:",
        avt_delay_steps,
        "steps /",
        avt_delay_steps * 10,
        "minutes",
    )
    print("Samples:", model.artifact["n_samples"])
    print(
        "Sulfur calibration samples:",
        model.artifact[
            "sulfur_calibration_samples"
        ],
    )


if __name__ == "__main__":
    main()