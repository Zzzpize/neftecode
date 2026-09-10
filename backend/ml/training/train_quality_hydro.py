import json
from pathlib import Path

import pandas as pd

from ml.models.quality_hydro import QualityHydroModel
from ml.vak import VAKCatalog


DATA_DIR = Path("../data")
ARTIFACT_PATH = Path(
    "ml/artifacts/quality_hydro.pkl"
)

TRAIN_START = pd.Timestamp("2023-01-01")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2025-01-01")


def main() -> None:
    feature_path = DATA_DIR / "ml_features.parquet"

    if not feature_path.exists():
        raise FileNotFoundError(
            "ml_features.parquet is missing; "
            "run python -m ml.training.prepare_features"
        )

    frame = pd.read_parquet(feature_path)

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    train = frame.loc[
        frame["date"].ge(TRAIN_START)
        & frame["date"].lt(TRAIN_END_EXCLUSIVE)
    ].copy()

    config_path = DATA_DIR / "ml_features_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            "ml_features_config.json is missing; "
            "run python -m ml.training.prepare_features"
        )

    feature_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    avt_delay_steps = int(
        feature_config["avt_delay_steps"]
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