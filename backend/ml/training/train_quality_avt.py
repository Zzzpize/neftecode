from pathlib import Path

import pandas as pd

from ml.models.quality_avt import QualityAVTModel
from ml.vak import VAKCatalog


DATA_DIR = Path("../data")
ARTIFACT_PATH = Path("ml/artifacts/quality_avt.pkl")

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

    train = frame.loc[
        frame["date"].ge(TRAIN_START)
        & frame["date"].lt(TRAIN_END_EXCLUSIVE)
    ].copy()

    vak_catalog = VAKCatalog.load(
        DATA_DIR / "vac_formulas.parquet"
    )

    model = QualityAVTModel.fit(
        frame=train,
        vak_catalog=vak_catalog,
    )
    model.save(str(ARTIFACT_PATH))

    print(f"Model saved to {ARTIFACT_PATH}")
    print("Samples:", model.artifact["n_samples"])


if __name__ == "__main__":
    main()