from pathlib import Path

import pandas as pd

from ml.models.quality_avt import QualityAVTModel
from ml.vak import VAKCatalog


DATA_DIR = Path("../data")
ARTIFACT_PATH = Path("ml/artifacts/quality_avt.pkl")

TRAIN_START = pd.Timestamp("2023-01-01")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2025-01-01")


def main() -> None:
    frame = pd.read_parquet(DATA_DIR / "master.parquet")
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