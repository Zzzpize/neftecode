from pathlib import Path

import pandas as pd
from data_layer.feature_registry import model_feature_names
from ml.models.anomaly import AnomalyDetector


from ml.paths import DATA_DIR, ARTIFACT_DIR, ML_DIR
REGISTRY_PATH = ML_DIR.parent / "data_layer" / "feature_registry.yaml"
ARTIFACT_PATH = ARTIFACT_DIR / "anomaly.pkl"

TRAIN_START = pd.Timestamp("2023-01-01")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2025-01-01")


def load_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    return [
        name
        for name in model_feature_names("anomaly", REGISTRY_PATH)
        if name in frame.columns
    ]


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

    feature_columns = load_feature_columns(train)

    # Когда появится разметка инцидентов:
    #
    # normal_mask = ~train["is_known_incident"]
    #
    # Пока разметки нет, используется весь train-период.
    normal_mask = None

    detector = AnomalyDetector.fit(
        frame=train,
        feature_columns=feature_columns,
        normal_mask=normal_mask,
        contamination=0.01,
        z_threshold=4.0,
        stale_threshold_hours=1.0,
    )

    detector.save(str(ARTIFACT_PATH))

    print(f"Model saved to {ARTIFACT_PATH}")
    print(
        "Training samples:",
        detector.artifact["n_samples"],
    )
    print(
        "Feature count:",
        len(detector.artifact["feature_columns"]),
    )


if __name__ == "__main__":
    main()
