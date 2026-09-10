# Запуск: python -m ml.training.train_all

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.models.anomaly import AnomalyDetector
from ml.models.quality_avt import (
    TARGET_CONFIG as AVT_TARGET_CONFIG,
    QualityAVTModel,
)
from ml.models.quality_hydro import (
    TARGET_CONFIG as HYDRO_TARGET_CONFIG,
    QualityHydroModel,
)
from ml.training.prepare_features import prepare_features
from ml.training.train_anomaly import load_feature_columns
from ml.validation.walk_forward import (
    calibrate_quality_model,
    run_walk_forward_validation,
    split_by_time,
)
from ml.vak import VAKCatalog


DATA_DIR = Path("../data")
ARTIFACT_DIR = Path("ml/artifacts")

AVT_ARTIFACT_PATH = ARTIFACT_DIR / "quality_avt.pkl"
HYDRO_ARTIFACT_PATH = ARTIFACT_DIR / "quality_hydro.pkl"
ANOMALY_ARTIFACT_PATH = ARTIFACT_DIR / "anomaly.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics_report.json"


def main() -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1/6 Preparing features...", flush=True)

    frame, avt_delay_steps = prepare_features(
        save=True,
    )
    periods = split_by_time(frame)
    train = periods["train"]

    vak_catalog = VAKCatalog.load(
        DATA_DIR / "vac_formulas.parquet"
    )

    print("2/6 Training quality models...", flush=True)

    quality_avt = QualityAVTModel.fit(
        frame=train,
        vak_catalog=vak_catalog,
    )

    # На validation подтверждён дрейф hydro-процесса. В соответствии с
    # разделом «Риски» tz_ml.md используем последние 12 месяцев train.
    hydro_train = train.loc[
        pd.to_datetime(train["date"]).ge(
            pd.Timestamp("2024-01-01")
        )
    ].copy()

    quality_hydro = QualityHydroModel.fit(
        frame=hydro_train,
        vak_catalog=vak_catalog,
        avt_delay_steps=avt_delay_steps,
    )

    print("3/6 Calibrating on validation...", flush=True)

    avt_calibration = calibrate_quality_model(
        quality_avt,
        periods["validation"],
        target_config=AVT_TARGET_CONFIG,
    )
    hydro_calibration = calibrate_quality_model(
        quality_hydro,
        periods["validation"],
        target_config=HYDRO_TARGET_CONFIG,
    )

    print("4/6 Training anomaly detector...", flush=True)

    anomaly = AnomalyDetector.fit(
        frame=train,
        feature_columns=load_feature_columns(train),
        normal_mask=None,
        contamination=0.01,
        z_threshold=4.0,
        stale_threshold_hours=1.0,
    )

    print("5/6 Saving models...", flush=True)

    quality_avt.save(str(AVT_ARTIFACT_PATH))
    quality_hydro.save(str(HYDRO_ARTIFACT_PATH))
    anomaly.save(str(ANOMALY_ARTIFACT_PATH))

    print("6/6 Running walk-forward validation...", flush=True)

    metrics = run_walk_forward_validation(
        frame=frame,
        quality_avt=quality_avt,
        quality_hydro=quality_hydro,
        anomaly=anomaly,
    )
    metrics["model_selection"] = {
        "quality_avt": {
            "training_window": "2023-01-01/2024-12-31",
            "disabled_ml_targets": avt_calibration[
                "disabled_ml_targets"
            ],
        },
        "quality_hydro": {
            "training_window": "2024-01-01/2024-12-31",
            "disabled_ml_targets": hydro_calibration[
                "disabled_ml_targets"
            ],
            "sulfur_risk_threshold": quality_hydro.artifact.get(
                "sulfur_risk_threshold"
            ),
        },
        "interval_calibration": (
            "rolling_conformal_30_previous_labels"
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(f"AVT model: {AVT_ARTIFACT_PATH}", flush=True)
    print(f"Hydro model: {HYDRO_ARTIFACT_PATH}", flush=True)
    print(f"Anomaly model: {ANOMALY_ARTIFACT_PATH}", flush=True)
    print(f"Metrics: {METRICS_PATH}", flush=True)
    print(
        "AVT-to-hydro delay:",
        avt_delay_steps,
        "steps",
        flush=True,
    )


if __name__ == "__main__":
    main()
