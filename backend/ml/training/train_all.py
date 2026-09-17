# Запуск: python -m ml.training.train_all

from __future__ import annotations

import json
import argparse
import shutil
from datetime import datetime
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
    evaluate_quality_model,
    run_walk_forward_validation,
    split_by_time,
)
from ml.vak import VAKCatalog


from ml.paths import DATA_DIR, ARTIFACT_DIR

AVT_ARTIFACT_PATH = ARTIFACT_DIR / "quality_avt.pkl"
HYDRO_ARTIFACT_PATH = ARTIFACT_DIR / "quality_hydro.pkl"
ANOMALY_ARTIFACT_PATH = ARTIFACT_DIR / "anomaly.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML and evaluate the chronological holdout")
    parser.add_argument("--reuse-features", action="store_true",
                        help="Read existing ML cache; do not regenerate features or write data layer")
    args = parser.parse_args()
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1/6 Preparing features...", flush=True)

    if args.reuse_features:
        frame = pd.read_parquet(DATA_DIR / "ml_training.parquet")
        feature_config = json.loads((DATA_DIR / "ml_features_config.json").read_text())
        avt_delay_steps = int(feature_config["avt_delay_steps"])
    else:
        frame, avt_delay_steps = prepare_features(save=True)
    periods = split_by_time(frame)
    train = periods["train"]

    vak_catalog = VAKCatalog.load_expert()

    print("2/6 Training quality models...", flush=True)

    quality_avt = QualityAVTModel.fit(
        frame=train,
        vak_catalog=vak_catalog,
    )

    # Окно ТЗ: 2023–2024. Сужать его можно только после отдельной
    # проверки дрейфа на validation, не по предположению и не по test.
    hydro_train = train

    quality_hydro = QualityHydroModel.fit(
        frame=hydro_train,
        vak_catalog=vak_catalog,
        avt_delay_steps=avt_delay_steps,
    )
    # TЗ допускает скользящее окно при дрейфе. Решение принимается только
    # на validation, до просмотра финального test этой конфигурации.
    comparison = {}
    hydro_window = "2023-01-01/2024-12-31"
    sulfur_config = {"sulfur_ppm": HYDRO_TARGET_CONFIG["sulfur_ppm"]}
    full_metrics = evaluate_quality_model(quality_hydro, periods["validation"], target_config=sulfur_config)
    comparison[hydro_window] = full_metrics["sulfur_ppm"].get("mae")
    recent_train = train.loc[pd.to_datetime(train["date"]).ge("2024-01-01")]
    if not recent_train.empty and len(recent_train) < len(train):
        recent = QualityHydroModel.fit(recent_train, vak_catalog, avt_delay_steps=avt_delay_steps)
        recent_metrics = evaluate_quality_model(recent, periods["validation"], target_config=sulfur_config)
        recent_mae = recent_metrics["sulfur_ppm"].get("mae")
        comparison["2024-01-01/2024-12-31"] = recent_mae
        full_mae = comparison[hydro_window]
        if recent_mae is not None and (full_mae is None or recent_mae < full_mae):
            quality_hydro, hydro_window = recent, "2024-01-01/2024-12-31"
    print("Hydro validation window selection:", comparison, "chosen:", hydro_window, flush=True)
    quality_hydro.artifact["sulfur_online"] = {"window": 30, "min_samples": 5}

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

    existing = [p for p in (AVT_ARTIFACT_PATH, HYDRO_ARTIFACT_PATH, ANOMALY_ARTIFACT_PATH, METRICS_PATH) if p.exists()]
    if existing:
        backup = ARTIFACT_DIR / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, backup / path.name)
        print("Previous artifacts backed up:", backup, flush=True)

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
        "reference_version": "expert_xlsx_2026_09_17",
        "lims_availability_delay_hours": 4,
        "sulfur_target_source": "raw_pak_with_separate_lims_reference_metrics",
        "action_validation": "model_scenarios_not_causal_historical_proof",
        "hydro_window_validation_mae": comparison,
        "quality_avt": {
            "training_window": "2023-01-01/2024-12-31",
            "disabled_ml_targets": avt_calibration[
                "disabled_ml_targets"
            ],
        },
        "quality_hydro": {
            "training_window": hydro_window,
            "disabled_ml_targets": hydro_calibration[
                "disabled_ml_targets"
            ],
            "sulfur_risk_threshold": quality_hydro.artifact.get(
                "sulfur_risk_threshold"
            ),
        },
        "interval_calibration": (
            "fixed_conformal_validation_only_identical_to_predict"
        ),
        "sulfur_online_calibration": "past_PAK_only_bias_and_90pct_interval_window30",
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
