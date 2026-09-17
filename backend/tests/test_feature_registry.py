from pathlib import Path

import pandas as pd

from data_layer.feature_registry import (
    controllable_ranges,
    load_feature_registry,
    model_feature_names,
    validate_feature_registry,
)
from data_layer.simulator import Simulator, get_simulator
from ml.feature_engineering import build_anomaly_features
from ml.models.quality_avt import QualityAVTModel
from ml.models.quality_hydro import QualityHydroModel
from ml.optimizer.pareto import ParetoOptimizer


def test_registry_has_complete_metadata_for_all_model_features():
    registry = load_feature_registry()
    validate_feature_registry(registry)

    for model in ("quality_avt", "quality_hydro", "anomaly", "anomaly_context"):
        assert model_feature_names(model)

    assert any(name.startswith("lims__") for name in registry)
    assert any("_roll_mean_" in name for name in registry)
    assert any(name.startswith("anomaly_z__") for name in registry)


def test_optimizer_constraints_are_built_from_registry():
    expected = controllable_ranges(["avt_T55", "hydro_T6"])
    constraints = ParetoOptimizer.constraints_from_registry(
        hard={"sulfur_ppm": (0.0, 10.0)},
        max_deviation_pct=5.0,
        tags=["avt_T55", "hydro_T6"],
    )

    assert constraints.controllable_ranges == expected


def test_simulator_returns_quality_and_anomaly_features(tmp_path):
    dates = pd.date_range("2025-01-01", periods=20, freq="10min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "avt_T55": list(range(20)),
            "hydro_T6": list(range(100, 120)),
            "avt_lag_10m__T55": [float("nan"), *range(19)],
        }
    )
    path = tmp_path / "ml_features.parquet"
    frame.to_parquet(path, index=False)

    simulator = Simulator(path)
    state = simulator.get_state(dates[-1].to_pydatetime())
    expected = build_anomaly_features(frame, ["avt_T55", "hydro_T6"]).tail(1)

    assert state.at[0, "avt_lag_10m__T55"] == 18
    assert "anomaly_z__30m__avt_T55" in state.columns
    assert "anomaly_stale_h__hydro_T6" in state.columns
    assert state.at[0, "anomaly_z__30m__avt_T55"] == expected.iloc[0][
        "anomaly_z__30m__avt_T55"
    ]
    assert state.at[0, "anomaly_stale_h__hydro_T6"] == expected.iloc[0][
        "anomaly_stale_h__hydro_T6"
    ]

    assert isinstance(get_simulator(tmp_path), Simulator)


def test_registry_covers_saved_model_features():
    registry = load_feature_registry()
    artifact_dir = Path("ml/artifacts")

    for model_class, filename, registry_scope in (
        (QualityAVTModel, "quality_avt.pkl", "quality_avt"),
        (QualityHydroModel, "quality_hydro.pkl", "quality_hydro"),
    ):
        path = artifact_dir / filename
        if not path.exists():
            continue
        model = model_class.load(str(path))
        assert set(model.artifact["feature_columns"]) <= set(registry)
        assert set(model.artifact["feature_columns"]) <= set(
            model_feature_names(registry_scope)
        )
