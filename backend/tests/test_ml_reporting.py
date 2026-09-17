import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ml.reporting as reporting
from ml.availability import inference_frame
from ml.models import QualityAVTModel, QualityHydroModel


ROOT = Path(__file__).resolve().parents[2]


def test_notebook_sources_compile():
    notebook = json.loads((ROOT / "notebooks/ml_report.ipynb").read_text())
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    sources = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        sources.append(source)
        if cell["cell_type"] == "code":
            compile(source, cell["id"], "exec")
            # Users are expected to execute and save their demonstration outputs.
            assert isinstance(cell.get("outputs", []), list)
    joined = "\n".join(sources)
    assert "_apply_online_interval_calibration" not in joined
    assert "build_calibration_history" not in joined
    assert 'demo_control_tags = ["hydro_T6", "hydro_F9", "hydro_P13"]' in joined
    assert "ml_training.parquet" in joined
    assert all(name in ids for name in ("expert-scenarios-run", "blending-anomaly-demo", "runtime-report"))


def test_production_ml_does_not_import_higher_project_layers():
    prohibited = {"agents", "app", "llm", "frontend"}
    for path in (ROOT / "backend/ml").rglob("*.py"):
        # Legacy HTTP mock has a known app.schemas dependency. Removing it
        # requires changing its HTTP consumer, outside this ML-only task.
        if path.relative_to(ROOT / "backend/ml").as_posix() == "optimizer/mock.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            assert not any(m.split(".")[0] in prohibited for m in modules), path


def test_acceptance_does_not_treat_missing_evidence_as_pass():
    report = {"quality_avt": {"CFPP": {"coverage_90": None, "mae": None}},
              "quality_hydro": {"sulfur_ppm": {"coverage_90": .9, "mae": 1.,
                                               "baseline_mae": 2., "spec_recall": 1.,
                                               "n_actual_over_spec": 0}}}
    table = reporting.acceptance_table(report)
    assert table.iloc[0]["status"] == "NOT VERIFIED"
    assert table.iloc[-1]["status"] == "NOT VERIFIED"
    assert (table["status"] == "PASS").sum() == 2
    report["quality_hydro"]["sulfur_ppm"].update(coverage_90=.7, n_actual_over_spec=4)
    table = reporting.acceptance_table(report)
    assert table.iloc[-1]["status"] == "PASS"
    assert (table["status"] == "FAIL").sum() == 1


def test_observations_use_sample_truth_and_fixed_prediction_intervals(monkeypatch):
    col = "lims__авт__pt1__50_t"
    frame = pd.DataFrame({"date": pd.date_range("2025-07-01", periods=3, freq="h"),
                          col: [200.] * 3, col + "_age_h": [4., 5., 6.],
                          "label__" + col: [300., 300., 320.],
                          "label__" + col + "_age_h": [0., 1., 0.]})
    arrays = (np.full(3, 280.), np.full(3, 290.), np.full(3, 270.), np.full(3, 310.))
    monkeypatch.setattr(reporting, "_batch_prediction_arrays", lambda *a, **k: arrays)
    config = {"target_column": col, "age_column": col + "_age_h", "absolute_range": (150., 600.)}
    result = reporting.prediction_observations(None, frame, "T50", config)
    assert result.actual.tolist() == [300., 320.]
    assert result.low.tolist() == [270., 270.]
    assert result.high.tolist() == [310., 310.]
    changed = frame.copy()
    changed["label__" + col] = [500., 500., 520.]
    updated = reporting.prediction_observations(None, changed, "T50", config)
    np.testing.assert_array_equal(result[["prediction", "low", "high"]],
                                  updated[["prediction", "low", "high"]])


@pytest.mark.parametrize("model", [QualityAVTModel(), QualityHydroModel()])
def test_explain_rejects_unavailable_laboratory_labels(model):
    state = pd.DataFrame({"date": [pd.Timestamp("2025-07-01")], "label__lims__x": [1.]})
    with pytest.raises(ValueError, match="Training labels"):
        model.explain(state)
    assert model.explain(inference_frame(state)).top_features == []


def test_notebook_report_cells_execute_with_controlled_fixture(monkeypatch, tmp_path):
    """Exercise plots and three scenarios, not claim accuracy on synthetic data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data_layer.feature_registry import controllable_ranges
    from ml.models.quality_avt import TARGET_CONFIG as AVT_CONFIG
    from ml.models.quality_hydro import TARGET_CONFIG as HYDRO_CONFIG
    from ml.optimizer import ParetoOptimizer

    notebook = json.loads((ROOT / "notebooks/ml_report.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
    namespace = {}
    exec(compile(cells["setup"], "setup", "exec"), namespace)
    # Do not produce screenshots or persist fixture results in the notebook.
    namespace["display"] = lambda *args, **kwargs: None
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    exec(compile(cells["expert-preflight"], "expert-preflight", "exec"), namespace)

    ranges = controllable_ranges(["hydro_T6", "hydro_F9", "hydro_P13"])
    frame = pd.DataFrame({"date": pd.date_range("2025-07-01", periods=10, freq="h"),
                          "ml_lims_delay_h": 4., "avt_F30": 50., "avt_F32": 50., "avt_F65": 200.})
    for tag, bounds in ranges.items():
        frame[tag] = sum(bounds) / 2
    values = {"sulfur_ppm": 6., "T50": 280., "T90": 320., "T95": 340.,
              "D15": 830., "CFPP": -10., "cetane_number": 53.}

    def model_fixture(cls, config):
        for target, item in config.items():
            column = item["target_column"]
            frame[column] = values[target]
            if item.get("age_column"):
                age = item["age_column"]
                frame[age] = 4.
                frame["label__" + column] = values[target]
                frame["label__" + age] = 0.
        return cls({"trained": True, "reference_version": "expert_xlsx_2026_09_17",
                    "feature_columns": list(ranges), "models": {}, "vak_formulas": {},
                    "fallback_baselines": {t: values[t] for t in config},
                    "residual_quantiles": {t: {.1: -.1, .5: 0., .9: .1} for t in config},
                    "n_samples": {t: 200 for t in config}, "interval_coverage": .9,
                    "feature_q01": {t: b[0] for t, b in ranges.items()},
                    "feature_q99": {t: b[1] for t, b in ranges.items()}})

    avt, hydro = model_fixture(QualityAVTModel, AVT_CONFIG), model_fixture(QualityHydroModel, HYDRO_CONFIG)
    metric = {"n_samples": 10, "training_samples": 200, "mae": .1, "rmse": .2,
              "coverage_90": .9, "baseline_mae": .3, "baseline_kind": "train_median"}
    # Temporary fixture outputs only: the repository's real caches/artifacts are untouched.
    fixture_report = {"quality_avt": {t: dict(metric) for t in AVT_CONFIG},
                      "quality_hydro": {t: dict(metric) for t in HYDRO_CONFIG}}
    namespace.update(ARTIFACT_DIR=tmp_path, FEATURES_PATH=tmp_path / "ml_training.parquet",
                     METRICS_PATH=tmp_path / "metrics_report.json",
                     AVT_MODEL_PATH=tmp_path / "quality_avt.pkl",
                     HYDRO_MODEL_PATH=tmp_path / "quality_hydro.pkl")
    frame.to_parquet(namespace["FEATURES_PATH"])
    namespace["METRICS_PATH"].write_text(json.dumps(fixture_report))
    avt.save(str(namespace["AVT_MODEL_PATH"]))
    hydro.save(str(namespace["HYDRO_MODEL_PATH"]))
    # Avoid running an expensive search in a notebook unit test.
    monkeypatch.setattr(ParetoOptimizer, "MAX_OPTIMIZATION_SECONDS", .4)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or cell["id"] in {"setup", "expert-preflight"}:
            continue
        exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    assert len(namespace["scenario_summary"]) == 3
    assert not any(c.startswith("label__") for c in namespace["demo_row"])
    assert namespace["observations"][("Hydro", "T95")].shape[0] == 10


def test_trained_optimizer_cannot_bypass_registry_controls():
    from data_layer.feature_registry import controllable_ranges
    from ml.models import BlendingModel
    from ml.optimizer import ParetoOptimizer
    from ml.types import OptimizationConstraints

    optimizer = ParetoOptimizer(QualityAVTModel({"trained": True}), QualityHydroModel(), BlendingModel())
    bounds = controllable_ranges(["hydro_T6"])["hydro_T6"]
    state = pd.DataFrame({"hydro_T6": [sum(bounds) / 2], "hydro_T11": [300.]})
    requested = OptimizationConstraints({}, {"hydro_T6": (-10000., 10000.)}, 10000.)
    assert optimizer._build_search_ranges(state, requested)["hydro_T6"] == bounds
    requested.controllable_ranges = {"hydro_T11": (0., 500.)}
    with pytest.raises(ValueError, match="not controllable"):
        optimizer._build_search_ranges(state, requested)


def test_online_sulfur_does_not_use_current_or_future_truth():
    from ml.feature_engineering import adjust_sulfur_history
    frame = pd.DataFrame({"date": pd.date_range("2025-07-01", periods=40, freq="10min"),
                          "pak_sulfur_ppm": np.linspace(7., 9., 40)})
    means, lows, highs = np.full(40, 6.), np.full(40, 4.), np.full(40, 8.)
    original = adjust_sulfur_history(frame, means, lows, highs)
    changed = frame.copy()
    changed.loc[20:, "pak_sulfur_ppm"] = 100.
    modified = adjust_sulfur_history(changed, means, lows, highs)
    for left, right in zip(original[:3], modified[:3]):
        np.testing.assert_array_equal(left[:21], right[:21])
    assert len(original[3]) <= 31
    assert original[0][-1] > 6.


def test_online_sulfur_serving_matches_batch_and_actions_do_not_update_history():
    from ml.models.quality_hydro import TARGET_CONFIG
    from ml.validation.walk_forward import _batch_prediction_arrays
    values = {"sulfur_ppm": 6., "T50": 280., "T90": 320., "T95": 340.,
              "D15": 830., "cetane_number": 53.}
    model = QualityHydroModel({"trained": True, "sulfur_online": {"window": 30, "min_samples": 5},
                              "feature_columns": ["hydro_T6"], "models": {}, "vak_formulas": {},
                              "fallback_baselines": values,
                              "residual_quantiles": {t: {.1: -1., .5: 0., .9: 1.} for t in values},
                              "n_samples": {t: 200 for t in values},
                              "feature_q01": {"hydro_T6": 300.}, "feature_q99": {"hydro_T6": 400.}})
    frame = pd.DataFrame({"date": pd.date_range("2025-07-01", periods=40, freq="10min"),
                          "pak_sulfur_ppm": np.linspace(7., 9., 40), "hydro_T6": 350.})
    _, mean, low, high = _batch_prediction_arrays(model, frame, target="sulfur_ppm", config=TARGET_CONFIG["sulfur_ppm"])
    for i in range(len(frame)):
        prediction = model.predict(frame.iloc[[i]]).predictions["sulfur_ppm"]
        np.testing.assert_allclose([prediction.mean, prediction.low, prediction.high], [mean[i], low[i], high[i]])
    history = list(model._sulfur_history)
    model.predict_after_action(frame.tail(1), {"hydro_T6": 351.})
    assert history == model._sulfur_history
    repeated = model.predict(frame.tail(1)).predictions["sulfur_ppm"]
    np.testing.assert_allclose([repeated.mean, repeated.low, repeated.high], [mean[-1], low[-1], high[-1]])
