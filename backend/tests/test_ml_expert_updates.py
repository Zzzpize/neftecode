import numpy as np
import pandas as pd
import pytest

from data_layer.feature_registry import controllable_ranges, load_feature_registry
from ml.availability import apply_lims_availability, inference_frame, validate_available_state
from ml.action_response import response_factor, fit_response
from ml.envelope import fit_envelope, outside_envelope
from ml.models.blending import BlendingModel
from ml.models.quality_hydro import QualityHydroModel
from ml.optimizer.pareto import ParetoOptimizer
from ml.scenario_blending import default_scenario, evaluate_blend
from ml.types import Interval, QualityPrediction, OptimizationConstraints
from ml.vak import VAKCatalog


def test_expert_formulas_match_independent_control_calculations():
    catalog = VAKCatalog.load_expert()
    assert len(catalog.names) == 17
    state = pd.DataFrame([{
        "avt_F65": 789.54, "avt_F32": 79., "avt_F30": 98.97,
        "avt_T66": 255.04, "avt_T33": 335.41, "avt_P67": 1.10, "avt_P4": 3.85,
        "hydro_T12": 175.10, "hydro_F15": 2787.38, "hydro_W7": .143,
        "hydro_T23": 234.20, "hydro_F1": 2.613, "hydro_F26": 201.23,
        "hydro_P13": 3.759, "hydro_F9": 171.09, "hydro_T6": 360.13,
        "hydro_F2": 88476.80, "lims__гидроочистка__pt2__95_t": 347.5,
    }])
    expected = {"AVT6:240-350:D15": 849.83, "AVT6:240-350:CFPP": -5.53,
                "24-2000:GODT:T90": 324.92, "24-2000:GODT:T50": 263.86,
                "24-2000:GODT:T95": 343.54}
    for name, value in expected.items():
        assert catalog.evaluate(name, state).iloc[0] == pytest.approx(value, abs=.03)


def test_lims_four_hour_delay_keeps_sample_truth_and_previous_available_result():
    dates = pd.date_range("2024-01-01", periods=7, freq="h")
    col = "lims__гидроочистка__pt2__d15"
    frame = pd.DataFrame({"date": dates, col: [800., 800., 900., 900., 900., 900., 900.],
                          col + "_age_h": [0., 1., 0., 1., 2., 3., 4.]})
    result = apply_lims_availability(frame)
    assert result[col].iloc[:4].isna().all()
    assert result[col].iloc[4:6].tolist() == [800., 800.]
    assert result[col].iloc[6] == 900.
    assert result["label__" + col].iloc[2] == 900.
    assert result[col + "_age_h"].iloc[4] == 4.
    pd.testing.assert_frame_equal(result, apply_lims_availability(result))
    known = inference_frame(result)
    assert not any(c.startswith("label__") for c in known)
    validate_available_state(known.tail(1))
    with pytest.raises(ValueError, match="Training labels"):
        validate_available_state(result.tail(1))
    with pytest.raises(ValueError, match="availability policy"):
        validate_available_state(frame.tail(1))


def test_future_lims_result_cannot_change_earlier_features():
    col = "lims__авт__pt1__50_t"
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60, freq="10min"),
                          col: [280.] * 30 + [300.] * 30,
                          col + "_age_h": [i / 6 for i in range(30)] * 2})
    changed = frame.copy()
    changed.loc[30:, col] = 999.
    a = inference_frame(apply_lims_availability(frame))
    b = inference_frame(apply_lims_availability(changed))
    pd.testing.assert_frame_equal(a.iloc[:54], b.iloc[:54])


def test_updated_units_controls_and_analyzers():
    r = load_feature_registry()
    hydro_controls = {k for k in controllable_ranges() if k.startswith("hydro_")}
    assert hydro_controls == {"hydro_T6", "hydro_F9", "hydro_P13"}
    assert r["hydro_P8"]["unit"] == "MPa"
    assert r["hydro_F25"]["unit"] == "Nm3/h"
    assert r["hydro_Q20"]["unit"] == "ppm"
    assert r["avt_F30"]["unit"] == "t/h"


def test_response_respects_horizon_and_direction():
    response = {"coefficients": {"hydro_T6": -.5}, "scales": {"hydro_T6": 10.},
                "delay_minutes": {"hydro_T6": 30}}
    state = pd.DataFrame([{"hydro_T6": 350., "ml_action_horizon_minutes": 20}])
    assert response_factor(state, {"hydro_T6": 360.}, response) == 1.
    state["ml_action_horizon_minutes"] = 180
    assert response_factor(state, {"hydro_T6": 360.}, response) < 1.
    assert response_factor(state, {"hydro_T6": 340.}, response) > 1.
    state["ml_action_horizon_minutes"] = 181
    with pytest.raises(ValueError, match="horizon"):
        response_factor(state, {}, response)


def test_response_is_fitted_with_train_only_bounds_and_physical_signs():
    rng = np.random.default_rng(7)
    controls = rng.normal(size=(300, 3))
    sulfur = np.exp(2 - .2 * controls[:, 0] + .1 * controls[:, 1] - .1 * controls[:, 2])
    frame = pd.DataFrame(controls, columns=["hydro_T6", "hydro_F9", "hydro_P13"])
    frame["date"] = pd.date_range("2024-01-01", periods=300, freq="10min")
    frame["pak_sulfur_ppm"] = sulfur
    fitted = fit_response(frame)
    assert all(0 <= d <= 180 for d in fitted["delay_minutes"].values())
    assert fitted["coefficients"]["hydro_T6"] < 0
    assert fitted["coefficients"]["hydro_F9"] > 0
    assert fitted["coefficients"]["hydro_P13"] < 0


def prediction(sulfur=8., cetane=48.):
    return QualityPrediction({"sulfur_ppm": Interval(sulfur, sulfur - 1, sulfur + 1, "ppm"),
                              "T95": Interval(340., 338., 342., "C"),
                              "cetane_number": Interval(cetane, cetane - 1, cetane + 1, "dimensionless")},
                             {}, "medium", [])


def test_additive_effect_cost_and_dose_are_configurable():
    config = default_scenario()
    base, base_metrics = evaluate_blend(BlendingModel(), prediction(), config, .2, .5, 0.)
    boosted, boosted_metrics = evaluate_blend(BlendingModel(), prediction(), config, .2, .5, .03)
    assert boosted.predictions["cetane_number"].mean - base.predictions["cetane_number"].mean == pytest.approx(3.)
    assert boosted.predictions["sulfur_ppm"].mean == pytest.approx(base.predictions["sulfur_ppm"].mean * .97)
    assert boosted_metrics["cost"] == pytest.approx(base_metrics["cost"] * .97 + 100 * .03)
    config["cetane_gain_per_additive_pct"] = 2.
    stronger, _ = evaluate_blend(BlendingModel(), prediction(), config, .2, .5, .03)
    assert stronger.predictions["cetane_number"].mean > boosted.predictions["cetane_number"].mean
    with pytest.raises(ValueError, match="maximum 3%"):
        evaluate_blend(BlendingModel(), prediction(), config, .2, .5, .031)


def test_blend_changes_when_raw_quality_or_recipe_changes():
    b = BlendingModel()
    config = default_scenario()
    p1, _ = evaluate_blend(b, prediction(), config, .2, 1., 0.)
    p2, _ = evaluate_blend(b, prediction(sulfur=14.), config, .2, 1., 0.)
    p3, _ = evaluate_blend(b, prediction(), config, .2, 0., 0.)
    assert p2.predictions["sulfur_ppm"].mean > p1.predictions["sulfur_ppm"].mean
    assert p3.predictions["sulfur_ppm"].mean > p1.predictions["sulfur_ppm"].mean


def test_joint_envelope_rejects_unseen_combination_inside_individual_ranges():
    x = np.linspace(300., 380., 300)
    frame = pd.DataFrame({"hydro_T6": x, "hydro_F9": x / 2})
    env = fit_envelope(frame)
    assert not outside_envelope(pd.DataFrame([{"hydro_T6": 340., "hydro_F9": 170.}]), env)
    assert outside_envelope(pd.DataFrame([{"hydro_T6": 340., "hydro_F9": 150.}]), env)


def test_nonfinite_prediction_cannot_pass_hard_constraints():
    p = prediction()
    p.predictions["sulfur_ppm"].high = float("nan")
    reason, violation = ParetoOptimizer._check_hard_constraints(
        prediction=p, constraints=OptimizationConstraints({"sulfur_ppm": (0., 10.)}, {}, 0))
    assert "nonfinite_prediction" in reason
    assert violation > 0


def test_hydro_action_surrogate_and_feed_scenario_change_actual_model_output():
    from ml.models.quality_hydro import TARGET_CONFIG
    columns = ["hydro_T6", "hydro_F9", "hydro_P13"]
    baselines = {p: 300. for p in TARGET_CONFIG}
    baselines.update(sulfur_ppm=8., cetane_number=49., D15=830.)
    model = QualityHydroModel({
        "trained": True, "feature_columns": columns,
        "feature_q01": dict(zip(columns, [300., 100., 3.])),
        "feature_q99": dict(zip(columns, [400., 300., 5.])),
        "fallback_baselines": baselines, "models": {}, "vak_formulas": {},
        "residual_quantiles": {}, "n_samples": {p: 120 for p in TARGET_CONFIG},
        "action_response": {"coefficients": {"hydro_T6": -.5},
                            "scales": {"hydro_T6": 10.}, "delay_minutes": {"hydro_T6": 30}},
    })
    state = pd.DataFrame([dict(zip(columns, [350., 180., 4.]))])
    current = model.predict(state)
    hotter = model.predict_after_action(state, {"hydro_T6": 360.})
    assert hotter.predictions["sulfur_ppm"].mean < current.predictions["sulfur_ppm"].mean
    state["ml_feed_sulfur_multiplier"] = 1.5
    assert model.predict(state).predictions["sulfur_ppm"].mean == pytest.approx(12.)
    assert model.predict_after_action(state, {"hydro_T6": 360.}).predictions["sulfur_ppm"].mean == pytest.approx(hotter.predictions["sulfur_ppm"].mean * 1.5)
    assert state.at[0, "hydro_T6"] == 350.


def test_optimizer_blends_and_reacts_to_quality_assignment():
    class AVT:
        artifact = {}
        def predict(self, state):
            return QualityPrediction({}, {}, "low", [])
    class Hydro:
        artifact = {}
        def predict(self, state):
            return prediction(sulfur=8.)
    optimizer = ParetoOptimizer(AVT(), Hydro(), BlendingModel())
    optimizer.MAX_OPTIMIZATION_SECONDS = .2
    state = pd.DataFrame([{"hydro_T6": 350., "ml_blending_scenario": True}])
    limits = {"sulfur_ppm": (0., 10.), "T95": (0., 360.), "cetane_number": (45., 100.)}
    constraints = OptimizationConstraints(limits, {"hydro_T6": (345., 355.)}, 3.)
    variants = optimizer.find_pareto(state, constraints, n_variants=2)
    assert variants
    assert all("blend__additive_fraction" in v.action and "cost" in v.metrics for v in variants)
    limits["cetane_number"] = (90., 100.)
    rejected = optimizer.find_pareto(state, constraints, n_variants=2)
    assert rejected and not any(v.feasible for v in rejected)


def test_tank_capacity_is_a_real_constraint():
    result, metrics = evaluate_blend(BlendingModel(), prediction(), default_scenario(), .5, 1., 0.)
    assert metrics["tank_capacity_violation"] == 1.
