import pickle

import pandas as pd
import pytest

from ml.models import (
    AnomalyDetector,
    BlendingModel,
    QualityAVTModel,
    QualityHydroModel,
)
from ml.optimizer import ParetoOptimizer
from ml.types import (
    AnomalyReport,
    BlendedProduct,
    Component,
    Explanation,
    Interval,
    OptimizationConstraints,
    QualityPrediction,
)


@pytest.fixture
def state() -> pd.DataFrame:
    return pd.DataFrame([{"avt_temperature": 348.0, "hydro_feed": 44.5}])


@pytest.mark.parametrize(
    ("model_class", "expected_targets"),
    [
        (QualityAVTModel, {"T50", "T90", "D15", "CFPP"}),
        (QualityHydroModel, {"sulfur_ppm", "T50", "T90", "D15", "T95", "cetane_number"}),
    ],
)
def test_quality_model_contract(model_class, expected_targets, state):
    model = model_class()

    prediction = model.predict(state)
    changed_prediction = model.predict_after_action(state, {"avt_temperature": 350.0})
    explanation = model.explain(state)

    assert isinstance(prediction, QualityPrediction)
    assert set(prediction.predictions) == expected_targets
    assert all(isinstance(value, Interval) for value in prediction.predictions.values())
    assert prediction.confidence in {"high", "medium", "low"}
    assert isinstance(changed_prediction, QualityPrediction)
    assert isinstance(explanation, Explanation)
    assert state.at[0, "avt_temperature"] == 348.0


@pytest.mark.parametrize(
    "model_class",
    [QualityAVTModel, QualityHydroModel, AnomalyDetector],
)
def test_artifact_load_contract(model_class, tmp_path):
    artifact = {"version": "smoke-test"}
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as file:
        pickle.dump(artifact, file)

    model = model_class.load(str(artifact_path))

    assert isinstance(model, model_class)
    assert model.artifact == artifact


def test_anomaly_detector_contract(state):
    report = AnomalyDetector().score(state)

    assert isinstance(report, AnomalyReport)
    assert 0.0 <= report.anomaly_score <= 1.0
    assert isinstance(report.flagged_tags, list)
    assert isinstance(report.stale_tags, list)


def test_blending_model_contract():
    components = [
        Component("first", {"sulfur_ppm": 4.0, "D15": 830.0}, mass_flow=60.0),
        Component("second", {"sulfur_ppm": 8.0, "D15": 840.0}, mass_flow=40.0),
    ]

    product = BlendingModel().blend(components, fractions=[0.5, 0.5])

    assert isinstance(product, BlendedProduct)
    assert product.properties["sulfur_ppm"] == pytest.approx(6.0)
    assert product.properties["D15"] == pytest.approx(835.0)
    assert product.total_mass == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("components", "fractions"),
    [
        ([], []),
        ([Component("only", {"D15": 830.0}, mass_flow=1.0)], [0.5]),
        ([Component("only", {"D15": 830.0}, mass_flow=1.0)], [0.5, 0.5]),
    ],
)
def test_blending_model_rejects_invalid_input(components, fractions):
    with pytest.raises(ValueError):
        BlendingModel().blend(components, fractions)


def test_pareto_optimizer_contract(state):
    optimizer = ParetoOptimizer(
        quality_avt=QualityAVTModel(),
        quality_hydro=QualityHydroModel(),
        blending=BlendingModel(),
    )
    constraints = OptimizationConstraints(
        hard={"sulfur_ppm": (0.0, 10.0)},
        controllable_ranges={"hydro_feed": (40.0, 50.0)},
        max_deviation_pct=10.0,
    )

    variants = optimizer.find_pareto(state, constraints, n_variants=10)

    assert isinstance(variants, list)


@pytest.mark.parametrize(
    "invalid_state",
    [
        {"avt_temperature": 348.0},
        pd.DataFrame(),
        pd.DataFrame([{"value": 1.0}, {"value": 2.0}]),
    ],
)
def test_models_reject_state_that_is_not_one_row(invalid_state):
    expected_error = TypeError if isinstance(invalid_state, dict) else ValueError

    with pytest.raises(expected_error):
        QualityAVTModel().predict(invalid_state)
    with pytest.raises(expected_error):
        QualityHydroModel().predict(invalid_state)
    with pytest.raises(expected_error):
        AnomalyDetector().score(invalid_state)
