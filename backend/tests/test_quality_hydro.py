import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    build_hydro_features,
    estimate_avt_hydro_delay,
)
from ml.models.quality_hydro import (
    QualityHydroModel,
)
from ml.types import Interval


def test_build_hydro_features_shifts_avt_values():
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=4,
                freq="10min",
            ),
            "avt_T55": [100.0, 110.0, 120.0, 130.0],
            "hydro_T5": [300.0, 301.0, 302.0, 303.0],
        }
    )

    result = build_hydro_features(
        frame,
        avt_delay_steps=2,
    )

    column = "avt_lag_2__T55"

    assert pd.isna(result.loc[0, column])
    assert pd.isna(result.loc[1, column])
    assert result.loc[2, column] == pytest.approx(100.0)
    assert result.loc[3, column] == pytest.approx(110.0)


def test_delay_is_estimated_only_from_past_avt_values():
    rng = np.random.default_rng(42)
    rows = 300
    expected_delay = 12

    avt = rng.normal(size=rows)
    sulfur = pd.Series(avt).shift(
        expected_delay
    ).to_numpy()

    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="10min",
            ),
            "avt_F30": avt,
            "pak_sulfur_ppm": sulfur,
        }
    )

    delay = estimate_avt_hydro_delay(
        frame,
        min_delay_steps=1,
        max_delay_steps=24,
    )

    assert delay == expected_delay


def test_sulfur_calibration_is_forward_only():
    frame = pd.DataFrame(
        {
            "pak_sulfur_ppm": [
                8.0,
                8.0,
                8.0,
                8.0,
            ],
            "lims_sulfur": [
                np.nan,
                10.0,
                10.0,
                12.0,
            ],
            "lims_age": [
                np.nan,
                0.0,
                1.0,
                0.0,
            ],
        }
    )

    config = {
        "target_column": "pak_sulfur_ppm",
        "calibration_column": "lims_sulfur",
        "calibration_age_column": "lims_age",
    }
    artifact = {}

    target, mask = (
        QualityHydroModel
        ._calibrated_sulfur_target(
            train=frame,
            config=config,
            artifact=artifact,
        )
    )

    # До первого ЛИМС никакая будущая поправка
    # использоваться не должна.
    assert target.iloc[0] == pytest.approx(8.0)

    # После ЛИМС=10 поправка равна +2.
    assert target.iloc[1] == pytest.approx(10.0)
    assert target.iloc[2] == pytest.approx(10.0)

    # Новый ЛИМС=12 меняет поправку только с этого момента.
    assert target.iloc[3] == pytest.approx(12.0)

    assert mask.all()
    assert artifact["sulfur_calibration_samples"] == 2


def test_sulfur_risk_increases_above_limit():
    low_risk = QualityHydroModel._sulfur_spec_risk(
        Interval(
            mean=7.0,
            low=6.0,
            high=8.0,
            unit="ppm",
        )
    )
    high_risk = QualityHydroModel._sulfur_spec_risk(
        Interval(
            mean=12.0,
            low=10.0,
            high=14.0,
            unit="ppm",
        )
    )

    assert 0.0 <= low_risk <= 1.0
    assert 0.0 <= high_risk <= 1.0
    assert high_risk > low_risk


def test_predict_after_action_does_not_mutate_state():
    model = QualityHydroModel()

    state = pd.DataFrame(
        {
            "hydro_T5": [350.0],
            "hydro_T11": [340.0],
        }
    )
    original = state.copy(deep=True)

    result = model.predict_after_action(
        state,
        {"T5": 360.0},
    )

    pd.testing.assert_frame_equal(state, original)
    assert result.confidence == "low"


def test_unknown_action_tag_is_rejected():
    model = QualityHydroModel()
    state = pd.DataFrame(
        {"hydro_T5": [350.0]}
    )

    with pytest.raises(KeyError):
        model.predict_after_action(
            state,
            {"UNKNOWN": 1.0},
        )


def test_non_numeric_action_is_rejected():
    model = QualityHydroModel()
    state = pd.DataFrame(
        {"hydro_T5": [350.0]}
    )

    with pytest.raises(TypeError):
        model.predict_after_action(
            state,
            {"T5": "hot"},  # type: ignore[arg-type]
        )