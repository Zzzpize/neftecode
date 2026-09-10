import numpy as np
import pandas as pd
import pytest

from ml.validation.walk_forward import (
    _apply_online_interval_calibration,
    _fresh_label_mask,
    _physical_target_mask,
    _regression_metrics,
    split_by_time,
)


def test_split_by_time_does_not_mix_periods():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2024-01-01",
                    "2025-03-01",
                ]
            ),
            "value": [3.0, 1.0, 2.0],
        }
    )

    periods = split_by_time(frame)

    assert periods["train"]["value"].tolist() == [1.0]
    assert periods["validation"]["value"].tolist() == [2.0]
    assert periods["test"]["value"].tolist() == [3.0]


def test_lims_value_is_counted_only_when_sample_is_new():
    frame = pd.DataFrame(
        {
            "target": [10.0, 10.0, 10.0, 12.0],
            "target_age_h": [0.0, 0.1, 0.2, 0.0],
        }
    )

    mask = _fresh_label_mask(
        frame,
        target_column="target",
        age_column="target_age_h",
    )

    assert mask.tolist() == [
        True,
        False,
        False,
        True,
    ]


def test_regression_metrics():
    metrics = _regression_metrics(
        actual=np.asarray([1.0, 2.0]),
        predicted=np.asarray([1.5, 1.5]),
        low=np.asarray([1.0, 1.0]),
        high=np.asarray([2.0, 2.0]),
        baseline=np.asarray([0.0, 0.0]),
    )

    assert metrics["n_samples"] == 2
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["coverage_90"] == pytest.approx(1.0)
    assert metrics["baseline_mae"] == pytest.approx(1.5)
    assert metrics["ml_beats_baseline"] is True


def test_impossible_laboratory_values_are_excluded():
    mask = _physical_target_mask(
        np.asarray([0.0, 320.0, 350.0]),
        {"absolute_range": (200.0, 500.0)},
    )

    assert mask.tolist() == [False, True, True]


def test_online_calibration_does_not_use_current_actual_value():
    history = [1.0] * 30
    predicted = np.asarray([100.0] * 5)

    first_low, first_high = _apply_online_interval_calibration(
        actual=np.asarray([101.0] * 5),
        predicted=predicted,
        history=history,
    )
    changed_low, changed_high = _apply_online_interval_calibration(
        actual=np.asarray(
            [1_000.0, 1_000.0, 1_000.0, 1_000.0, 101.0]
        ),
        predicted=predicted,
        history=history,
    )

    assert changed_low[0] == first_low[0]
    assert changed_high[0] == first_high[0]
    assert changed_low[-1] < first_low[-1]
    assert changed_high[-1] > first_high[-1]
