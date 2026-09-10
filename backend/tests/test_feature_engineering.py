import pandas as pd
import pytest

from ml.feature_engineering import (
    add_causal_time_features,
    build_quality_features,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=15,
                freq="10min",
            ),
            "avt_F30": list(range(15)),
            "avt_T55": list(range(100, 115)),
            "hydro_T5": list(range(300, 315)),
        }
    )


def test_lags_are_causal():
    result = add_causal_time_features(
        make_frame(),
        ["avt_F30"],
    )

    assert result.loc[1, "avt_lag_10m__F30"] == 0
    assert result.loc[3, "avt_lag_30m__F30"] == 0
    assert result.loc[6, "avt_lag_60m__F30"] == 0


def test_rolling_does_not_include_current_value():
    frame = make_frame()
    frame.loc[3, "avt_F30"] = 1_000_000

    result = add_causal_time_features(
        frame,
        ["avt_F30"],
    )

    # Для строки 3 используются только строки 0, 1, 2.
    assert result.loc[
        3,
        "avt_roll_mean_30m__F30",
    ] == pytest.approx(1.0)


def test_future_change_does_not_change_past_features():
    original = make_frame()
    changed = original.copy()
    changed.loc[10:, "avt_F30"] = 999_999

    original_features = add_causal_time_features(
        original,
        ["avt_F30"],
    )
    changed_features = add_causal_time_features(
        changed,
        ["avt_F30"],
    )

    pd.testing.assert_frame_equal(
        original_features.iloc[:10],
        changed_features.iloc[:10],
    )


def test_quality_pipeline_preserves_lims_age():
    frame = make_frame()
    frame["lims__авт__pt1__50_t_age_h"] = range(15)

    result = build_quality_features(
        frame,
        avt_delay_steps=2,
    )

    assert result[
        "lims__авт__pt1__50_t_age_h"
    ].equals(
        frame["lims__авт__pt1__50_t_age_h"]
    )

    assert "avt_lag_2__F30" in result.columns
    assert "avt_runtime_proxy_h" in result.columns
    assert "hydro_catalyst_runtime_proxy_h" in result.columns