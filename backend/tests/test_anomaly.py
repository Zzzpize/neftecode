import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    build_anomaly_features,
)
from ml.models.anomaly import AnomalyDetector
from ml.types import AnomalyReport


@pytest.fixture
def history() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 500

    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-01",
                periods=rows,
                freq="10min",
            ),
            "avt_T55": rng.normal(
                350.0,
                2.0,
                rows,
            ),
            "hydro_T6": rng.normal(
                315.0,
                1.5,
                rows,
            ),
        }
    )


@pytest.fixture
def detector(
    history: pd.DataFrame,
) -> AnomalyDetector:
    return AnomalyDetector.fit(
        history,
        feature_columns=[
            "avt_T55",
            "hydro_T6",
        ],
        contamination=0.01,
    )


def test_normal_state_returns_valid_report(
    history: pd.DataFrame,
    detector: AnomalyDetector,
):
    prepared = build_anomaly_features(
        history,
        ["avt_T55", "hydro_T6"],
    )

    report = detector.score(
        prepared.tail(1)
    )

    assert isinstance(report, AnomalyReport)
    assert 0.0 <= report.anomaly_score <= 1.0
    assert isinstance(report.flagged_tags, list)
    assert isinstance(report.stale_tags, list)


def test_outlier_is_flagged(
    history: pd.DataFrame,
    detector: AnomalyDetector,
):
    changed = history.copy()
    changed.loc[
        changed.index[-1],
        "hydro_T6",
    ] = 500.0

    prepared = build_anomaly_features(
        changed,
        ["avt_T55", "hydro_T6"],
    )

    report = detector.score(
        prepared.tail(1)
    )

    assert report.is_anomaly is True
    assert report.is_out_of_envelope is True
    assert "hydro_T6" in report.flagged_tags
    assert report.anomaly_score >= 0.75


def test_stale_signal_is_detected(
    history: pd.DataFrame,
    detector: AnomalyDetector,
):
    changed = history.copy()
    changed.loc[
        changed.index[-12:],
        "avt_T55",
    ] = 350.0

    prepared = build_anomaly_features(
        changed,
        ["avt_T55", "hydro_T6"],
    )

    report = detector.score(
        prepared.tail(1)
    )

    assert report.is_anomaly is True
    assert "avt_T55" in report.stale_tags


def test_missing_feature_is_reported(
    detector: AnomalyDetector,
):
    state = pd.DataFrame(
        [{
            "date": pd.Timestamp(
                "2025-01-01"
            ),
            "avt_T55": 350.0,
        }]
    )

    report = detector.score(state)

    assert report.is_anomaly is True
    assert "hydro_T6" in report.flagged_tags


def test_save_and_load(
    detector: AnomalyDetector,
    history: pd.DataFrame,
    tmp_path,
):
    path = tmp_path / "anomaly.pkl"

    detector.save(str(path))
    loaded = AnomalyDetector.load(str(path))

    prepared = build_anomaly_features(
        history,
        ["avt_T55", "hydro_T6"],
    )

    report = loaded.score(
        prepared.tail(1)
    )

    assert isinstance(report, AnomalyReport)
    assert loaded.artifact["trained"] is True


@pytest.mark.parametrize(
    "invalid_state",
    [
        {},
        pd.DataFrame(),
        pd.DataFrame(
            [{"value": 1.0}, {"value": 2.0}]
        ),
    ],
)
def test_invalid_state_is_rejected(
    invalid_state,
):
    detector = AnomalyDetector()

    expected = (
        TypeError
        if isinstance(invalid_state, dict)
        else ValueError
    )

    with pytest.raises(expected):
        detector.score(invalid_state)   #type: ignore