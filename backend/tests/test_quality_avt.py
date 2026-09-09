import numpy as np
import pandas as pd
import pytest

import ml.models.quality_avt as quality_avt_module
from ml.models.quality_avt import QualityAVTModel
from ml.types import QualityPrediction
from ml.vak import VAKCatalog


def make_training_frame(n_rows: int = 40) -> pd.DataFrame:
    avt_f30 = np.linspace(10.0, 20.0, n_rows)
    baseline = 100.0 + 2.0 * avt_f30

    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-01",
                periods=n_rows,
                freq="10min",
            ),
            "avt_F30": avt_f30,
            "avt_T33": np.linspace(290.0, 310.0, n_rows),
            "lims__авт__pt1__50_t": baseline + 3.0,
            "lims__авт__pt1__50_t_age_h": 0.0,
        }
    )


def make_vak_catalog() -> VAKCatalog:
    return VAKCatalog.from_frame(
        pd.DataFrame(
            [
                {
                    "block": "ЭЛОУ-АВТ-6. 240-350",
                    "name": "AVT6:240-350:T50",
                    "formula": "100 + 2*F30",
                }
            ]
        )
    )


def test_avt_model_is_trained_on_vak_residual(monkeypatch):
    monkeypatch.setattr(
        quality_avt_module,
        "MIN_TRAIN_SAMPLES",
        10,
    )

    frame = make_training_frame()
    model = QualityAVTModel.fit(
        frame=frame,
        vak_catalog=make_vak_catalog(),
    )

    state = frame.iloc[[-1]][["avt_F30", "avt_T33"]]
    prediction = model.predict(state)

    assert isinstance(prediction, QualityPrediction)
    assert model.artifact["n_samples"]["T50"] == len(frame)
    assert 0.5 in model.artifact["models"]["T50"]

    interval = prediction.predictions["T50"]

    assert interval.low <= interval.mean <= interval.high
    assert interval.mean == pytest.approx(
        100.0 + 2.0 * state.iloc[0]["avt_F30"] + 3.0,
        abs=1.0,
    )


def test_predict_after_action_does_not_change_original_state(monkeypatch):
    monkeypatch.setattr(
        quality_avt_module,
        "MIN_TRAIN_SAMPLES",
        10,
    )

    frame = make_training_frame()
    model = QualityAVTModel.fit(
        frame=frame,
        vak_catalog=make_vak_catalog(),
    )

    state = frame.iloc[[-1]][["avt_F30", "avt_T33"]].copy()
    original_value = state.iloc[0]["avt_F30"]

    before = model.predict(state)
    after = model.predict_after_action(
        state,
        {"F30": original_value + 5.0},
    )

    assert state.iloc[0]["avt_F30"] == original_value
    assert (
        after.predictions["T50"].mean
        != before.predictions["T50"].mean
    )