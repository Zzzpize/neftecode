"""Read-only helpers for the ML demonstration; never calibrate on test labels."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from ml.availability import label_column
from ml.validation.walk_forward import (
    _batch_prediction_arrays, _fresh_label_mask, _physical_target_mask,
)


def prediction_observations(model, frame, target, config) -> pd.DataFrame:
    evaluation = frame.sort_values("date").reset_index(drop=True)
    columns = ["date", "actual", "baseline", "prediction", "low", "high", "source_position"]
    mask = _fresh_label_mask(evaluation, target_column=config["target_column"],
                             age_column=config.get("age_column"))
    positions = np.flatnonzero(mask.to_numpy())
    if not positions.size:
        return pd.DataFrame(columns=columns)
    baseline, prediction, low, high = _batch_prediction_arrays(
        model, evaluation, target=target, config=config)
    actual = pd.to_numeric(evaluation.iloc[positions][label_column(
        evaluation, config["target_column"])], errors="coerce").to_numpy(dtype=float)
    valid = _physical_target_mask(actual, config) & np.isfinite(actual) & np.isfinite(prediction[positions])
    positions, actual = positions[valid], actual[valid]
    # Unknown interval is kept as NaN, not replaced with a fictitious narrow band.
    return pd.DataFrame({"date": evaluation.iloc[positions]["date"].to_numpy(),
                         "actual": actual, "baseline": baseline[positions],
                         "prediction": prediction[positions], "low": low[positions],
                         "high": high[positions], "source_position": positions})


def acceptance_table(report: dict) -> pd.DataFrame:
    """Per-target checks: missing evidence must not count as a passed criterion."""
    rows = []

    def check(name, value, passed):
        known = isinstance(value, (int, float)) and math.isfinite(value)
        rows.append({"criterion": name, "value": value,
                     "status": ("PASS" if passed else "FAIL") if known else "NOT VERIFIED"})

    for model in ("quality_avt", "quality_hydro"):
        for target, metrics in report.get(model, {}).items():
            prefix = f"{model}.{target}"
            coverage = metrics.get("coverage_90")
            check(f"{prefix}: coverage 85–95%", coverage,
                  coverage is not None and .85 <= coverage <= .95)
            mae, baseline = metrics.get("mae"), metrics.get("baseline_mae")
            gain = baseline - mae if mae is not None and baseline is not None else None
            check(f"{prefix}: MAE improvement over reported baseline", gain,
                  gain is not None and gain > 0)
    sulfur = report.get("quality_hydro", {}).get("sulfur_ppm", {})
    recall = sulfur.get("spec_recall")
    # No positive examples cannot demonstrate recall.
    if sulfur.get("n_actual_over_spec", 0) == 0:
        recall = None
    check("sulfur: recall ≥90% (PAK reference)", recall, recall is not None and recall >= .9)
    return pd.DataFrame(rows, columns=["criterion", "value", "status"])
