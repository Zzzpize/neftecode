"""Separate laboratory truth at sample time from operator availability."""
from __future__ import annotations

import pandas as pd

LIMS_DELAY_HOURS = 4.0
AVAILABILITY_MARKER = "ml_lims_delay_h"


def label_column(frame: pd.DataFrame, column: str) -> str:
    candidate = f"label__{column}"
    return candidate if candidate in frame.columns else column


def apply_lims_availability(frame: pd.DataFrame) -> pd.DataFrame:
    if AVAILABILITY_MARKER in frame.columns:
        if not frame[AVAILABILITY_MARKER].eq(LIMS_DELAY_HOURS).all():
            raise ValueError("Unsupported LIMS availability policy")
        if any(c.startswith("lims__") for c in frame) and not any(c.startswith("label__") for c in frame):
            raise ValueError("Training requires ml_training.parquet, not inference-only ml_features.parquet")
        return frame.copy()
    result = frame.sort_values("date").reset_index(drop=True).copy()
    result["date"] = pd.to_datetime(result["date"])
    updates = {}
    for column in frame.columns:
        if not column.startswith("lims__") or column.endswith("_age_h"):
            continue
        age_column = f"{column}_age_h"
        if age_column not in result:
            raise ValueError(f"Missing sample age: {age_column}")
        values = pd.to_numeric(result[column], errors="coerce")
        age = pd.to_numeric(result[age_column], errors="coerce")
        updates[f"label__{column}"] = values
        updates[f"label__{age_column}"] = age
        events = pd.DataFrame({
            "sample_time": (result["date"] - pd.to_timedelta(age, unit="h")).dt.round("s"),
            "value": values,
        }).dropna().drop_duplicates("sample_time", keep="last").sort_values("sample_time")
        events["available_time"] = events["sample_time"] + pd.Timedelta(hours=LIMS_DELAY_HOURS)
        known = pd.merge_asof(result[["date"]], events,
                              left_on="date", right_on="available_time", direction="backward")
        updates[column] = known["value"]
        updates[age_column] = (result["date"] - known["sample_time"]).dt.total_seconds() / 3600
    result = result.drop(columns=[c for c in updates if c in result])
    return pd.concat([result, pd.DataFrame(updates, index=result.index),
                      pd.DataFrame({AVAILABILITY_MARKER: LIMS_DELAY_HOURS}, index=result.index)], axis=1)


def inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in frame if c.startswith("label__")])


def validate_available_state(state: pd.DataFrame) -> None:
    if any(c.startswith("label__") for c in state):
        raise ValueError("Training labels must not be passed to predict; use inference_frame")
    if any(c.startswith("lims__") for c in state):
        if AVAILABILITY_MARKER not in state or not state[AVAILABILITY_MARKER].eq(LIMS_DELAY_HOURS).all():
            raise ValueError("Rebuild ml_features.parquet: LIMS availability policy is missing")
