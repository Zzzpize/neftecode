"""Bounded observational surrogate; effects are assumptions, not causal proof."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

CONTROLS = ("hydro_T6", "hydro_F9", "hydro_P13")
# Higher temperature/pressure improves desulfurization; higher feed rate
# reduces residence time. These signs are scenario assumptions.
SIGNS = (-1, 1, -1)


def fit_response(frame: pd.DataFrame) -> dict:
    if not set((*CONTROLS, "pak_sulfur_ppm")) <= set(frame):
        return {}
    data = frame.sort_values("date").reset_index(drop=True)
    sulfur = pd.to_numeric(data["pak_sulfur_ppm"], errors="coerce")
    sulfur = sulfur.where(sulfur.gt(0))
    log_sulfur = np.log(sulfur)
    lagged = pd.DataFrame(index=data.index)
    delays = {}
    scales = {}
    centers = {}
    for name in CONTROLS:
        values = pd.to_numeric(data[name], errors="coerce")
        scores = {lag: abs(values.diff().shift(lag).corr(log_sulfur.diff())) for lag in range(19)}
        valid_scores = {lag: score for lag, score in scores.items() if np.isfinite(score)}
        delay = max(valid_scores, key=valid_scores.get) if valid_scores else 0
        delays[name] = delay * 10
        centers[name] = float(values.median())
        scale = float(values.quantile(.99) - values.quantile(.01))
        scales[name] = max(scale, 1e-6)
        lagged[name] = (values.shift(delay) - centers[name]) / scales[name]
    valid = lagged.notna().all(axis=1) & log_sulfur.notna()
    if valid.sum() < 100:
        return {}
    x = lagged.loc[valid].to_numpy()
    y = log_sulfur.loc[valid].to_numpy()
    # Ridge regularization with sign constraints avoids a surrogate
    # recommending physically inverted effects learned from confounding.
    x = np.column_stack([np.ones(len(x)), x])
    design = np.vstack([x, np.diag([0., 1., 1., 1.])])
    result = lsq_linear(design, np.r_[y, np.zeros(4)],
                       bounds=([-np.inf, -np.inf, 0., -np.inf],
                               [np.inf, 0., np.inf, 0.]))
    return {"coefficients": dict(zip(CONTROLS, result.x[1:].tolist())),
            "scales": scales, "delay_minutes": delays, "default_horizon_minutes": 180,
            "basis": "sign_constrained_observational_log_sulfur_surrogate"}


def response_factor(state: pd.DataFrame, action: dict[str, float], response: dict) -> float:
    horizon = float(state.iloc[0].get("ml_action_horizon_minutes", 180))
    if not np.isfinite(horizon) or not 0 <= horizon <= 180:
        raise ValueError("Action horizon must be between 0 and 180 minutes")
    delta = 0.
    for name, coefficient in response.get("coefficients", {}).items():
        if name not in action:
            continue
        delay = response["delay_minutes"][name]
        progress = max(0., min(1., (horizon - delay) / max(180 - delay, 10)))
        delta += coefficient * (action[name] - float(state.iloc[0][name])) / response["scales"][name] * progress
    return float(np.exp(np.clip(delta, -2., 2.)))
