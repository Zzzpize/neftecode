from __future__ import annotations

import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.schemas import Interval, QualityPrediction, Variant


def _load_registry(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rand_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


_FOCUSES = [
    ("safety", {"safety": 0.90, "yield": 0.45, "energy": 0.55, "wear": 0.65}),
    ("safety", {"safety": 0.85, "yield": 0.55, "energy": 0.60, "wear": 0.55}),
    ("safety", {"safety": 0.80, "yield": 0.60, "energy": 0.50, "wear": 0.60}),
    ("yield", {"safety": 0.55, "yield": 0.90, "energy": 0.50, "wear": 0.55}),
    ("yield", {"safety": 0.60, "yield": 0.85, "energy": 0.55, "wear": 0.60}),
    ("yield", {"safety": 0.65, "yield": 0.80, "energy": 0.60, "wear": 0.55}),
    ("energy", {"safety": 0.55, "yield": 0.50, "energy": 0.90, "wear": 0.60}),
    ("energy", {"safety": 0.60, "yield": 0.55, "energy": 0.85, "wear": 0.65}),
    ("wear", {"safety": 0.60, "yield": 0.60, "energy": 0.55, "wear": 0.90}),
    ("wear", {"safety": 0.55, "yield": 0.65, "energy": 0.60, "wear": 0.85}),
    ("balanced", {"safety": 0.70, "yield": 0.72, "energy": 0.68, "wear": 0.70}),
    ("balanced", {"safety": 0.75, "yield": 0.68, "energy": 0.70, "wear": 0.72}),
]


def _generate_action(
    row: pd.Series,
    controllable: dict[str, dict],
    focus: str,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    """Возвращает (action, delta) для указанного профиля."""
    action: dict[str, float] = {}
    delta: dict[str, float] = {}

    priority_tags = {
        "safety":   ["hydro_T5", "hydro_T6", "hydro_T11", "hydro_F14"],
        "yield":    ["avt_F30", "avt_F32", "avt_F65", "avt_T55"],
        "energy":   ["avt_F5", "avt_F26", "avt_F27", "avt_F28"],
        "wear":     ["avt_T55", "hydro_T5", "avt_F19"],
        "balanced": list(controllable.keys())[:5],
    }
    candidates = [t for t in priority_tags.get(focus, list(controllable.keys())) if t in controllable]
    picked = rng.choice(candidates, size=min(3, len(candidates)), replace=False) if candidates else []

    for tag in picked:
        cfg = controllable[tag]
        if tag not in row.index or pd.isna(row[tag]):
            continue
        current = float(row[tag])
        span = cfg["range_max"] - cfg["range_min"]
        if span <= 0:
            continue
        change = rng.normal(0, 0.035) * span
        new_val = float(np.clip(current + change, cfg["range_min"], cfg["range_max"]))
        if abs(new_val - current) > span * 0.005:
            action[tag] = round(new_val, 2)
            delta[tag] = round(new_val - current, 2)

    return action, delta


def _predict_sulfur(current: float, metrics: dict[str, float], rng: np.random.Generator) -> tuple[float, float, float]:
    """Прогноз серы: чем выше safety, тем ниже сера."""
    target = 6.0 + (1.0 - metrics["safety"]) * 6.0
    predicted = float(np.clip(target + rng.normal(0, 0.4), 1.5, current + 1))
    low = max(0.0, predicted - 1.2)
    high = predicted + 1.5
    return round(predicted, 2), round(low, 2), round(high, 2)


def _build_variant(
    row: pd.Series,
    controllable: dict[str, dict],
    focus: str,
    base_metrics: dict[str, float],
    rng: np.random.Generator,
    current_sulfur: float,
) -> Variant:
    action, delta = _generate_action(row, controllable, focus, rng)

    metrics = {
        k: float(np.clip(v + rng.normal(0, 0.02), 0.1, 1.0))
        for k, v in base_metrics.items()
    }
    metrics = {k: round(v, 3) for k, v in metrics.items()}

    predicted, low, high = _predict_sulfur(current_sulfur, metrics, rng)
    feasible = predicted < 10.0
    infeasible_reason = (
        f"прогноз серы {predicted:.1f} мг/кг превышает лимит 10" if not feasible else None
    )
    spec_risk = float(np.clip((predicted - 6) / 8, 0.0, 1.0))

    return Variant(
        id=_rand_id(),
        action=action,
        delta=delta,
        metrics=metrics,
        predicted=QualityPrediction(
            sulfur=Interval(mean=predicted, low=low, high=high, unit="ppm"),
            confidence="medium",
            spec_risk={"sulfur_over_10": round(spec_risk, 2)},
        ),
        feasible=feasible,
        infeasible_reason=infeasible_reason,
    )


def _pareto_filter(variants: list[Variant]) -> list[Variant]:
    keys = ("safety", "yield", "energy", "wear")
    keep: list[Variant] = []
    for i, v in enumerate(variants):
        dominated = False
        for j, u in enumerate(variants):
            if i == j:
                continue
            if all(u.metrics[k] >= v.metrics[k] for k in keys) and any(
                u.metrics[k] > v.metrics[k] for k in keys
            ):
                dominated = True
                break
        if not dominated:
            keep.append(v)
    return keep


def generate_pareto_front(
    state: pd.DataFrame,
    registry_path: Path,
    n_candidates: int | None = None,
    seed: int | None = None,
) -> list[Variant]:
    rng = np.random.default_rng(seed)
    registry = _load_registry(registry_path)
    controllable = {t: c for t, c in registry.items() if c.get("controllable")}

    row = state.iloc[0]
    current_sulfur = float(row.get("pak_sulfur_ppm", 8.0))
    if np.isnan(current_sulfur):
        current_sulfur = 8.0

    variants: list[Variant] = []
    for focus, base_metrics in _FOCUSES:
        variants.append(_build_variant(row, controllable, focus, base_metrics, rng, current_sulfur))

    feasible = [v for v in variants if v.feasible]
    front = _pareto_filter(feasible or variants)
    return front
