"""Adjustable ML-only demonstrations; no agents/HTTP/frontend involved."""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import pandas as pd

from ml.models import QualityAVTModel, QualityHydroModel, BlendingModel
from ml.optimizer import ParetoOptimizer
from ml.paths import DATA_DIR, ARTIFACT_DIR
from ml.scenario_blending import default_scenario


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="2025-08-01 12:00:00")
    parser.add_argument("--feed-sulfur-multiplier", type=float, default=1.)
    parser.add_argument("--sulfur-limit", type=float, default=10.)
    parser.add_argument("--t95-max", type=float, default=360.)
    parser.add_argument("--cetane-min", type=float, default=51.)
    parser.add_argument("--horizon-minutes", type=float, default=180.)
    parser.add_argument("--scenario", type=Path)
    args = parser.parse_args()
    if args.feed_sulfur_multiplier <= 0 or not 0 <= args.horizon_minutes <= 180:
        parser.error("Feed multiplier must be positive and horizon between 0 and 180 minutes")
    frame = pd.read_parquet(DATA_DIR / "ml_features.parquet")
    state = frame.loc[pd.to_datetime(frame["date"]).le(pd.Timestamp(args.timestamp))].tail(1).reset_index(drop=True)
    if state.empty:
        parser.error("No data at/before timestamp")
    state["ml_feed_sulfur_multiplier"] = args.feed_sulfur_multiplier
    state["ml_action_horizon_minutes"] = args.horizon_minutes
    scenario = json.loads(args.scenario.read_text()) if args.scenario else default_scenario()
    state["ml_blending_scenario"] = pd.Series([scenario])
    avt = QualityAVTModel.load(str(ARTIFACT_DIR / "quality_avt.pkl"))
    hydro = QualityHydroModel.load(str(ARTIFACT_DIR / "quality_hydro.pkl"))
    if any(m.artifact.get("reference_version") != "expert_xlsx_2026_09_17" for m in (avt, hydro)):
        parser.error("Artifacts are stale: run ml.training.train_all")
    optimizer = ParetoOptimizer(avt, hydro, BlendingModel())
    if hydro.artifact.get("sulfur_online"):
        history = frame.loc[pd.to_datetime(frame["date"]).lt(pd.Timestamp(state.iloc[0]["date"]))].tail(90)
        for i in range(len(history)):
            hydro.predict(history.iloc[[i]])
    constraints = optimizer.constraints_from_registry(
        hard={"sulfur_ppm": (0., args.sulfur_limit), "T95": (0., args.t95_max),
              "cetane_number": (args.cetane_min, 100.)}, max_deviation_pct=3.,
        tags=["hydro_T6", "hydro_F9", "hydro_P13"])
    variants = optimizer.find_pareto(state, constraints)
    print(json.dumps(clean_json({"basis": "model_scenario_not_causal_validation",
                                "current": asdict(hydro.predict(state)),
                                "variants": [asdict(v) for v in variants]}),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
