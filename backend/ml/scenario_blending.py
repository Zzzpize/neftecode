"""Explicit synthetic tank scenario, isolated from historical metrics."""
from __future__ import annotations
import json
import math
from pathlib import Path
from ml.types import Component, Interval, QualityPrediction


def default_scenario() -> dict:
    return json.loads(Path(__file__).with_name("blending_scenario.json").read_text())


def evaluate_blend(blending, prediction: QualityPrediction, config: dict,
                   stored_fraction: float, tank_share: float, additive_fraction: float):
    if not 0 <= stored_fraction <= float(config["max_stored_fraction"]) < 1:
        raise ValueError("Invalid stored fraction")
    if not 0 <= tank_share <= 1 or not 0 <= additive_fraction <= min(.03, float(config["max_additive_fraction"])):
        raise ValueError("Invalid tank share or additive dose (maximum 3%)")
    tanks = [Component(**tank) for tank in config["tanks"]]
    if len(tanks) != 2:
        raise ValueError("The baseline scenario requires two stored tanks")
    properties = ("sulfur_ppm", "T95", "cetane_number")
    for name in properties:
        iv = prediction.predictions.get(name)
        if iv is None or not all(math.isfinite(v) for v in (iv.low, iv.mean, iv.high)):
            prediction.confidence = "low"
            prediction.warnings.append(f"Blending unavailable: missing finite {name}")
            return prediction, {"cost": 1e6, "additive_fraction": additive_fraction}
    fractions = [1 - stored_fraction, stored_fraction * tank_share, stored_fraction * (1 - tank_share)]
    flow = float(config["hydro_mass_flow"])
    if not math.isfinite(flow) or flow <= 0:
        raise ValueError("hydro_mass_flow must be positive")
    # Component.mass_flow is the actual selected flow, not tank capacity.
    components = [Component("hydro_product", {}, flow), *tanks]
    total = flow / max(fractions[0], 1e-6)
    for component, fraction in zip(components, fractions):
        if component.name != "hydro_product" and total * fraction > component.mass_flow:
            return prediction, {"cost": 1e6, "tank_capacity_violation": 1.0}
        component.mass_flow = total * fraction
    output = {}
    for bound in ("low", "mean", "high"):
        components[0].properties = {p: getattr(prediction.predictions[p], bound) for p in properties}
        mixed = blending.blend(components, fractions).properties
        # Additive has zero sulfur; distillation effect is neglected in this scenario.
        mixed["sulfur_ppm"] *= 1 - additive_fraction
        mixed["cetane_number"] += 100 * additive_fraction * float(config["cetane_gain_per_additive_pct"])
        output[bound] = mixed
    result = QualityPrediction(
        predictions={p: Interval(output["mean"][p], output["low"][p], output["high"][p],
                                 prediction.predictions[p].unit) for p in properties},
        spec_risk={}, confidence="low",
        warnings=prediction.warnings + ["Synthetic tank blend; T95 and cetane mixing and additive efficiency are model assumptions"],
    )
    from ml.models.quality_hydro import QualityHydroModel
    result.spec_risk["sulfur_over_10"] = QualityHydroModel._sulfur_spec_risk(result.predictions["sulfur_ppm"], coverage=.9)
    sulfur_components = [prediction.predictions["sulfur_ppm"].mean, *[t.properties["sulfur_ppm"] for t in tanks]]
    weight = float(config["desulfurization_cost_weight"])
    diesel_cost = sum(f * (1 + weight * max(0., 1 - s / 10)) for f, s in zip(fractions, sulfur_components))
    cost = (1 - additive_fraction) * diesel_cost + additive_fraction * float(config["additive_cost_ratio"])
    return result, {"cost": cost, "additive_fraction": additive_fraction,
                    "stored_fraction": stored_fraction, "tank_share": tank_share,
                    "product_mass_flow": total / (1 - additive_fraction)}
