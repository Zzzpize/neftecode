"""Feature registry loading, validation, and derived-feature synchronization."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


REGISTRY_PATH = Path(__file__).with_name("feature_registry.yaml")


class FeatureRegistryError(ValueError):
    """The feature registry is missing required or inconsistent metadata."""


def load_feature_registry(
    path: str | Path = REGISTRY_PATH,
    *,
    validate: bool = True,
) -> dict[str, dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise FeatureRegistryError("feature registry must contain a mapping")

    registry: dict[str, dict[str, Any]] = {}
    for name, metadata in raw.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise FeatureRegistryError(f"invalid registry entry: {name!r}")
        registry[name] = metadata

    if validate:
        validate_feature_registry(registry)
    return registry


def validate_feature_registry(
    registry: dict[str, dict[str, Any]],
    *,
    required_columns: Iterable[str] = (),
) -> None:
    required_fields = {
        "source",
        "unit",
        "role",
        "controllable",
        "range_min",
        "range_max",
    }

    for name, metadata in registry.items():
        missing = required_fields - metadata.keys()
        if missing:
            raise FeatureRegistryError(
                f"{name}: missing metadata fields: {', '.join(sorted(missing))}"
            )

        if metadata["role"] not in {"feature", "target"}:
            raise FeatureRegistryError(f"{name}: role must be feature or target")

        if not isinstance(metadata["controllable"], bool):
            raise FeatureRegistryError(f"{name}: controllable must be boolean")

        minimum = metadata["range_min"]
        maximum = metadata["range_max"]
        if minimum is not None and maximum is not None:
            if not math.isfinite(float(minimum)) or not math.isfinite(float(maximum)):
                raise FeatureRegistryError(f"{name}: ranges must be finite")
            if float(minimum) > float(maximum):
                raise FeatureRegistryError(f"{name}: range_min exceeds range_max")

        if metadata["controllable"] and (minimum is None or maximum is None):
            raise FeatureRegistryError(f"{name}: controllable feature needs a range")

    absent = sorted(set(required_columns) - registry.keys())
    if absent:
        raise FeatureRegistryError(
            "columns are absent from feature registry: " + ", ".join(absent)
        )


def model_feature_names(
    model: str,
    path: str | Path = REGISTRY_PATH,
) -> list[str]:
    registry = load_feature_registry(path)
    return [
        name
        for name, metadata in registry.items()
        if metadata["role"] == "feature" and model in metadata.get("used_by", [])
    ]


def engineered_time_feature_sources(
    path: str | Path = REGISTRY_PATH,
) -> list[str]:
    registry = load_feature_registry(path)
    return [
        name
        for name, metadata in registry.items()
        if metadata["role"] == "feature"
        and metadata.get("engineer_time_features") is True
    ]


def controllable_ranges(
    tags: Iterable[str] | None = None,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, tuple[float, float]]:
    registry = load_feature_registry(path)
    requested = set(tags) if tags is not None else None
    result: dict[str, tuple[float, float]] = {}

    for name, metadata in registry.items():
        short_name = str(metadata.get("tag", name))
        if requested is not None and name not in requested and short_name not in requested:
            continue
        if not metadata["controllable"]:
            continue
        result[name] = (float(metadata["range_min"]), float(metadata["range_max"]))

    if requested is not None:
        resolved = set(result) | {
            str(registry[name].get("tag", name)) for name in result
        }
        missing = sorted(requested - resolved)
        if missing:
            raise FeatureRegistryError(
                "unknown or non-controllable tags: " + ", ".join(missing)
            )

    return result


def sync_feature_registry(
    frame: pd.DataFrame,
    *,
    avt_delay_steps: int,
    path: str | Path = REGISTRY_PATH,
) -> dict[str, dict[str, Any]]:
    """Adds all model inputs/targets and refreshes their observed ranges."""

    destination = Path(path)
    registry = load_feature_registry(destination, validate=False)

    for name, metadata in registry.items():
        metadata.setdefault("controllable", False)
        metadata.setdefault("unit", "unknown")
        metadata.setdefault("role", "feature")
        metadata.setdefault("source", "unknown")
        if (
            name in frame.columns
            and (
                "range_min" not in metadata
                or "range_max" not in metadata
            )
        ):
            numeric = pd.to_numeric(frame[name], errors="coerce").dropna()
            minimum = float(numeric.quantile(0.01)) if not numeric.empty else None
            maximum = float(numeric.quantile(0.99)) if not numeric.empty else None
            metadata.setdefault("range_min", minimum)
            metadata.setdefault("range_max", maximum)
        else:
            metadata.setdefault("range_min", None)
            metadata.setdefault("range_max", None)

    for name, metadata in registry.items():
        used_by: list[str] = []
        if name.startswith("avt_") and metadata["role"] == "feature":
            used_by.extend(["quality_avt", "anomaly"])
        elif name.startswith("hydro_") and metadata["role"] == "feature":
            used_by.extend(["quality_hydro", "anomaly"])
        metadata["used_by"] = used_by

    for name in ("pak_sulfur_ppm", "pak_d15"):
        if name in registry:
            registry[name]["used_by"] = ["quality_hydro"]

    target_specs = {
        "lims__авт__pt1__50_t": ("C", "quality_avt"),
        "lims__авт__pt1__90_t": ("C", "quality_avt"),
        "lims__авт__pt1__d15": ("kg/m3", "quality_avt"),
        "lims__авт__pt1__cfpp": ("C", "quality_avt"),
        "lims__гидроочистка__pt2__mg_sulfur": ("ppm", "quality_hydro"),
        "lims__гидроочистка__pt2__50_t": ("C", "quality_hydro"),
        "lims__гидроочистка__pt2__90_t": ("C", "quality_hydro"),
        "lims__гидроочистка__pt2__d15": ("kg/m3", "quality_hydro"),
    }

    for name, (unit, model) in target_specs.items():
        if name not in frame.columns:
            continue
        registry[name] = _metadata_for_series(
            frame[name],
            source="lims_long.parquet",
            unit=unit,
            role="target",
            used_by=[model],
            description=f"Laboratory target {name}",
        )

        age_name = f"{name}_age_h"
        if age_name in frame.columns:
            registry[age_name] = _metadata_for_series(
                frame[age_name],
                source="derived:time_join",
                unit="h",
                role="feature",
                used_by=[model],
                description=f"Age of latest laboratory value for {name}",
            )

    for name in frame.columns:
        used_by = _derived_feature_consumers(name, avt_delay_steps)
        if not used_by:
            continue
        source_column = _source_column(name)
        if source_column in registry and ("_lag_" in name or "_roll_" in name):
            registry.get(source_column, {})["engineer_time_features"] = True
        unit = _derived_unit(name, registry.get(source_column, {}))
        registry[name] = _metadata_for_series(
            frame[name],
            source=f"derived:{source_column}",
            unit=unit,
            role="feature",
            used_by=used_by,
            description=f"Causal derived feature from {source_column}",
        )

    anomaly_sources = [
        name
        for name, metadata in registry.items()
        if "anomaly" in metadata.get("used_by", [])
    ]
    for source_column in anomaly_sources:
        for window_name in ("30m", "60m", "120m"):
            name = f"anomaly_z__{window_name}__{source_column}"
            registry[name] = {
                "controllable": False,
                "description": f"Causal rolling z-score for {source_column}",
                "range_max": None,
                "range_min": None,
                "role": "feature",
                "source": f"runtime:{source_column}",
                "unit": "dimensionless",
                "used_by": ["anomaly_context"],
            }

        name = f"anomaly_stale_h__{source_column}"
        registry[name] = {
            "controllable": False,
            "description": f"Hours since last change of {source_column}",
            "range_max": None,
            "range_min": 0.0,
            "role": "feature",
            "source": f"runtime:{source_column}",
            "unit": "h",
            "used_by": ["anomaly_context"],
        }

    validate_feature_registry(registry)
    destination.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return registry


def _metadata_for_series(
    series: pd.Series,
    *,
    source: str,
    unit: str,
    role: str,
    used_by: list[str],
    description: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        minimum = maximum = None
    else:
        minimum = float(numeric.quantile(0.01))
        maximum = float(numeric.quantile(0.99))

    return {
        "controllable": False,
        "description": description,
        "range_max": maximum,
        "range_min": minimum,
        "role": role,
        "source": source,
        "unit": unit,
        "used_by": used_by,
    }


def _derived_feature_consumers(name: str, avt_delay_steps: int) -> list[str]:
    if name.startswith(("avt_lag_10m__", "avt_lag_30m__", "avt_lag_60m__", "avt_roll_")):
        return ["quality_avt"]
    if name == "avt_runtime_proxy_h":
        return ["quality_avt"]
    if name.startswith(("hydro_lag_", "hydro_roll_")):
        return ["quality_hydro"]
    if name == "hydro_catalyst_runtime_proxy_h":
        return ["quality_hydro"]
    if name.startswith(f"avt_lag_{avt_delay_steps}__"):
        return ["quality_hydro"]
    if name.startswith("anomaly_"):
        return ["anomaly_context"]
    return []


def _source_column(name: str) -> str:
    if name.startswith("anomaly_z__"):
        return name.split("__", maxsplit=2)[-1]
    if name.startswith("anomaly_stale_h__"):
        return name.removeprefix("anomaly_stale_h__")
    if "__" in name:
        prefix, short_name = name.rsplit("__", maxsplit=1)
        installation = prefix.split("_", maxsplit=1)[0]
        return f"{installation}_{short_name}"
    if name == "avt_runtime_proxy_h":
        return "date"
    if name == "hydro_catalyst_runtime_proxy_h":
        return "date"
    return name


def _derived_unit(name: str, source_metadata: dict[str, Any]) -> str:
    if name.startswith("anomaly_z__"):
        return "dimensionless"
    if name.startswith("anomaly_stale_h__") or name.endswith("runtime_proxy_h"):
        return "h"
    unit = str(source_metadata.get("unit", "unknown"))
    if re.search(r"_roll_slope_", name):
        return f"{unit}/10min"
    return unit
