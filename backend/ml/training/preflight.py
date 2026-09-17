"""Read-only audit before rebuilding features/training."""
import json
import pandas as pd
from ml.paths import DATA_DIR
from ml.models.quality_avt import TARGET_CONFIG as AVT_TARGETS
from ml.models.quality_hydro import TARGET_CONFIG as HYDRO_TARGETS
from ml.vak import VAKCatalog


def main():
    frame = pd.read_parquet(DATA_DIR / "master.parquet")
    date = pd.to_datetime(frame["date"])
    train = date.ge("2023-01-01") & date.lt("2025-01-01")
    report = {"reference_version": "expert_xlsx_2026_09_17", "targets": {}, "missing_vak_inputs": {}}
    for model, configs in (("quality_avt", AVT_TARGETS), ("quality_hydro", HYDRO_TARGETS)):
        for name, config in configs.items():
            column = config["target_column"]
            valid = frame[column].notna() if column in frame else pd.Series(False, index=frame.index)
            age_name = config.get("age_column")
            if age_name:
                age = pd.to_numeric(frame.get(age_name, pd.Series(float("nan"), index=frame.index)), errors="coerce")
                valid &= age.le(.25) & (age.shift().isna() | age.diff().lt(0) | age.eq(0))
            valid &= train
            if column in frame and config.get("absolute_range"):
                valid &= pd.to_numeric(frame[column], errors="coerce").between(*config["absolute_range"])
            n = int(valid.sum())
            report["targets"][f"{model}.{name}"] = {"n_train_labels": n, "enough_for_ml": n >= 100,
                                                          "baseline": config.get("vak_name") or "train_median"}
    catalog = VAKCatalog.load_expert()
    for name in catalog.names:
        missing = sorted(set(catalog.get(name).required_columns) - set(frame))
        if missing:
            report["missing_vak_inputs"][name] = missing
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
