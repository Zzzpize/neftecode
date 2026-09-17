"""Joint historical support of operating controls, fitted on train only."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from data_layer.feature_registry import controllable_ranges


def fit_envelope(frame: pd.DataFrame) -> dict:
    columns = [c for c in controllable_ranges() if c in frame]
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 100 or not columns:
        return {}
    center = data.median()
    scale = (data.quantile(.99) - data.quantile(.01)).clip(lower=1e-6)
    normalized = ((data - center) / scale).iloc[::max(1, len(data) // 5000)]
    model = NearestNeighbors(n_neighbors=2).fit(normalized.to_numpy())
    distances = model.kneighbors(normalized.to_numpy())[0][:, 1]
    return {"columns": columns, "center": center.to_dict(), "scale": scale.to_dict(),
            "threshold": max(float(np.quantile(distances, .99)) * 1.5, .05), "model": model}


def outside_envelope(state: pd.DataFrame, envelope: dict) -> bool:
    values = pd.to_numeric(state.reindex(columns=envelope["columns"]).iloc[0], errors="coerce")
    if not np.isfinite(values).all():
        return True
    normalized = (values - pd.Series(envelope["center"])) / pd.Series(envelope["scale"])
    distance = envelope["model"].kneighbors(normalized.to_numpy()[None, :], n_neighbors=1)[0][0, 0]
    return bool(distance > envelope["threshold"])
