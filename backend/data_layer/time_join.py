"""Синхронизация телеметрии, ПАК и ЛИМС по времени в единый master frame."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_MASTER_NAME = "master.parquet"
_LIMS_WIDE_NAME = "lims_wide.parquet"


def build_master_frame(cache_dir: Path, force: bool = False) -> Path:
    """Собирает master frame: телеметрия + ПАК + ЛИМС с возрастом.

    Требует, чтобы loaders уже отработали и в cache_dir лежали parquet исходников.
    """
    cache_dir = Path(cache_dir)
    dst = cache_dir / _MASTER_NAME

    sources = [
        cache_dir / "avt_tags.parquet",
        cache_dir / "hydro_tags.parquet",
        cache_dir / "pak_sulfur.parquet",
        cache_dir / "pak_d15.parquet",
        cache_dir / "lims_long.parquet",
    ]
    if not force and dst.exists():
        dst_m = dst.stat().st_mtime
        if all(dst_m >= s.stat().st_mtime for s in sources):
            log.info("master frame cached: %s", dst)
            return dst

    log.info("building master frame")
    avt = pd.read_parquet(cache_dir / "avt_tags.parquet")
    hydro = pd.read_parquet(cache_dir / "hydro_tags.parquet")

    avt = avt.rename(columns={c: f"avt_{c}" for c in avt.columns if c != "date"})
    hydro = hydro.rename(columns={c: f"hydro_{c}" for c in hydro.columns if c != "date"})

    master = pd.merge(avt, hydro, on="date", how="outer").sort_values("date")

    pak_sulfur = pd.read_parquet(cache_dir / "pak_sulfur.parquet").sort_values("date")
    pak_sulfur = pak_sulfur.rename(columns={"value": "pak_sulfur_ppm"})
    master = pd.merge_asof(
        master, pak_sulfur, on="date",
        tolerance=pd.Timedelta("10min"), direction="backward",
    )

    pak_d15 = pd.read_parquet(cache_dir / "pak_d15.parquet").sort_values("date")
    pak_d15 = pak_d15.rename(columns={"value": "pak_d15"})
    master = pd.merge_asof(
        master, pak_d15, on="date",
        tolerance=pd.Timedelta("10min"), direction="backward",
    )

    lims_wide = _build_lims_wide(cache_dir, force)
    master = _asof_merge_lims(master, lims_wide)

    master = master.reset_index(drop=True)
    master.to_parquet(dst, index=False)
    log.info("wrote %s: %d rows, %d cols", dst.name, len(master), master.shape[1])
    return dst


def _build_lims_wide(cache_dir: Path, force: bool) -> pd.DataFrame:
    dst = cache_dir / _LIMS_WIDE_NAME
    src = cache_dir / "lims_long.parquet"

    if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(dst)

    log.info("pivoting LIMS to wide format")
    lims = pd.read_parquet(src)
    lims["key"] = lims.apply(_lims_key, axis=1)
    wide = lims.pivot_table(
        index="date", columns="key", values="value", aggfunc="last",
    ).reset_index().sort_values("date")
    wide.to_parquet(dst, index=False)
    log.info("wrote %s: %d rows, %d indicators", dst.name, len(wide), wide.shape[1] - 1)
    return wide


def _lims_key(row: pd.Series) -> str:
    inst = _slug(row["installation"] or "unknown")
    point = _slug(str(row["sampling_point"] or "0"))
    indicator = _slug(row["indicator"] or "unknown")
    return f"lims__{inst}__pt{point}__{indicator}"


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _asof_merge_lims(master: pd.DataFrame, lims_wide: pd.DataFrame) -> pd.DataFrame:
    """Для каждой ЛИМС-колонки asof-merge backward и колонка возраста в часах."""
    if lims_wide.empty or lims_wide.shape[1] <= 1:
        return master

    for col in lims_wide.columns:
        if col == "date":
            continue
        sub = lims_wide[["date", col]].dropna().sort_values("date")
        if sub.empty:
            continue
        sub = sub.rename(columns={"date": f"{col}_ts"})
        master = pd.merge_asof(
            master.sort_values("date"),
            sub,
            left_on="date",
            right_on=f"{col}_ts",
            direction="backward",
        )
        age = (master["date"] - master[f"{col}_ts"]).dt.total_seconds() / 3600.0
        master[f"{col}_age_h"] = age
        master = master.drop(columns=[f"{col}_ts"])

    return master


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from app.config import settings
    build_master_frame(settings.data_dir)


if __name__ == "__main__":
    main()
