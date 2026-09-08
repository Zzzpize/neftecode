"""Чтение исходников хакатона в parquet-кеш."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_all(hackathon_dir: Path, cache_dir: Path, force: bool = False) -> dict[str, Path]:
    """Собирает все источники в parquet. Возвращает пути к результирующим файлам."""
    hackathon_dir = Path(hackathon_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    outputs["avt"] = _load_telemetry(
        hackathon_dir / "data" / "avt_tags.csv",
        cache_dir / "avt_tags.parquet",
        force,
    )
    outputs["hydro"] = _load_telemetry(
        hackathon_dir / "data" / "242000_tags.csv",
        cache_dir / "hydro_tags.parquet",
        force,
    )
    pak = _load_pak(_find(hackathon_dir, "Выгрузка ПАК*.xlsx"), cache_dir, force)
    outputs.update(pak)
    outputs["lims"] = _load_lims(
        _find(hackathon_dir, "ЛИМСы*.xlsx"),
        cache_dir / "lims_long.parquet",
        force,
    )
    outputs["tag_dict"] = _load_tag_dict(
        _find(hackathon_dir, "Теги_хакатон.xlsx"),
        cache_dir / "tag_dict.parquet",
        force,
    )
    outputs["vac_formulas"] = _load_vac_formulas(
        _find(hackathon_dir, "Теги_хакатон.xlsx"),
        cache_dir / "vac_formulas.parquet",
        force,
    )

    return outputs


def _find(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} не найден в {directory}")
    return matches[0]


def _needs_rebuild(src: Path, dst: Path, force: bool) -> bool:
    if force or not dst.exists():
        return True
    return src.stat().st_mtime > dst.stat().st_mtime


def _load_telemetry(src: Path, dst: Path, force: bool) -> Path:
    if not _needs_rebuild(src, dst, force):
        log.info("telemetry cached: %s", dst.name)
        return dst
    log.info("loading telemetry: %s", src.name)
    df = pd.read_csv(src)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(dst, index=False)
    log.info("wrote %s: %d rows, %d cols", dst.name, len(df), df.shape[1])
    return dst


def _load_pak(src: Path, cache_dir: Path, force: bool) -> dict[str, Path]:
    sulfur_dst = cache_dir / "pak_sulfur.parquet"
    d15_dst = cache_dir / "pak_d15.parquet"

    if not force and sulfur_dst.exists() and d15_dst.exists():
        src_m = src.stat().st_mtime
        if sulfur_dst.stat().st_mtime > src_m and d15_dst.stat().st_mtime > src_m:
            log.info("PAK cached")
            return {"pak_sulfur": sulfur_dst, "pak_d15": d15_dst}

    log.info("loading PAK: %s", src.name)
    raw = pd.read_excel(src, header=None, skiprows=2)

    sulfur = pd.DataFrame({
        "date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
        "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
    }).dropna().sort_values("date").reset_index(drop=True)

    d15 = pd.DataFrame({
        "date": pd.to_datetime(raw.iloc[:, 3], errors="coerce"),
        "value": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
    }).dropna().sort_values("date").reset_index(drop=True)

    sulfur.to_parquet(sulfur_dst, index=False)
    d15.to_parquet(d15_dst, index=False)
    log.info("wrote pak_sulfur: %d rows; pak_d15: %d rows", len(sulfur), len(d15))
    return {"pak_sulfur": sulfur_dst, "pak_d15": d15_dst}


def _load_lims(src: Path, dst: Path, force: bool) -> Path:
    """ЛИМС: блочная разметка (Установка, точка отбора, показатель), пары колонок (date, value)."""
    if not _needs_rebuild(src, dst, force):
        log.info("LIMS cached: %s", dst.name)
        return dst

    log.info("loading LIMS: %s", src.name)
    header = pd.read_excel(src, header=None, nrows=3)
    data = pd.read_excel(src, header=None, skiprows=4)

    blocks = header.iloc[0].ffill()
    indicators = header.iloc[1]
    units = header.iloc[2]

    rows: list[dict] = []
    n_cols = data.shape[1]
    for col in range(0, n_cols, 2):
        if col + 1 >= n_cols or pd.isna(indicators.iloc[col]):
            continue
        block = str(blocks.iloc[col])
        indicator = str(indicators.iloc[col]).strip()
        unit = str(units.iloc[col]).strip() if not pd.isna(units.iloc[col]) else None

        ts_series = pd.to_datetime(data.iloc[:, col], errors="coerce")
        val_series = pd.to_numeric(data.iloc[:, col + 1], errors="coerce")

        mask = ts_series.notna() & val_series.notna()
        for ts, val in zip(ts_series[mask], val_series[mask]):
            rows.append({
                "date": ts,
                "block": block,
                "indicator": indicator,
                "unit": unit,
                "value": float(val),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["installation"] = df["block"].apply(_parse_installation)
        df["sampling_point"] = df["block"].apply(_parse_sampling_point)
        df["product"] = df["block"].apply(_parse_product)
        df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(dst, index=False)
    log.info("wrote %s: %d rows", dst.name, len(df))
    return dst


_INSTALLATION_RE = re.compile(r"Установка\s*'([^']+)'")
_POINT_RE = re.compile(r"Точка отбора\s*'([^']+)'")
_PRODUCT_RE = re.compile(r"Продукт\s*'([^']+)'")


def _parse_installation(block: str) -> str | None:
    m = _INSTALLATION_RE.search(block)
    return m.group(1) if m else None


def _parse_sampling_point(block: str) -> str | None:
    m = _POINT_RE.search(block)
    return m.group(1) if m else None


def _parse_product(block: str) -> str | None:
    m = _PRODUCT_RE.search(block)
    return m.group(1) if m else None


def _load_tag_dict(src: Path, dst: Path, force: bool) -> Path:
    if not _needs_rebuild(src, dst, force):
        log.info("tag_dict cached")
        return dst

    log.info("loading tag dictionary: %s", src.name)
    kip = pd.read_excel(src, sheet_name="КИП")

    rows: list[dict] = []
    for _, row in kip.iterrows():
        avt_desc = row.get("АВТ (описание)")
        avt_tag = row.get("АВТ")
        if isinstance(avt_tag, str) and avt_tag.strip():
            rows.append({
                "installation": "АВТ",
                "tag": avt_tag.strip(),
                "description": str(avt_desc).strip() if not pd.isna(avt_desc) else None,
            })
        hydro_desc = row.get("24-2000 (описание)")
        hydro_tag = row.get("24-2000")
        if isinstance(hydro_tag, str) and hydro_tag.strip():
            rows.append({
                "installation": "24-2000",
                "tag": hydro_tag.strip(),
                "description": str(hydro_desc).strip() if not pd.isna(hydro_desc) else None,
            })

    df = pd.DataFrame(rows)
    df["kind"] = df["tag"].str[0].map({
        "T": "temperature",
        "P": "pressure",
        "F": "flow_vol",
        "W": "flow_mass",
        "L": "level",
        "D": "density",
    })
    df.to_parquet(dst, index=False)
    log.info("wrote %s: %d tags", dst.name, len(df))
    return dst


def _load_vac_formulas(src: Path, dst: Path, force: bool) -> Path:
    if not _needs_rebuild(src, dst, force):
        return dst

    log.info("loading VAC formulas: %s", src.name)
    vac = pd.read_excel(src, sheet_name="ВАК", header=None)

    rows: list[dict] = []
    n_cols = vac.shape[1]
    for col in range(0, n_cols, 2):
        if col + 1 >= n_cols:
            continue
        header_val = vac.iloc[0, col]
        block = str(header_val) if not pd.isna(header_val) else None
        for r in range(1, len(vac)):
            name = vac.iloc[r, col]
            formula = vac.iloc[r, col + 1]
            if pd.isna(name) or pd.isna(formula):
                continue
            rows.append({
                "block": block,
                "name": str(name).strip(),
                "formula": str(formula).strip(),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    log.info("wrote %s: %d formulas", dst.name, len(df))
    return df if isinstance(df, Path) else dst


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from app.config import settings
    outputs = load_all(settings.hackathon_dir, settings.data_dir)
    for key, path in outputs.items():
        print(f"  {key:15s} -> {path}")


if __name__ == "__main__":
    main()
