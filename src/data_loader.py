from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import pandas as pd

CANONICAL_COLUMNS = {
    "timestamp": ["timestamp", "datetime", "date time", "time", "date"],
    "equipment_id": ["equipment_id", "equipment id", "chiller", "chiller_id", "unit", "asset"],
    "flow_lps": ["chilled water rate (l/sec)", "chilled water rate", "chilled_water_rate", "flow", "flow l/sec"],
    "cooling_water_temp_c": ["cooling water temperature (°c)", "cooling water temperature", "cooling_water_temperature", "cooling water temp", "condenser water temperature"],
    "building_load_rt": ["building load (rt)", "building load", "building_load", "load", "cooling load"],
    "energy_kwh": ["chiller energy consumption (kwh)", "chiller energy consumption", "energy consumption", "energy", "kwh"],
    "outside_temp_f": ["outside temperature (°f)", "outside temperature", "outside_temperature", "ambient temperature", "outdoor temperature"],
    "dew_point_f": ["dew point (°f)", "dew point", "dew_point"],
    "humidity_pct": ["humidity (%)", "humidity", "relative humidity", "humidity_pct"],
    "wind_speed_mph": ["wind speed (mph)", "wind speed", "wind_speed"],
    "pressure_in": ["pressure (in)", "pressure", "barometric pressure"],
}
REQUIRED = ["timestamp", "equipment_id", "flow_lps", "cooling_water_temp_c", "building_load_rt", "energy_kwh", "outside_temp_f", "dew_point_f", "humidity_pct", "wind_speed_mph", "pressure_in"]


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def detect_columns(columns) -> tuple[dict[str, str], list[str]]:
    lookup = {_key(col): col for col in columns}
    mapping = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in [canonical, *aliases]:
            if _key(alias) in lookup:
                mapping[canonical] = lookup[_key(alias)]
                break
    return mapping, [column for column in REQUIRED if column not in mapping]


def load_csv(source) -> tuple[pd.DataFrame, dict]:
    if isinstance(source, (str, Path)):
        frame = pd.read_csv(source)
        name = Path(source).name
    else:
        raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
        frame = pd.read_csv(BytesIO(raw))
        name = getattr(source, "name", "uploaded.csv")
    mapping, missing = detect_columns(frame.columns)
    if missing:
        raise ValueError("Missing mandatory columns: " + ", ".join(missing))
    renamed = frame.rename(columns={source_name: canonical for canonical, source_name in mapping.items()})
    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="coerce")
    renamed["equipment_id"] = renamed["equipment_id"].astype("string").str.strip()
    for column in REQUIRED:
        if column not in ("timestamp", "equipment_id"):
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed = renamed.sort_values(["equipment_id", "timestamp"], kind="stable").reset_index(drop=True)
    report = {"file_name": name, "mapping": mapping, "missing_columns": missing}
    return renamed, report


def validate_dataset(frame: pd.DataFrame) -> dict:
    duplicate_mask = frame.duplicated(["equipment_id", "timestamp"], keep=False)
    valid_timestamps = frame["timestamp"].dropna()
    gaps = frame.sort_values(["equipment_id", "timestamp"]).groupby("equipment_id")["timestamp"].diff().dt.total_seconds().div(60).dropna()
    numeric = [column for column in REQUIRED if column not in ("timestamp", "equipment_id")]
    missing = frame[REQUIRED].isna().sum().to_dict()
    invalid_timestamps = int(frame["timestamp"].isna().sum())
    return {
        "rows": len(frame),
        "equipment_count": int(frame["equipment_id"].nunique(dropna=True)),
        "equipment_ids": sorted(frame["equipment_id"].dropna().unique().tolist()),
        "start": valid_timestamps.min() if not valid_timestamps.empty else None,
        "end": valid_timestamps.max() if not valid_timestamps.empty else None,
        "missing": missing,
        "missing_cells": int(frame[REQUIRED].isna().sum().sum()),
        "missing_pct": float(frame[REQUIRED].isna().mean().mean() * 100),
        "duplicates": int(duplicate_mask.sum()),
        "invalid_timestamps": invalid_timestamps,
        "gap_count_over_90m": int((gaps > 90).sum()),
        "gap_max_minutes": float(gaps.max()) if not gaps.empty else 0.0,
        "gap_median_minutes": float(gaps.median()) if not gaps.empty else 0.0,
        "numeric_ranges": {column: (float(frame[column].min()), float(frame[column].max())) for column in numeric if frame[column].notna().any()},
    }
