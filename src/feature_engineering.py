from __future__ import annotations
import numpy as np
import pandas as pd

BASE_NUMERIC = ["flow_lps", "cooling_water_temp_c", "building_load_rt", "energy_kwh", "outside_temp_f", "dew_point_f", "humidity_pct", "wind_speed_mph", "pressure_in"]


def build_features(frame: pd.DataFrame, interpolation_limit: int = 2) -> tuple[pd.DataFrame, dict]:
    data = frame.copy().sort_values(["equipment_id", "timestamp"], kind="stable").reset_index(drop=True)
    before = data[BASE_NUMERIC].isna().sum()
    data[BASE_NUMERIC] = data.groupby("equipment_id", group_keys=False)[BASE_NUMERIC].transform(lambda values: values.interpolate(limit=interpolation_limit, limit_direction="both"))
    after = data[BASE_NUMERIC].isna().sum()
    data["hour"] = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["outside_temp_c"] = (data["outside_temp_f"] - 32) * 5 / 9
    data["dew_point_c"] = (data["dew_point_f"] - 32) * 5 / 9
    data["outside_to_cooling_temp_delta_c"] = data["outside_temp_c"] - data["cooling_water_temp_c"]
    data["gap_minutes"] = data.groupby("equipment_id")["timestamp"].diff().dt.total_seconds().div(60)
    group = data.groupby("equipment_id", group_keys=False)
    for lag in [1, 2, 6, 12, 48]:
        data[f"energy_lag_{lag}"] = group["energy_kwh"].shift(lag)
        data[f"energy_lag_{lag}_valid"] = data["gap_minutes"].fillna(0).le(max(90, lag * 30 * 1.5)).astype(int)
    for column, prefix in [("energy_kwh", "energy"), ("building_load_rt", "load"), ("flow_lps", "flow"), ("cooling_water_temp_c", "cooling_temp")]:
        for window in [3, 6, 12, 48]:
            data[f"{prefix}_rolling_mean_{window}"] = group[column].transform(lambda values: values.rolling(window, min_periods=2).mean())
            data[f"{prefix}_rolling_std_{window}"] = group[column].transform(lambda values: values.rolling(window, min_periods=2).std())
        data[f"{prefix}_change"] = group[column].diff()
        data[f"{prefix}_pct_change"] = group[column].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    data["load_rolling_mean_6"] = group["building_load_rt"].transform(lambda values: values.rolling(6, min_periods=2).mean())
    data["energy_per_load"] = data["energy_kwh"] / data["building_load_rt"].replace(0, np.nan)
    data["energy_per_flow"] = data["energy_kwh"] / data["flow_lps"].replace(0, np.nan)
    data["load_per_flow"] = data["building_load_rt"] / data["flow_lps"].replace(0, np.nan)
    data["energy_kw"] = data["energy_kwh"] * 2
    data["estimated_cop"] = (data["building_load_rt"] * 3.517) / data["energy_kw"].replace(0, np.nan)
    data["cop_rolling_2h"] = np.nan
    for _, equipment_data in data.groupby("equipment_id", sort=False):
        ordered = equipment_data.sort_values("timestamp")
        rolling = ordered.set_index("timestamp")["estimated_cop"].rolling("2h", min_periods=1).mean()
        data.loc[rolling.index if rolling.index.isin(data.index).all() else ordered.index, "cop_rolling_2h"] = rolling.to_numpy()
    data["cop_alert"] = data["cop_rolling_2h"].lt(4.5).fillna(False)
    report = {"missing_before": before.to_dict(), "imputed": (before - after).clip(lower=0).to_dict(), "remaining_missing": after.to_dict(), "interpolation_limit_observations": interpolation_limit}
    return data, report


def model_columns(data: pd.DataFrame) -> list[str]:
    candidates = [
        "energy_kwh", "flow_lps", "cooling_water_temp_c", "building_load_rt", "outside_temp_f", "dew_point_f", "humidity_pct", "wind_speed_mph", "pressure_in",
        "hour", "day_of_week", "month", "is_weekend", "gap_minutes", "energy_per_load", "energy_per_flow", "load_per_flow",
        "outside_temp_c", "dew_point_c", "outside_to_cooling_temp_delta_c", "load_rolling_mean_6",
        "energy_change", "energy_pct_change", "load_change", "flow_change", "cooling_temp_change",
        "energy_rolling_mean_3", "energy_rolling_std_3", "energy_rolling_mean_12", "energy_rolling_std_12", "load_rolling_mean_12", "flow_rolling_mean_12",
    ]
    return [column for column in candidates if column in data.columns]
