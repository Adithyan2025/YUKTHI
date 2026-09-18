from __future__ import annotations

import numpy as np
import pandas as pd
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
from sklearn.ensemble import HistGradientBoostingRegressor


BASELINE_FEATURES = [
    "flow_lps", "cooling_water_temp_c", "building_load_rt", "outside_temp_c", "dew_point_c",
    "humidity_pct", "wind_speed_mph", "pressure_in", "hour", "day_of_week", "month",
    "is_weekend", "load_rolling_mean_6", "outside_to_cooling_temp_delta_c",
]


def run_energy_baseline(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = features.copy()
    columns = [column for column in BASELINE_FEATURES if column in data.columns]
    data["expected_energy_kwh"] = np.nan
    data["energy_residual_kwh"] = np.nan
    data["baseline_deviation_score"] = 0.0
    if len(data) < 20 or not columns:
        return data, {"model": "XGBoost Regressor", "status": "insufficient_data", "features": columns}

    usable = data[columns].replace([np.inf, -np.inf], np.nan).copy()
    usable = usable.fillna(usable.median(numeric_only=True))
    valid = data["energy_kwh"].notna() & usable.notna().all(axis=1)
    if valid.sum() < 20:
        return data, {"model": "XGBoost Regressor", "status": "insufficient_data", "features": columns}

    ordered = data.loc[valid].sort_values("timestamp")
    split = max(10, int(len(ordered) * 0.8))
    train_index = ordered.index[:split]
    model_name = "XGBoost Regressor" if XGBRegressor is not None else "Gradient Boosting fallback (install XGBoost for requested model)"
    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04, subsample=0.85,
            colsample_bytree=0.85, objective="reg:squarederror", random_state=42,
            n_jobs=2, verbosity=0,
        )
    else:
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    model.fit(usable.loc[train_index], data.loc[train_index, "energy_kwh"])
    predictions = model.predict(usable.loc[valid])
    data.loc[valid, "expected_energy_kwh"] = predictions
    data.loc[valid, "energy_residual_kwh"] = data.loc[valid, "energy_kwh"] - predictions
    residuals = data.loc[train_index, "energy_kwh"] - model.predict(usable.loc[train_index])
    center = float(np.nanmedian(residuals))
    scale = float(np.nanmedian(np.abs(residuals - center)) * 1.4826)
    scale = max(scale, 1e-6)
    data["baseline_deviation_score"] = ((data["energy_residual_kwh"] - center).abs() / (4 * scale)).clip(0, 1).fillna(0)
    evaluation = data.loc[ordered.index[split:], ["energy_kwh", "expected_energy_kwh"]].dropna()
    rmse = float(np.sqrt(np.mean((evaluation["energy_kwh"] - evaluation["expected_energy_kwh"]) ** 2))) if not evaluation.empty else None
    return data, {
        "model": model_name,
        "status": "trained",
        "features": columns,
        "training_rows": int(len(train_index)),
        "evaluation_rows": int(len(evaluation)),
        "chronological_split": "first 80% train, final 20% evaluation",
        "evaluation_rmse_kwh": rmse,
        "purpose": "predict expected energy under current load, weather, and operating context",
    }