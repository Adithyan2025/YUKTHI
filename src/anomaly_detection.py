from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .feature_engineering import model_columns


def run_anomaly_detection(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = features.copy()
    columns = model_columns(data)
    if len(data) < 12 or not columns:
        data["raw_anomaly_score"] = 0.0
        data["anomaly_score"] = 0.0
        data["is_anomaly"] = False
        return data, {"model": "Isolation Forest", "features": columns, "status": "insufficient_data"}
    usable = data[columns].replace([np.inf, -np.inf], np.nan)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", IsolationForest(n_estimators=250, contamination="auto", random_state=42, n_jobs=-1))])
    valid_mask = usable.notna().any(axis=1)
    pipeline.fit(usable.loc[valid_mask])
    raw = np.full(len(data), np.nan)
    raw[valid_mask] = -pipeline.decision_function(usable.loc[valid_mask])
    finite = pd.Series(raw).replace([np.inf, -np.inf], np.nan).dropna()
    low, high = finite.quantile(0.05), finite.quantile(0.95)
    data["raw_anomaly_score"] = raw
    data["anomaly_score"] = ((data["raw_anomaly_score"] - low) / (high - low if high > low else 1)).clip(0, 1).fillna(0)
    cutoff = float(data.loc[valid_mask, "anomaly_score"].quantile(0.95)) if valid_mask.any() else 1.0
    data["is_anomaly"] = data["anomaly_score"].ge(max(0.55, cutoff))
    return data, {"model": "Isolation Forest", "features": columns, "status": "trained", "reference_rows": int(valid_mask.sum()), "score_cutoff": max(0.55, cutoff), "validation": "chronological reference features; no random split or labelled accuracy claimed"}
