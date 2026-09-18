import numpy as np
import pandas as pd
import xgboost
from src.pipeline import analyze


def test_pipeline_returns_scores_and_events():
    rng = np.random.default_rng(3)
    timestamps = pd.date_range("2026-01-01", periods=80, freq="30min")
    frame = pd.DataFrame({"timestamp": timestamps, "equipment_id": ["A"] * 80, "flow_lps": rng.normal(50, 2, 80), "cooling_water_temp_c": rng.normal(27, 1, 80), "building_load_rt": rng.normal(300, 20, 80), "energy_kwh": rng.normal(90, 4, 80), "outside_temp_f": rng.normal(72, 3, 80), "dew_point_f": rng.normal(60, 2, 80), "humidity_pct": rng.normal(55, 4, 80), "wind_speed_mph": rng.normal(8, 2, 80), "pressure_in": rng.normal(30, .1, 80)})
    result = analyze(frame)
    assert "anomaly_score" in result["scored"]
    assert result["scored"]["anomaly_score"].between(0, 1).all()
    assert result["model"]["model"] == "Isolation Forest"
    assert result["model"]["energy_baseline"]["model"] == "XGBoost Regressor"
    assert result["model"]["energy_baseline"]["status"] == "trained"
