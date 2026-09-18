import pandas as pd
from src.feature_engineering import build_features


def test_features_preserve_real_gap_and_impute_small_gap():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=5, freq="30min"), "equipment_id": ["A"] * 5, "flow_lps": [1, None, 3, 4, 5], "cooling_water_temp_c": [25] * 5, "building_load_rt": [100] * 5, "energy_kwh": [20, 21, 22, 23, 24], "outside_temp_f": [70] * 5, "dew_point_f": [60] * 5, "humidity_pct": [50] * 5, "wind_speed_mph": [4] * 5, "pressure_in": [30] * 5})
    features, report = build_features(frame)
    assert features.loc[1, "flow_lps"] == 2
    assert report["imputed"]["flow_lps"] == 1
    assert features.loc[1, "gap_minutes"] == 30
