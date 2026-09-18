import pandas as pd
from src.data_loader import detect_columns, validate_dataset


def test_column_aliases_and_quality():
    columns = ["DateTime", "Chiller", "Flow", "Cooling Water Temperature", "Building Load", "Energy", "Outside Temperature", "Dew Point", "Humidity", "Wind Speed", "Pressure"]
    mapping, missing = detect_columns(columns)
    assert not missing
    assert mapping["timestamp"] == "DateTime"
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01", "2026-01-01 02:00"], format="mixed"), "equipment_id": ["A", "A"], "flow_lps": [1, 2], "cooling_water_temp_c": [25, 26], "building_load_rt": [100, 110], "energy_kwh": [20, 21], "outside_temp_f": [70, 71], "dew_point_f": [60, 61], "humidity_pct": [50, 51], "wind_speed_mph": [4, 5], "pressure_in": [30, 30]})
    quality = validate_dataset(frame)
    assert quality["gap_count_over_90m"] == 1
