from __future__ import annotations

import pandas as pd


def _row(label: str, priority: str, evidence: str, action: str, score: float) -> dict:
    return {"hypothesis": label, "priority": priority, "evidence": evidence, "recommended_check": action, "signal_strength": round(min(score, 1.0), 2)}


def rank_hypotheses(event_rows: pd.DataFrame, history: pd.DataFrame, missing_pct: float) -> list[dict]:
    if event_rows.empty:
        return []
    event = event_rows
    observed_energy = float(event["energy_kwh"].median())
    expected = float(event["expected_energy_kwh"].median()) if "expected_energy_kwh" in event and event["expected_energy_kwh"].notna().any() else observed_energy
    residual_pct = abs(observed_energy - expected) / max(abs(expected), 1e-6)
    cop = float(event["cop_rolling_2h"].median()) if "cop_rolling_2h" in event and event["cop_rolling_2h"].notna().any() else None
    flow = float(event["flow_lps"].median())
    load = float(event["building_load_rt"].median())
    temp = float(event["cooling_water_temp_c"].median())
    rows = []

    if residual_pct >= 0.08 or (cop is not None and cop < 4.5):
        cop_text = f" Rolling COP is {cop:.2f}, below 4.5." if cop is not None and cop < 4.5 else ""
        rows.append(_row("Energy-efficiency deviation", "HIGH" if residual_pct >= 0.15 else "ATTENTION", f"Observed energy is {observed_energy:.2f} kWh versus expected {expected:.2f} kWh ({residual_pct * 100:.1f}% absolute deviation).{cop_text}", "Review the energy-to-load relationship, operating setpoints, and recent efficiency history.", min(1, residual_pct * 3 + (0.25 if cop is not None and cop < 4.5 else 0))))

    if "cooling_water_temp_c" in history and not history.empty:
        historical_temp = float(history["cooling_water_temp_c"].median())
        temp_delta = temp - historical_temp
        if abs(temp_delta) >= 2:
            rows.append(_row("Cooling-water condition", "ATTENTION", f"Event cooling-water temperature is {temp:.2f} C versus historical median {historical_temp:.2f} C ({temp_delta:+.2f} C).", "Review cooling-water supply, condenser conditions, and ambient conditions before concluding equipment degradation.", min(1, abs(temp_delta) / 5)))

    if "flow_lps" in history and not history.empty and "building_load_rt" in history:
        event_ratio = load / max(flow, 1e-6)
        historical_ratio = float((history["building_load_rt"] / history["flow_lps"].replace(0, pd.NA)).median())
        ratio_delta = abs(event_ratio - historical_ratio) / max(abs(historical_ratio), 1e-6)
        if ratio_delta >= 0.15:
            rows.append(_row("Flow/load relationship shift", "ATTENTION", f"Event load-to-flow ratio is {event_ratio:.2f}, compared with historical {historical_ratio:.2f} ({ratio_delta * 100:.1f}% relative shift).", "Inspect chilled-water flow readings, valves, pumps, and sensor calibration.", min(1, ratio_delta * 2)))

    if missing_pct >= 5 or event_rows[[column for column in ["flow_lps", "energy_kwh", "building_load_rt", "cooling_water_temp_c"] if column in event_rows]].isna().any().any():
        rows.append(_row("Data-quality or sensor issue", "ATTENTION", f"Missing data is {missing_pct:.1f}% across the dataset or event signals contain missing values.", "Validate sensor completeness and timestamp continuity before interpreting this event as physical equipment behaviour.", min(1, missing_pct / 20 + 0.2)))

    if not rows:
        rows.append(_row("Unclassified multivariate deviation", "ATTENTION", "Isolation Forest identified an unusual combination of operating variables, but no single diagnostic signal dominated.", "Review the event charts and operating history with an equipment specialist.", 0.55))
    return sorted(rows, key=lambda item: item["signal_strength"], reverse=True)[:3]


HYPOTHESIS_COMPONENTS = {
    "Energy-efficiency deviation": ("Compressor / condenser efficiency system", "Review compressor loading, condenser approach, setpoints, and energy meter behaviour."),
    "Cooling-water condition": ("Condenser and cooling-water circuit", "Inspect cooling-water temperature, condenser approach, tower operation, pumps, and valves."),
    "Flow/load relationship shift": ("Chilled-water flow circuit", "Inspect chilled-water pump, valves, flow sensor, and the relationship between flow and building load."),
    "Data-quality or sensor issue": ("Sensors and data acquisition", "Check sensor calibration, missing readings, timestamp continuity, and telemetry quality."),
    "Unclassified multivariate deviation": ("Multiple operating systems", "Review all event charts and operating logs with an equipment specialist."),
}


def summarize_whole_history(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event hypotheses; this ranks inspection areas, not confirmed failures."""
    if events.empty or "fault_hypotheses" not in events:
        return pd.DataFrame(columns=["equipment_id", "hypothesis", "likely_area_to_inspect", "events", "weighted_strength", "evidence", "recommended_check"])
    rows = []
    for _, event in events.iterrows():
        for item in event.get("fault_hypotheses", []) or []:
            component, check = HYPOTHESIS_COMPONENTS.get(item["hypothesis"], ("Multiple operating systems", item["recommended_check"]))
            weight = float(item.get("signal_strength", 0)) * max(1, int(event.get("observations", 1))) * max(float(event.get("max_score", 0)), 0.55)
            rows.append({"equipment_id": event["equipment_id"], "hypothesis": item["hypothesis"], "likely_area_to_inspect": component, "weighted_strength": weight, "evidence": item["evidence"], "recommended_check": check, "event_id": event["event_id"]})
    details = pd.DataFrame(rows)
    if details.empty:
        return details
    summary = details.groupby(["equipment_id", "hypothesis", "likely_area_to_inspect"], as_index=False).agg(events=("event_id", "nunique"), weighted_strength=("weighted_strength", "sum"), evidence=("evidence", "first"), recommended_check=("recommended_check", "first"))
    summary["weighted_strength"] = summary["weighted_strength"].round(2)
    return summary.sort_values(["equipment_id", "weighted_strength"], ascending=[True, False]).reset_index(drop=True)
