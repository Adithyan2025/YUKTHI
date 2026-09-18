from __future__ import annotations
import numpy as np
import pandas as pd
from .fault_hypotheses import rank_hypotheses


def _severity(score: float, persistence: int, quality_ok: bool) -> str:
    if score >= 0.82 or (score >= 0.68 and persistence >= 4):
        return "HIGH" if quality_ok else "ATTENTION"
    if score >= 0.55 or persistence >= 3:
        return "ATTENTION"
    return "NORMAL"


def build_events(scored: pd.DataFrame, quality: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = scored.copy().sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)
    data["event_id"] = pd.Series(pd.NA, index=data.index, dtype="string")
    event_rows = []
    event_number = 1
    for equipment, group in data.groupby("equipment_id", sort=True):
        anomaly_indices = group.index[group["is_anomaly"]].tolist()
        if not anomaly_indices:
            continue
        clusters = [[anomaly_indices[0]]]
        for index in anomaly_indices[1:]:
            previous = clusters[-1][-1]
            gap = (data.loc[index, "timestamp"] - data.loc[previous, "timestamp"]).total_seconds() / 60
            if gap <= 120:
                clusters[-1].append(index)
            else:
                clusters.append([index])
        for cluster in clusters:
            subset = data.loc[cluster]
            event_id = f"EVT-{event_number:04d}"
            data.loc[cluster, "event_id"] = event_id
            score = float(subset["anomaly_score"].max())
            persistence = len(subset)
            severity = _severity(score, persistence, quality.get("missing_pct", 0) < 10)
            start, end = subset["timestamp"].min(), subset["timestamp"].max()
            duration = max(0.0, (end - start).total_seconds() / 3600)
            baseline = data[(data["equipment_id"] == equipment) & (~data["is_anomaly"])].copy()
            load = float(subset["building_load_rt"].median())
            temp = float(subset["cooling_water_temp_c"].median())
            comparable = baseline[(baseline["building_load_rt"].between(load * 0.8, load * 1.2, inclusive="both")) & (baseline["cooling_water_temp_c"].between(temp - 3, temp + 3, inclusive="both"))]
            if comparable.empty:
                comparable = baseline
            typical = float(comparable["energy_kwh"].median()) if not comparable.empty else float(subset["energy_kwh"].median())
            observed = float(subset["energy_kwh"].median())
            difference_pct = ((observed - typical) / abs(typical) * 100) if typical else 0.0
            context = f"Energy median {observed:.2f} kWh versus {typical:.2f} kWh for comparable historical observations ({difference_pct:+.1f}%)."
            explanation = "Energy behaviour deviates from this equipment's historical pattern under comparable load and cooling-water conditions."
            if persistence >= 3:
                explanation += f" The signal persists across {persistence} observations."
            recommendation = "Review the full flagged period and operating history before concluding that equipment degradation is present."
            if abs(difference_pct) >= 15:
                recommendation = "Investigate the energy-to-load relationship and review operating conditions during the flagged period."
            if quality.get("missing_pct", 0) >= 10:
                recommendation += " Review sensor/data quality before interpreting the result as equipment behaviour."
            hypotheses = rank_hypotheses(subset, baseline, quality.get("missing_pct", 0))
            event_rows.append({"event_id": event_id, "equipment_id": equipment, "start": start, "end": end, "duration_hours": duration, "observations": persistence, "max_score": score, "average_score": float(subset["anomaly_score"].mean()), "severity": severity, "observed_energy": observed, "historical_energy": typical, "difference_pct": difference_pct, "load_median": load, "cooling_temp_median": temp, "context": context, "explanation": explanation, "recommendation": recommendation, "fault_hypotheses": hypotheses})
            event_number += 1
    events = pd.DataFrame(event_rows)
    return data, events
