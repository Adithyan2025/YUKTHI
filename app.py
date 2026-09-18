from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import hashlib

from src.data_loader import load_csv
from src.pipeline import analyze

st.set_page_config(page_title="YUKTI | Chiller Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#17242b; --muted:#6c7b80; --line:#dbe4e5; --mint:#0f766e; --orange:#c7672d; --red:#b43c3c; }
html, body, [class*="css"] { font-family: "Trebuchet MS", "Segoe UI", sans-serif; color: var(--ink); }
[data-testid="stSidebar"] { background: #f1f5f3; border-right: 1px solid var(--line); }
[data-testid="stMetricValue"] { font-family: Georgia, serif; letter-spacing: 0; }
.block-container { padding-top: 2rem; max-width: 1500px; }
.hero { padding: 1.3rem 1.5rem; border: 1px solid var(--line); border-left: 5px solid var(--mint); background: linear-gradient(110deg,#f8fbf8,#edf6f3); margin-bottom: 1.2rem; }
.hero h1 { font-family: Georgia, serif; font-weight: 500; margin: 0; font-size: 2.3rem; }
.hero p { color: var(--muted); margin: .35rem 0 0; }
.status { display:inline-block; padding:.22rem .55rem; border-radius: 3px; font-size:.72rem; font-weight:700; letter-spacing:.06em; }
.normal { color:#12634d; background:#dff3e9; } .attention { color:#95501c; background:#fff0d7; } .high { color:#9e2e2e; background:#fde1df; }
.section-title { font-family: Georgia, serif; font-size: 1.35rem; margin: 1.2rem 0 .5rem; }
.small-note { color: var(--muted); font-size: .85rem; }
</style>
""", unsafe_allow_html=True)


def demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2026-01-01", periods=960, freq="30min")
    frames = []
    for number, equipment in enumerate(["CHILLER-01", "CHILLER-02", "CHILLER-03"]):
        load = 340 + 100 * np.sin(np.arange(len(timestamps)) / 38 + number) + rng.normal(0, 18, len(timestamps))
        outside = 72 + 12 * np.sin(np.arange(len(timestamps)) / 120) + rng.normal(0, 2, len(timestamps))
        energy = 85 + load * .15 + number * 4 + rng.normal(0, 5, len(timestamps))
        if number == 1:
            energy[540:550] += 42
        frames.append(pd.DataFrame({"timestamp": timestamps, "equipment_id": equipment, "flow_lps": 45 + load / 20 + rng.normal(0, 2, len(timestamps)), "cooling_water_temp_c": 27 + (outside - 72) * .08 + rng.normal(0, .7, len(timestamps)), "building_load_rt": load.clip(80), "energy_kwh": energy.clip(1), "outside_temp_f": outside, "dew_point_f": outside - 12, "humidity_pct": 55 + rng.normal(0, 5, len(timestamps)), "wind_speed_mph": rng.uniform(1, 16, len(timestamps)), "pressure_in": 29.9 + rng.normal(0, .1, len(timestamps))}))
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Training the contextual anomaly model on your CSV...")
def run_analysis(frame: pd.DataFrame) -> dict:
    return analyze(frame)


def chart_line(data, y, title, color="equipment_id", markers=False):
    fig = px.line(data, x="timestamp", y=y, color=color, markers=markers, title=title, template="plotly_white")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    return fig


def score_class(value: str) -> str:
    return value.lower()


def event_table(events: pd.DataFrame):
    if events.empty:
        st.info("No anomaly events were detected for the selected period.")
        return
    shown = events[["event_id", "equipment_id", "start", "end", "duration_hours", "max_score", "severity", "observations"]].copy()
    shown.columns = ["Event ID", "Equipment", "Start", "End", "Duration (h)", "Max Score", "Severity", "Observations"]
    shown["Start"] = shown["Start"].dt.strftime("%Y-%m-%d %H:%M")
    shown["End"] = shown["End"].dt.strftime("%Y-%m-%d %H:%M")
    shown["Max Score"] = shown["Max Score"].round(2)
    st.dataframe(shown, use_container_width=True, hide_index=True)


def plot_event(data: pd.DataFrame, event: pd.Series):
    subset = data[(data["equipment_id"] == event.equipment_id) & (data["timestamp"].between(event.start - pd.Timedelta(hours=12), event.end + pd.Timedelta(hours=12)))].copy()
    subset["flag"] = np.where(subset["event_id"].eq(event.event_id), "Flagged event", "Context")
    fig = go.Figure()
    for flag, color in [("Context", "#9aa8a8"), ("Flagged event", "#b43c3c")]:
        part = subset[subset["flag"] == flag]
        fig.add_trace(go.Scatter(x=part.timestamp, y=part.energy_kwh, mode="lines+markers", name=flag, line=dict(color=color)))
    fig.update_layout(title="Energy behaviour around event", yaxis_title="Energy (kWh)", template="plotly_white", height=330, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart_line(subset, "anomaly_score", "Model anomaly score"), use_container_width=True)
    with right:
        st.plotly_chart(chart_line(subset, "building_load_rt", "Building load context"), use_container_width=True)


with st.sidebar:
    st.markdown("## YUKTI / 2026")
    st.caption("Contextual chiller intelligence")
    uploaded = st.file_uploader("Upload challenge CSV", type=["csv"], help="The app maps minor column-name variations automatically.")
    use_demo = st.button("Load demo dataset", use_container_width=True)
    if use_demo:
        st.session_state["data_source"] = "demo"
        st.session_state["dataset_key"] = "demo-v1"
    if uploaded is not None:
        st.session_state["data_source"] = "upload"
        st.session_state["uploaded_bytes"] = uploaded.getvalue()
        st.session_state["uploaded_name"] = uploaded.name
        st.session_state["dataset_key"] = hashlib.sha256(st.session_state["uploaded_bytes"]).hexdigest()
    if "data_source" not in st.session_state:
        st.markdown("### Start here")
        st.info("Upload the official CSV to begin. Demo mode is clearly labelled and uses synthetic data.")
        st.stop()
    if st.session_state["data_source"] == "demo":
        frame = demo_data()
        st.warning("DEMO DATA - NOT OFFICIAL CHALLENGE DATA")
    else:
        from io import BytesIO
        frame, ingest = load_csv(BytesIO(st.session_state["uploaded_bytes"]))
        st.caption(f"Loaded: {st.session_state.get('uploaded_name', 'uploaded.csv')}")
    st.markdown("### Model training")
    st.caption("The model is trained from the currently selected dataset. No precomputed anomaly results are used.")
    train_clicked = st.button("Train model on this CSV", type="primary", use_container_width=True)
    if train_clicked:
        st.session_state["trained_dataset_key"] = st.session_state["dataset_key"]
        st.session_state["trained_name"] = st.session_state.get("uploaded_name", "demo.csv")
    if st.session_state.get("trained_dataset_key") != st.session_state["dataset_key"]:
        st.info("Upload a CSV and click 'Train model on this CSV' to begin.")
        st.stop()
    try:
        result = run_analysis(frame)
    except Exception as error:
        st.error(f"Analysis could not run: {error}")
        st.stop()
    pages = ["Dashboard", "Equipment monitoring", "Anomaly explorer", "Investigation", "Data explorer", "Data quality", "Methodology", "Impact & outcomes"]
    page = st.radio("Navigate", pages, label_visibility="collapsed")
    st.divider()
    st.caption(f"Trained: {st.session_state.get('trained_name', 'dataset')} | {len(result['raw']):,} observations | {result['quality']['equipment_count']} equipment")

quality = result["quality"]
scored = result["scored"]
events = result["events"]

st.markdown(f'<div class="hero"><h1>Chiller intelligence</h1><p>Model trained on <strong>{st.session_state.get("trained_name", "the selected CSV")}</strong>. Context-aware monitoring for unusual equipment behaviour. Scores support investigation; they are not physical diagnoses.</p></div>', unsafe_allow_html=True)

if page == "Dashboard":
    high_count = int((events["severity"] == "HIGH").sum()) if not events.empty else 0
    cols = st.columns(4)
    cols[0].metric("Total equipment", quality["equipment_count"])
    cols[1].metric("Observations", f"{quality['rows']:,}")
    cols[2].metric("Anomaly events", len(events))
    cols[3].metric("High-priority events", high_count)
    st.markdown('<div class="section-title">Equipment status</div>', unsafe_allow_html=True)
    rows = []
    for equipment in quality["equipment_ids"]:
        equipment_events = events[events["equipment_id"] == equipment] if not events.empty else events
        latest = equipment_events.sort_values("end").iloc[-1] if not equipment_events.empty else None
        rows.append({"Equipment": equipment, "Status": latest.severity if latest is not None else "NORMAL", "Events": len(equipment_events), "Latest anomaly": latest.end.strftime("%Y-%m-%d %H:%M") if latest is not None else "-", "Max score": round(float(equipment_events.max_score.max()), 2) if not equipment_events.empty else 0.0, "Persistence": int(equipment_events.observations.sum()) if not equipment_events.empty else 0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Recent anomaly events</div>', unsafe_allow_html=True)
    event_table(events.sort_values("start", ascending=False).head(8))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart_line(scored, "energy_kwh", "Energy consumption by equipment"), use_container_width=True)
    with right:
        st.plotly_chart(chart_line(scored, "anomaly_score", "Model anomaly score"), use_container_width=True)

elif page == "Equipment monitoring":
    equipment_options = ["All"] + quality["equipment_ids"]
    selected = st.selectbox("Equipment", equipment_options)
    view = scored if selected == "All" else scored[scored["equipment_id"] == selected]
    if view.empty:
        st.info("No observations match this equipment selection.")
    else:
        latest = view.sort_values("timestamp").iloc[-1]
        status = "HIGH" if latest.anomaly_score >= .82 else "ATTENTION" if latest.anomaly_score >= .55 else "NORMAL"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest energy", f"{latest.energy_kwh:.2f} kWh")
        c2.metric("Latest load", f"{latest.building_load_rt:.2f} RT")
        c3.metric("Latest score", f"{latest.anomaly_score:.2f}")
        c4.metric("Current status", status)
        st.plotly_chart(chart_line(view, "energy_kwh", "Energy trend"), use_container_width=True)
        st.plotly_chart(chart_line(view, "building_load_rt", "Building load trend"), use_container_width=True)
        st.plotly_chart(chart_line(view, "anomaly_score", "Anomaly score trend", markers=True), use_container_width=True)
        st.plotly_chart(chart_line(view, "cooling_water_temp_c", "Cooling-water temperature"), use_container_width=True)

elif page == "Anomaly explorer":
    st.markdown('<div class="section-title">Anomaly event explorer</div>', unsafe_allow_html=True)
    if events.empty:
        event_table(events)
    else:
        a, b, c = st.columns(3)
        equipment = a.selectbox("Equipment", ["All"] + quality["equipment_ids"])
        severity = b.multiselect("Severity", ["ATTENTION", "HIGH"], default=["ATTENTION", "HIGH"])
        minimum = c.slider("Minimum score", 0.0, 1.0, 0.0, .01)
        filtered = events.copy()
        if equipment != "All": filtered = filtered[filtered.equipment_id == equipment]
        if severity: filtered = filtered[filtered.severity.isin(severity)]
        filtered = filtered[filtered.max_score >= minimum]
        event_table(filtered)
        if not filtered.empty:
            chosen = st.selectbox("Open event", filtered.event_id.tolist())
            st.session_state["selected_event"] = chosen
            st.caption("Use the Investigation page for the evidence view.")

elif page == "Investigation":
    if events.empty:
        st.info("No anomaly events are available to investigate.")
    else:
        event_ids = events.sort_values("start", ascending=False).event_id.tolist()
        default = st.session_state.get("selected_event", event_ids[0])
        chosen = st.selectbox("Select event", event_ids, index=event_ids.index(default) if default in event_ids else 0)
        event = events[events.event_id == chosen].iloc[0]
        st.markdown(f"## Potential abnormal behaviour <span class='status {score_class(event.severity)}'>{event.severity}</span>", unsafe_allow_html=True)
        st.caption(f"{event.equipment_id} | {event.start:%Y-%m-%d %H:%M} -> {event.end:%Y-%m-%d %H:%M}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model anomaly score", f"{event.max_score:.2f}")
        c2.metric("Duration", f"{event.duration_hours:.2f} h")
        c3.metric("Observations", event.observations)
        c4.metric("Energy difference", f"{event.difference_pct:+.1f}%")
        st.markdown('<div class="section-title">What happened?</div>', unsafe_allow_html=True)
        st.write(event.explanation)
        st.markdown('<div class="section-title">Supporting evidence</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            st.write(event.context)
            st.write(f"Median building load: **{event.load_median:.2f} RT**")
        with e2:
            st.write(f"Median cooling-water temperature: **{event.cooling_temp_median:.2f} C**")
            st.write(f"Historical comparable energy: **{event.historical_energy:.2f} kWh**")
        plot_event(scored, event)
        st.markdown('<div class="section-title">Recommended investigation</div>', unsafe_allow_html=True)
        st.info(event.recommendation)

elif page == "Data explorer":
    st.markdown('<div class="section-title">Raw and analyzed observations</div>', unsafe_allow_html=True)
    view = scored.copy()
    selected_equipment = st.selectbox("Equipment", ["All"] + quality["equipment_ids"])
    if selected_equipment != "All": view = view[view.equipment_id == selected_equipment]
    all_columns = view.columns.tolist()
    columns = st.multiselect("Columns", all_columns, default=[c for c in ["timestamp", "equipment_id", "energy_kwh", "building_load_rt", "anomaly_score", "is_anomaly", "event_id"] if c in all_columns])
    st.dataframe(view[columns] if columns else view, use_container_width=True, hide_index=True, height=500)
    st.download_button("Download analyzed CSV", view.to_csv(index=False).encode("utf-8"), "yukti_analyzed_results.csv", "text/csv")

elif page == "Data quality":
    st.markdown('<div class="section-title">Data quality and preprocessing</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnostic quality score", f"{quality['data_quality_score']:.0f}/100")
    c2.metric("Missing cells", f"{quality['missing_cells']:,}")
    c3.metric("Timestamp gaps > 90 min", quality["gap_count_over_90m"])
    st.caption("Application-derived diagnostic; not an official challenge metric.")
    st.write({"Rows": quality["rows"], "Equipment": quality["equipment_ids"], "Date range": f"{quality['start']} to {quality['end']}", "Median gap (minutes)": quality["gap_median_minutes"], "Largest gap (minutes)": quality["gap_max_minutes"], "Duplicate records": quality["duplicates"], "Invalid timestamps": quality["invalid_timestamps"]})
    st.markdown("#### Missing values by column")
    st.dataframe(pd.DataFrame({"Missing values": quality["missing"]}), use_container_width=True)
    st.markdown("#### Numerical ranges")
    st.dataframe(pd.DataFrame(quality["numeric_ranges"], index=["Minimum", "Maximum"]).T, use_container_width=True)
    st.markdown("#### Preprocessing report")
    st.json(result["preprocessing"])
    st.info("Small isolated numeric gaps are interpolated within equipment. Longer gaps remain missing. Timestamp gaps are data-quality issues, not automatic equipment faults.")

elif page == "Methodology":
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.write("The application learns unusual multivariate behaviour rather than applying fixed energy thresholds.")
    steps = [
        ("1. Ingestion", "Column aliases are mapped, timestamps are parsed, equipment is identified, and rows are sorted."),
        ("2. Feature engineering", "Time-of-day, lags, observation-window rolling statistics, changes, ratios, and actual timestamp gaps are derived."),
        ("3. Contextual model", "Isolation Forest sees energy together with load, flow, temperatures, weather, and relationship features."),
        ("4. Chronology", "The unsupervised baseline is fitted without a random train/test split; no labelled performance claims are invented."),
        ("5. Scoring", "The inverted model decision function is normalized to the application-defined 0-1 Model Anomaly Score."),
        ("6. Persistence", "Near-consecutive flagged observations are grouped using actual time differences into anomaly events."),
        ("7. Explanation", "Events compare the same equipment with approximate historical load and cooling-water conditions."),
        ("8. Recommendation", "Outputs suggest evidence-linked investigation steps and avoid unsupported component diagnoses."),
    ]
    for title, text in steps:
        st.markdown(f"**{title}**  \\n{text}")
    st.markdown("#### Current model run")
    st.json(result["model"])
    st.markdown("#### Limitations")
    st.write("Without labelled failures, the model cannot establish fault truth. Sensor quality, unusual but legitimate operating modes, and sparse equipment histories can affect results. Treat HIGH and ATTENTION as review priorities, not maintenance conclusions.")

else:
    st.markdown('<div class="section-title">Expected operational impact</div>', unsafe_allow_html=True)
    st.write("The application turns historical chiller measurements into prioritized, evidence-backed investigation signals. It supports decisions without claiming that a specific component has failed.")
    impact_items = [
        ("Earlier identification", "Persistent unusual behaviour is grouped into events so operators can see emerging periods that warrant review."),
        ("Energy performance visibility", "Energy is interpreted alongside building load, flow, temperature, weather, and historical behaviour."),
        ("Issue prioritization", "Application-defined NORMAL, ATTENTION, and HIGH severities help teams focus on the strongest signals first."),
        ("Health and degradation monitoring", "Equipment is compared primarily with its own historical behaviour under comparable operating conditions."),
        ("Reduced unnecessary consumption", "Energy-to-load deviations can prompt investigation of operating conditions before excess use persists."),
        ("Evidence-backed operations", "Every event includes score, persistence, context, historical comparison, charts, and a recommended investigation."),
    ]
    for title, description in impact_items:
        st.markdown(f"### {title}")
        st.write(description)
    st.info("These are potential applications of the analytical signals. Actual value depends on data quality, sensor reliability, operational context, and human investigation.")
