# YUKTI Chiller Intelligence

A Streamlit application for contextual, equipment-specific anomaly detection in historical chiller data. It accepts the challenge CSV dynamically and does not depend on row counts, dates, or hard-coded observations.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV from the sidebar. Required concepts are timestamp, equipment ID, chilled-water flow, cooling-water temperature, building load, energy, outside temperature, dew point, humidity, wind speed, and pressure. Minor column-name variations are mapped automatically; the Data Quality page shows the mapping.

## Method

1. Parse, normalize, sort, and validate the uploaded data.
2. Interpolate only small isolated numeric gaps (up to two observations per equipment); longer gaps remain missing and are reported.
3. Build time, lag, observation-window rolling, change, ratio, and actual-gap features. Lag names are only treated as time windows when observed timestamp gaps support that interpretation.
4. Fit an Isolation Forest to contextual variables and derived relationships. The pipeline uses equipment identity through equipment-specific event baselines and does not compare raw equipment values as if units were identical.
5. Normalize the inverted model decision function into an application-defined Model Anomaly Score from 0 to 1. This is not an industry risk score.
6. Group near-consecutive anomalous observations using actual timestamps. Event duration is calculated from event timestamps, not assumed sampling intervals.
7. Severity is application-defined: HIGH for very high scores or sustained elevated scores, ATTENTION for elevated scores or persistence, otherwise NORMAL. Poor data quality can downgrade HIGH to ATTENTION.
8. Explanations compare energy with historical observations for the same equipment under approximate load and cooling-water context. Recommendations are investigation prompts, not diagnoses.

There are no labelled anomaly targets, so the app does not claim accuracy, precision, recall, or F1. Diagnostics are anomaly counts, persistence, score distributions, and contextual historical comparisons. The current baseline is deliberately interpretable and replaceable.

## Project layout

- `app.py`: Streamlit user experience and charts
- `src/data_loader.py`: ingestion, column mapping, validation
- `src/feature_engineering.py`: time-series-aware features and limited imputation
- `src/anomaly_detection.py`: Isolation Forest scoring
- `src/events.py`: persistence, historical comparison, explanations, recommendations
- `src/pipeline.py`: reusable orchestration

## Data handling notes

Missing timestamps are not converted into equipment faults. Duplicate equipment/timestamp pairs, invalid numeric values, missing cells, and timestamp gaps are surfaced in Data Quality. CSV processing is in-memory and cached per upload for hackathon-scale datasets.
