# Sepsis Vitals

Vitals-only sepsis prediction tooling for low-resource hospitals.

The project hypothesis is intentionally narrow: a lightweight gradient-boosted
model using only temperature, heart rate, respiratory rate, systolic blood
pressure, SpO2, and GCS can predict sepsis risk early enough to support nurse
escalation in LMIC district hospitals.

## Current Status

Phase 0 engineering has started with:

- A tested feature engineering package in `src/sepsis_vitals`.
- A site data-quality report for early partner extracts.
- qSOFA, partial SIRS, shock-index, missingness, delta, rolling, and pediatric
  feature support.
- A data contract for partner-site feasibility review.
- A Phase 1 model scaffold plan.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Run Tests

```bash
python3 -m pytest -q
```

## Run Example Without Installing

```bash
PYTHONPATH=src python3 examples/run_feature_pipeline.py
```

## View The Static UI

Open `docs/index.html` directly or serve the `docs/` folder:

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## GitHub Pages

This repo is configured to publish `docs/` through GitHub Actions. After the
repo is pushed to GitHub, enable Pages with source set to GitHub Actions, then
push to `main`. The workflow in `.github/workflows/pages.yml` deploys the
static dashboard.

## Minimal Use

```python
import pandas as pd
from sepsis_vitals import build_feature_set
from sepsis_vitals.data_quality import summarize_vitals_quality

df = pd.read_csv("vitals.csv", parse_dates=["timestamp"])
quality = summarize_vitals_quality(df)
features = build_feature_set(
    df,
    patient_col="patient_id",
    time_col="timestamp",
    age_col="age_years",
    rolling_window=3,
)
```

## Required Input Columns

The pipeline needs `patient_id`, `timestamp`, and at least 3 of the 6 core
vitals. Full model development should target all 6:

- `temperature`
- `heart_rate`
- `resp_rate`
- `sbp`
- `spo2`
- `gcs`

Optional but recommended:

- `age_years`
- `site_id`
- `encounter_id`
- outcome/label columns defined during IRB and adjudication design

## Design Principles

- Missingness is a signal, not just a nuisance.
- qSOFA and partial SIRS are comparators as well as features.
- Whole-episode aggregates are off by default because they can leak future data.
- The first production model should be boring: logistic regression baseline,
  then XGBoost/LightGBM with calibration and SHAP.
