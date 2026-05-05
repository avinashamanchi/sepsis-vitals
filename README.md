# Sepsis Vitals

Vitals-only sepsis prediction tooling for low-resource hospitals.

The project hypothesis is intentionally narrow: a lightweight gradient-boosted
model using only temperature, heart rate, respiratory rate, systolic blood
pressure, SpO2, and GCS can predict sepsis risk early enough to support nurse
escalation in LMIC district hospitals.

## Current Status — Phase 0 ✓

- ✅ Feature engineering package (`src/sepsis_vitals`)
- ✅ Clinical scoring: qSOFA, partial SIRS, shock index, NEWS2-style, UVA-style
- ✅ Data quality & contract validation
- ✅ Phase 1 model scaffold (LightGBM + logistic baseline + SHAP + model card)
- ✅ FastAPI server with full security hardening
- ✅ 139 passing tests
- ✅ Static clinical dashboard (GitHub Pages)
- ✅ CI/CD via GitHub Actions

---

## Quick Start (VS Code)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# 2. Install all dependencies
pip install -e ".[dev]"

# 3. Run tests
python -m pytest -q

# 4. Run the feature pipeline demo
PYTHONPATH=src python3 examples/run_feature_pipeline.py

# 5. View the dashboard (open in browser)
python3 -m http.server 8000 --directory docs
# → http://localhost:8000

# 6. Start the API server (optional — needs ANTHROPIC_API_KEY)
cp .env.example .env               # fill in your keys
uvicorn sepsis_vitals.api:app --reload --port 8080
# → http://localhost:8080/docs
```

---

## Project Structure

```
sepsis-vitals/
├── src/sepsis_vitals/
│   ├── __init__.py          # public API surface
│   ├── scores.py            # qSOFA, SIRS, NEWS2, UVA, shock index
│   ├── features.py          # feature engineering pipeline
│   ├── data_quality.py      # partner data auditing
│   ├── model_scaffold.py    # Phase 1 model training & evaluation
│   ├── api.py               # FastAPI server
│   └── security.py          # rate limiting, injection guard, HMAC, crypto
├── tests/
│   ├── test_scores.py       # 40 tests
│   ├── test_features.py     # 66 tests
│   └── test_security.py     # 33 tests
├── examples/
│   └── run_feature_pipeline.py
├── docs/
│   └── index.html           # clinical dashboard (GitHub Pages)
├── .github/workflows/
│   ├── ci.yml               # test + lint on every push
│   └── pages.yml            # auto-deploy dashboard
├── .env.example             # secrets template — copy to .env
├── pyproject.toml
└── README.md
```

---

## Minimal Use

```python
import pandas as pd
from sepsis_vitals import build_feature_set
from sepsis_vitals.data_quality import summarize_vitals_quality
from sepsis_vitals.scores import compute_scores

df = pd.read_csv("vitals.csv", parse_dates=["timestamp"])
quality  = summarize_vitals_quality(df)
features = build_feature_set(
    df,
    patient_col="patient_id",
    time_col="timestamp",
    age_col="age_years",
    rolling_window=3,
)

# Score a single observation
bundle = compute_scores({
    "temperature": 38.9, "heart_rate": 118,
    "resp_rate": 26,     "sbp": 92,
    "spo2": 91,          "gcs": 13,
})
print(bundle.risk_level, bundle.alert_flag)  # → critical True
```

---

## Required Input Columns

`patient_id`, `timestamp`, and at least 3 of:

| Column        | Unit        |
|---------------|-------------|
| `temperature` | °C          |
| `heart_rate`  | bpm         |
| `resp_rate`   | breaths/min |
| `sbp`         | mmHg        |
| `spo2`        | %           |
| `gcs`         | 3–15        |

Optional: `age_years`, `site_id`, `encounter_id`

---

## Security Features

| Layer | Implementation |
|---|---|
| Rate limiting | Token-bucket per IP (5 req/s API, 0.5 req/s LLM) |
| API key safety | `SecretManager` — reads env vars only, never hard-coded |
| Input validation | Vital range checks + control character stripping |
| Prompt injection | 12-pattern regex guard + structural prompt isolation |
| Webhook security | HMAC-SHA256 + replay protection (5 min window) |
| Encrypted config | Fernet at-rest encryption for sensitive values |
| HTTP headers | CSP, HSTS, X-Frame-Options, X-Content-Type-Options |
| CORS | Explicit origin whitelist — never wildcard `*` |

---

## Design Principles

- Missingness is a signal, not just a nuisance.
- qSOFA and partial SIRS are comparators as well as features.
- Whole-episode aggregates are off by default (future data leakage).
- The first production model should be boring: logistic regression baseline,
  then XGBoost/LightGBM with calibration and SHAP.
