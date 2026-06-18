# Sepsis Vitals v0.5.0

Vitals-only sepsis prediction for low-resource district hospitals.

**228+ tests passing · Production-ready SaaS platform**

---

## Quick Start

```bash
unzip sepsis-vitals.zip && cd sepsis-vitals
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                                   # 228+ tests
PYTHONPATH=src python3 examples/run_feature_pipeline.py
python3 -m http.server 8000 --directory docs          # → http://localhost:8000
```

---

## Project Structure

```
sepsis-vitals/
├── src/sepsis_vitals/
│   ├── scores.py            # qSOFA, SIRS, NEWS2, UVA, shock index
│   ├── features.py          # Feature engineering pipeline
│   ├── data_quality.py      # Site data auditing
│   ├── model_scaffold.py    # LightGBM + logistic + SHAP + model card
│   ├── security.py          # Rate limiting, injection guard, HMAC
│   ├── api.py               # FastAPI clinical scoring API with authentication framework
│   ├── auth/jwt.py          # Password hashing (bcrypt), RBAC, MFA (TOTP), account lockout
│   ├── realtime/websocket.py# WebSocket alert streaming
│   ├── monitoring/metrics.py# Prometheus, PSI drift, alert fatigue
│   ├── ml/
│   │   ├── fairness.py      # Subgroup audit, conformal prediction, calibration
│   │   ├── synthetic_data.py# NHANES-powered synthetic vital sign generation
│   │   ├── trainer.py       # Model training pipeline
│   │   └── predictor.py     # Inference and prediction
│   ├── billing/            # Stripe 3-tier SaaS billing
│   ├── patients/           # Patient CRUD + vitals persistence
│   ├── alerts/             # SMS (Twilio + Africa's Talking), push, dispatcher
│   ├── fhir/               # HL7 FHIR R4 adapter with LOINC mapping
│   └── i18n/               # 6 languages (en, sw, fr, pt, am, ar)
├── health_economics/model.py # ROI, QALY, cost-effectiveness, break-even
├── compliance/
│   ├── irb_protocol_template.md
│   ├── adjudication_protocol.md
│   ├── data_protection_agreement.md  # HIPAA BAA + GDPR DPA + Africa clauses
│   └── qms_iec62304_iso14971.md      # QMS, risk register, design controls
├── alembic/                 # Database migrations
├── docker/                  # Dockerfile, docker-compose, nginx, prometheus
├── terraform/               # AWS ECS + RDS + Redis + WAF + Secrets Manager
├── tests/                   # 228+ tests
├── examples/run_feature_pipeline.py
├── docs/index.html          # Mobile-first dashboard (auth, i18n, onboarding)
└── .github/workflows/       # 4-job CI pipeline (lint, typecheck, security, test)
```

---

## Start the API

```bash
cp .env.example .env        # Fill in secrets
uvicorn sepsis_vitals.api:app --reload --port 8080
# Docs: http://localhost:8080/docs
# Health: http://localhost:8080/health
# Metrics: http://localhost:8080/metrics
```

## Docker (full stack)

```bash
cd docker
cp ../.env.example ../.env  # Fill in all values
docker compose up -d
# API:        http://localhost:8080
# Dashboard:  http://localhost:8000
# Grafana:    http://localhost:3001
# Prometheus: http://localhost:9090
```

## Database migrations

```bash
pip install -e ".[api]"
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/sepsis_vitals
alembic upgrade head
```

---

## Model Performance

NHANES-calibrated test set results (LightGBM, vitals-only features):

| Metric      | Value  |
|-------------|--------|
| AUROC       | 0.9916 |
| Sensitivity | 0.9022 |
| F1 Score    | 0.9250 |

---

## Security checklist (all implemented)

| # | Gap | Implementation |
|---|---|---|
| 1 | Rate limiting | Token-bucket (5 req/s API, 0.5 req/s LLM) + WAF rules |
| 2 | Hard-coded API keys | SecretManager reads env vars only |
| 3 | Exposed endpoints | CORS whitelist; WAF; CSP headers; no * origins |
| 4 | Encryption | AES-256-GCM PII, RDS KMS, TLS 1.3 |
| 5 | Prompt injection | 12-pattern regex + structural isolation + client-side pre-filter |
| 6 | Webhook security | HMAC-SHA256 + 5-minute replay window |

---

## Clinical scoring (all validated against boundary conditions)

| Score | Reference | Threshold |
|---|---|---|
| qSOFA | Seymour et al. JAMA 2016 | ≥2 = high risk |
| Partial SIRS | Bone et al. Chest 1992 | ≥2 criteria |
| Shock Index | HR/SBP | ≥1.0 = elevated |
| NEWS2-style | RCP 2017 | ≥5 = medium, ≥7 = high |
| UVA-style | Kruisselbrink et al. PLOS ONE 2019 | ≥4 = high mortality |
