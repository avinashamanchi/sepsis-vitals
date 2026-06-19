# Sepsis Vitals v0.6.0

Vitals + lab-augmented sepsis prediction for district hospitals.

**258+ tests passing · Production-hardened clinical platform**

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
│   ├── scores.py            # qSOFA, SIRS, NEWS2, UVA, shock index + lactate integration
│   ├── features.py          # Feature engineering pipeline (vitals + labs)
│   ├── data_quality.py      # Site data auditing
│   ├── model_scaffold.py    # LightGBM + logistic + SHAP + model card
│   ├── security.py          # Rate limiting, injection guard, HMAC, security headers
│   ├── api.py               # FastAPI API with OAuth2/JWT auth, security headers
│   ├── auth/jwt.py          # JWT tokens, password hashing, RBAC, MFA, user store
│   ├── realtime/websocket.py# WebSocket alert streaming (authenticated)
│   ├── monitoring/metrics.py# Prometheus, PSI drift, alert fatigue
│   ├── ml/
│   │   ├── fairness.py      # Subgroup audit, conformal prediction, calibration
│   │   ├── synthetic_data.py# NHANES + MIMIC-III synthetic data with confounders
│   │   ├── trainer.py       # Model training pipeline (vitals + labs)
│   │   ├── predictor.py     # Inference with persistent state (SQLite)
│   │   └── state_store.py   # SQLite patient state persistence (replaces in-memory)
│   ├── fhir/
│   │   ├── listener.py      # HL7v2 MLLP + FHIR R4 webhook auto-ingestion
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

Trained on synthetic data with sick-but-not-septic confounders (post-surgical,
dehydration, pain, COPD/HF exacerbation, viral infection). Lab values
(lactate, WBC, procalcitonin) included. **AUROC is realistic — not inflated.**

> **Note:** Synthetic data provides a development baseline only. Production
> deployment requires retrospective validation on MIMIC-IV/eICU and a
> prospective clinical trial per FDA SaMD guidelines.

| Metric      | Value  | Note |
|-------------|--------|------|
| AUROC       | ~0.88  | Realistic with confounders |
| Sensitivity | ~0.58  | At default threshold (0.5) |
| Specificity | ~0.94  | Low false-alarm rate |
| Sens@90Spec | ~0.70  | Clinical operating point |

---

## Security checklist (all implemented)

| # | Gap | Implementation |
|---|---|---|
| 1 | Rate limiting | Token-bucket (10 req/s API, 2 req/s ML, 0.5 req/s copilot, 1 req/s billing, 5 req/s webhook) |
| 2 | Auth by default | `SEPSIS_AUTH_ENABLED=true`; JWT with ephemeral dev secret; production requires `SEPSIS_JWT_SECRET` |
| 3 | Exposed endpoints | CORS whitelist; `/docs` + `/openapi.json` disabled in production; `/metrics` requires auth |
| 4 | WebSocket auth | Token-based handshake; rejects unauthenticated connections |
| 5 | LLM isolation | Enterprise flag required; de-identified data only; per-user copilot rate limiting |
| 6 | Security headers | HSTS (2yr preload), CSP, nosniff, X-XSS-Protection, Permissions-Policy |
| 7 | Prompt injection | 21-pattern regex (incl. leet-speak, XML, role hijack) + structural isolation |
| 8 | Webhook security | HMAC-SHA256 + replay window + Stripe IP allowlist + event deduplication |
| 9 | Password security | PBKDF2-SHA256 (200K iterations) or bcrypt + exponential lockout |
| 10 | Patient state | SQLite WAL mode (persists across restarts, multi-worker safe) |
| 11 | Key management | No hardcoded secrets; ephemeral dev keys; production raises on missing secrets |
| 12 | Health endpoint | Minimal response in production (no model name, auth status, or connection count) |

---

## Clinical scoring (all validated against boundary conditions)

| Score | Reference | Threshold |
|---|---|---|
| qSOFA | Seymour et al. JAMA 2016 | ≥2 = high risk |
| Partial SIRS | Bone et al. Chest 1992 | ≥2 criteria |
| Shock Index | HR/SBP | ≥1.0 = elevated |
| NEWS2-style | RCP 2017 | ≥5 = medium, ≥7 = high |
| UVA-style | Kruisselbrink et al. PLOS ONE 2019 | ≥4 = high mortality |
