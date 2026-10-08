# Sepsis Vitals

Investigational sepsis early-warning software for retrospective research and
prospective silent-mode evaluation.

> **Not for patient care.** The current model was trained on synthetic data and
> has not completed external clinical validation or regulatory review. Its
> outputs must not be used for diagnosis, treatment, triage, or autonomous
> alerting.

## What this repository contains

- A FastAPI research API for clinical scores, model evaluation, FHIR/HL7
  ingestion, audit trails, and simulated monitoring.
- A React research workspace for ward-review workflow testing.
- A synthetic-data development model with explicit provenance metadata.
- Validation scaffolding for retrospective cohorts, subgroup analysis,
  calibration, drift monitoring, and silent-mode studies.
- Draft study, adjudication, risk-management, and data-governance documents.

The valuable part of this project is the research and workflow infrastructure.
The synthetic model is a development fixture—not the commercial moat and not
evidence of clinical benefit.

## Local setup

Python 3.10–3.12 and Node.js 20+ are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,ml,dev]"
cp .env.example .env
pytest -q
```

Tests run against a throwaway SQLite database (`tests/conftest.py`); they never
touch your local `sepsis_vitals.db`.

Create or upgrade the database schema, then start the API. Alembic owns the
schema; run migrations with the same `SEPSIS_PII_KEY` the API uses, because
migration 003 backfills blind indexes from encrypted values.

```bash
alembic upgrade head
uvicorn sepsis_vitals.api:app --reload --port 8080
```

Start the frontend in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

The frontend calls the API under `/api`. The Vite dev server (and the nginx
container) strips that prefix and forwards to port 8080; WebSocket traffic on
`/ws` is proxied the same way. To exercise the static synthetic demo locally,
set `VITE_DEMO_MODE=true` in `frontend/.env`.

Password-reset emails need `SMTP_HOST` (plus credentials) and `SEPSIS_APP_URL`,
the public URL of the web app; see [.env.example](.env.example). Without SMTP
the request succeeds but no email is sent.

### Docker Compose

`docker/docker-compose.yml` starts Postgres, Redis, the API, the dashboard
(nginx on port 8000) and monitoring. The API container runs
`alembic upgrade head` before starting (`SEPSIS_RUN_MIGRATIONS`); infrastructure
ports are bound to `127.0.0.1` only. Model artifacts are not baked into the
image (`.dockerignore` excludes `models/*.joblib`), so `/predict` returns 503
until a model directory is provided.

## Validation status

| Gate | Status | Meaning |
|---|---|---|
| Engineering baseline | Complete | Reproducible artifacts, model metadata, tests, audit scaffolding |
| Synthetic development testing | Complete | Useful only for software development |
| Retrospective external validation | Not complete | No unseen hospital cohort has established transportability |
| Prospective silent-mode study | Not complete | Alert burden, lead time, and workflow effects are unknown |
| Regulatory review | Not started | No FDA clearance, CE marking, or equivalent authorization |
| Clinical deployment | Not permitted | Research and evaluation use only |

The metrics under `models/` describe synthetic test data. They must never be
presented as clinical performance.

### What the synthetic evidence can and cannot show

[`reports/synthetic_pipeline_audit.md`](reports/synthetic_pipeline_audit.md)
quantifies the limits of the development pipeline:

- **The generator ties the label to age.** Demographics alone reach AUROC ≈
  0.72. Septic rows are not more physiologically abnormal on average than
  non-septic rows.
- **The headline row-level AUROC (≈ 0.90) measures recognising rows after
  labelled onset.** Discrimination of *future* sepsis from pre-onset rows is
  ≈ 0.72.
- **Scoring without history costs accuracy.** A single observation has no
  trends or observation gap, which lowers AUROC to ≈ 0.82 and roughly halves
  mean predicted risk. Since 2026-10 the API builds features with the training
  pipeline and passes recorded history for registered patients
  (`tests/test_inference_parity.py`). Unregistered patient IDs are still scored
  as first observations.

Reproduce with `python scripts/audit_synthetic_pipeline.py`. The proposed
intended use, numeric launch gates and analysis plan are in
[`compliance/intended_use_and_validation_plan.md`](compliance/intended_use_and_validation_plan.md).

### No-labs ablation

The matched development experiment in
[`reports/no_labs_ablation.md`](reports/no_labs_ablation.md) retrains the same
candidate models with every raw, derived, and missingness-based lab feature
removed. On the identical 3,000-patient synthetic held-out split, AUROC was
0.8584 without labs versus 0.9158 with the full feature set (change −0.0574).
This is evidence about the behavior of the synthetic development pipeline—not
evidence of performance in a hospital or patient population. Much of the
no-labs arm's discrimination comes from demographics that the generator links
to the label, so the ablation says little about vital signs alone (see the
audit above).

Reproduce it with:

```bash
python scripts/run_no_labs_ablation.py --patients 20000 --cv-folds 5
```

## Security posture

The code includes authentication, authorization, audit, encryption, rate-limit,
and tenant-isolation controls. Those controls are not the same as a completed
security program or compliance certification.

For production-like research deployments:

1. Disable self-registration and provision users through an approved process.
2. Use managed Postgres and Redis with encryption in transit and at rest.
3. Set all production secrets through a secret manager.
4. Complete threat modeling, penetration testing, access review, backup/restore
   testing, incident response exercises, and vendor assessments.
5. Do not process PHI without the required legal agreements and an approved
   deployment architecture.

Billing, treatment bundles, the clinical copilot, public registration, and
external LLM processing are disabled by default. They are not part of the
investigational release's intended workflow. See [.env.example](.env.example)
for the explicit feature gates; enabling a gate does not establish clinical,
security, privacy, or regulatory readiness.

## Model artifacts

The default artifact is a gradient-boosting development model trained on
synthetic adult trajectories. Its metadata records the training source,
limitations, and research-only regulatory status. Re-training and evaluation
entry points are:

```bash
python -m sepsis_vitals.train --data-source synthetic
python generate_validation_report.py
```

The scikit-learn runtime is pinned to the version used to serialize the checked-in
joblib artifacts. Retrain and version the artifacts before changing that runtime.

Real-world validation requires an appropriately governed, independent dataset
representative of the intended patient population. Training and testing at the
same institution is not sufficient evidence of generalizability.

## Repository map

```text
src/sepsis_vitals/     API, auth, ingestion, scoring, monitoring, and ML
frontend/              React research workspace and public evidence site
tests/                 Python unit and integration tests
compliance/            Draft study and quality-system documents
models/                Development artifacts and provenance
docker/                Local deployment stack
terraform/             Infrastructure scaffold; not a certified environment
```

See [STARTUP_REVIEW.md](STARTUP_REVIEW.md) for the product teardown, the
keep/cut decisions, and the launch gates, and
[PROJECT_REVIEW.md](PROJECT_REVIEW.md) for the 2026-10 independent review:
security, clinical-logic and ML-validity findings, the fixes made, and the
prioritized plan.

### Tenant scoping

Every patient, FHIR, alert, monitor and WebSocket path is scoped to the
caller's site. Only `system_admin` is unscoped. A user without a site
assignment sees no patient data until an administrator assigns one with
`PUT /auth/users/{id}/site`.
