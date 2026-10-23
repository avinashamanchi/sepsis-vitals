# Current state (Phase 0)

Snapshot of commit `28e1e0a` (branch `review/independent-review-fixes`, PR #4),
taken before any Phase 1 change. Read only: nothing here was changed to
produce it.

**This software is not approved for clinical use.** It is a research
prototype. Its model was trained only on synthetic data and has had no
clinical, external or prospective validation. `clinical_use` is
`not-permitted` in every API response. Nothing below is a claim of clinical
performance or safety.

## 1. Commands

| Task | Command | Notes |
|---|---|---|
| Install (Python) | `pip install --require-hashes -r requirements/dev.txt && pip install --no-deps -e .` | Hashed lock. `requirements/deploy.txt` is the runtime lock. |
| Train | `python retrain.py --output <dir> [--replace]` | Defaults: synthetic source, 50,000 patients, seed 42, legacy generator, `current_state` label. `--source mimic` needs authorized MIMIC-IV access. The committed model is not replaced without `--replace`, which archives the old one first. |
| Evaluate (ablation) | `python scripts/run_no_labs_ablation.py` | Synthetic. Default 20,000 patients, seed 42, 1,000 bootstrap replicates. |
| Evaluate (profiles) | `python scripts/evaluate_synthetic_profiles.py --horizon-hours <h>` | Synthetic. The horizon is a required argument because it is a clinical choice; reports used 12 h as an engineering example. |
| Evaluate (pipeline audit) | `python scripts/audit_synthetic_pipeline.py` | Train/serve parity on synthetic data. |
| Inference (library) | `SepsisPredictor(model_dir="models").predict(vitals, patient_id=..., history=[...])` | |
| API | `uvicorn sepsis_vitals.api:app` | Dev mode uses SQLite and needs no secrets. Production mode (`SEPSIS_ENV=production`) refuses to start without its secrets and database. |
| Full stack | `docker compose -f docker/docker-compose.yml up` | The container entrypoint runs `alembic upgrade head` unless `SEPSIS_RUN_MIGRATIONS=false`. |
| Frontend | `cd frontend && npm ci && npm run dev` | Node is blocked on the review workstation; the frontend is verified in CI only. |
| Migrations | `python -m alembic upgrade head` | PostgreSQL only. SQLite dev databases are created by `init_db`. Head at 28e1e0a was `e5f6a7b8c9d0`; Phase 1 adds `f6a7b8c9d0e1`. |
| Tests | `pytest`; `cd frontend && npm test` | CI runs 25 jobs, including PostgreSQL, Docker, compose E2E and browser E2E. |

## 2. Data flow

```
input ─► validation ─► feature pipeline ─► model ─► bands + rule floor ─► alert ─► response / UI ─► audit
```

| Step | Where | What happens |
|---|---|---|
| Input | `POST /predict` (one observation), `POST /predict/batch` (≤10), FHIR `Observation`/`Bundle`, HL7 v2 via MLLP, `/simulator/*` | `/predict` takes one observation. History comes from the database for registered patients (last 4 readings). FHIR and HL7 store readings, then hand them to the continuous monitor. |
| Validation | `schemas.VitalsInput` | Per-field bounds (for example heart rate 0–350, temperature 25–45 °C) and at least 3 measurements. There is no time-series contract: no timestamps in the request and no checks for ordering, duplicates or window. Unknown fields are silently ignored. |
| Feature pipeline | `predictor._build_feature_vector` → `trainer.prepare_features` | The same code as training: deltas, rolling mean/std over 3 rows, missingness indicators, observation gap. Missing values are imputed with training medians. A missing age silently becomes 55. The scaler (none for the committed model) is applied exactly once (tested). |
| Model | `SepsisPredictor.predict` | GradientBoosting 2.0.0, 58 features, sigmoid-calibrated on synthetic data. Integrity is checked against `models/manifest.json` (SHA-256 of each artifact and of the feature schema). |
| Bands | `_model_risk_level` | No `dual_thresholds` in the artifact, so fixed bands apply: moderate ≥ 0.25, high ≥ 0.50, critical ≥ 0.75. **These are engineering placeholders with no clinical derivation.** |
| Rule floor | `scores.compute_scores` | qSOFA, SIRS, NEWS2-style, shock index. `risk_level` is the higher of the rule level and the model level, so the model never lowers the rule level. |
| Alert | `predict` | `alert = rule alert OR level high/critical OR probability > 0.6`. The 0.6 is also an engineering placeholder. |
| Response | `PredictionResponse` | Score, levels, "95% CI", top factors, a research interpretation, model name/version, artifact provenance, `clinical_use: not-permitted`. |
| UI | `frontend/src/pages/Predict.tsx` | Shows the score, both levels, the CI, "Model drivers (not causal)", validation status and "Clinical use: not permitted". |
| Audit | `api._persist_prediction` → `prediction_records`; middleware → `audit_log` | See section 7. |

## 3. Model backends

| Backend | This workstation (macOS, env-s5) | Linux CI and Docker |
|---|---|---|
| GradientBoosting, RandomForest, LogisticRegression | usable | usable |
| LightGBM | unavailable: native runtime (libomp) missing | usable |
| XGBoost | unavailable: native runtime (libomp) missing | usable |

`trainer.load_optional_backend` tells *unavailable* (not installed, or its native runtime is missing) from *broken* (a leftover directory without `__init__`, a failing internal import, a missing estimator class), which raises `BrokenBackendError`. `SEPSIS_REQUIRE_ALL_BACKENDS=1` makes any gap fatal. The served model is GradientBoosting and needs neither optional backend.

## 4. What the model expects

| Item | Contract |
|---|---|
| Fields | temperature (°C), heart_rate (/min), resp_rate (/min), sbp, dbp, map (mmHg), spo2 (%), gcs (3–15), lactate (mmol/L), wbc (×10⁹/L), procalcitonin (ng/mL), on_supplemental_o2, spo2_scale2 (booleans), age_years, five comorbidity flags (`has_hypertension`, `has_diabetes`, `has_ckd`, `has_copd`, `has_heart_failure`). |
| Units | Fixed, as listed. The JSON API does not accept unit declarations. FHIR ingest converts UCUM units. |
| Frequency | The synthetic generator produced roughly hourly observations. There is no minimum frequency and no maximum gap. The observation gap is itself a feature. |
| Window | The current observation plus up to 4 earlier ones. Only the last 3 rows affect the rolling features. |
| Missingness | Any field may be missing (at least 3 measurements are required). Missing values are imputed with training medians, with missingness indicators set. Labs are usually missing. |
| Identifiers | `patient_id` is free text, up to 100 characters. It is stored in plain text in `prediction_records`. History lookup works only for IDs registered in `patients`. |

## 5. What the model returns

| Output | State |
|---|---|
| Score | `risk_probability` in [0, 1]. Calibrated on synthetic data only, so **not a probability for patients**. |
| Threshold | Fixed placeholder bands (section 2), not versioned, and not recorded with the prediction. |
| Horizon | **None.** The committed model was trained on *current-state* labels ("is this row in a sepsis episode"), not "onset within H hours". Any prediction horizon is undefined until a clinician sets one. |
| Uncertainty | No conformal predictor ships. The "95% CI" is ±1.96 × the standard deviation of the last 20% of boosting stages. **That is not a confidence interval** for the patient's risk; it measures how much the last trees moved the output. |
| Explanation | `top_risk_factors` is the model's *global* feature importance, the same five features in the same order for every patient, decorated with textbook reference ranges. The UI labels it "Model drivers". **It is not a per-patient explanation.** |
| Disclaimers | `research_only: true`, `clinical_use: not-permitted`, `validation_status`, `intended_use`, and a research interpretation without treatment advice. |

## 6. Synthetic versus real; validation done

- **All model training and evaluation used synthetic data** from hand-written generators (`ml/synthetic_data.py`). No real or de-identified patient data has been used, and none is in the repository.
- The MIMIC-IV loader is tested only on committed synthetic fixtures. No MIMIC access is authorized.
- Synthetic results (reports/): validation AUROC about 0.92 in model metadata; the no-labs ablation and the profile evaluation have patient-level bootstrap CIs. The synthetic generator ties risk to age (finding M1), so these numbers mostly reflect the generator's assumptions. **They are not evidence of clinical performance.**
- Engineering validation: about 860 Python tests, frontend tests, and 25 CI jobs (all passing on 28e1e0a), including PostgreSQL migrations and browser E2E.
- Not done: retrospective validation on real data, external validation, prospective silent-mode evaluation, subgroup fairness on real data, human-factors evaluation, clinical review.

## 7. Defects found in this review

All eleven are addressed in Phase 1 (see `docs/inference_contract.md`; commit
recorded in PROJECT_REVIEW.md §10). The list below describes the state at
28e1e0a.

1. **Audit persistence fails open.** `_persist_prediction` logs a warning and the API still returns the prediction if the database write fails. The docstring says "the prediction is the priority". Predictions can therefore be shown without any record.
2. **The continuous-monitor path writes no prediction record.** FHIR/HL7 readings → `VitalsIngester.ingest_single` → `predict` → a WebSocket `sepsis_alert`/`deterioration_alert`, with no `prediction_records` row. Alerts from this path are not traceable.
3. **Records are not immutable.** `prediction_records` is an ordinary table with no update/delete guard and no content hash.
4. **Records are not fully traceable.** They do not store the threshold policy, a request ID or an input hash. Artifact SHA-256, feature-schema SHA-256 and pipeline version are stored, inside the `output_scores` JSON.
5. **The uncertainty is mislabelled** (section 5).
6. **The explanation is mislabelled** (section 5).
7. **The UI keeps a stale result on failure.** `Predict.tsx` keeps the previous result on screen when a new request fails.
8. **The UI's comorbidities are silently dropped.** The UI sends `hypertension`, the API expects `has_hypertension`, and Pydantic ignores unknown keys, so the checkboxes have never had any effect.
9. **No data-quality warnings.** Imputed fields, a defaulted age, and missing history are not reported.
10. **No time-series payload contract** (section 2).
11. **A model failure during scoring** surfaces as an unhandled 500, not as a clear "no risk calculated".

## 8. What is missing for a pilot

| Area | Missing |
|---|---|
| Clinical | An intended-use statement, prediction horizon, alert thresholds and their derivation, alert routing and escalation policy, the reference standard (sepsis definition), the human-factors workflow and pilot success/stop criteria. All of these are clinician-owned (`compliance/clinical_requirements_needed.md`). |
| Evidence | Retrospective validation on approved de-identified data, calibration on the target population, subgroup analysis, alert-burden estimates and a silent-mode prospective protocol. |
| Data governance | A data-use agreement, IRB or ethics approval, a de-identification method, a retention policy and a data-provenance record. |
| Security | MFA enforcement for existing users (not enabled without approval), branch protection (not applied), an incident-response process, a penetration test and a privacy impact assessment. |
| Operations | A pre-serve migration step on ECS, a rollback runbook, health checks for artifact and feature-schema compatibility and recent inference success, and structured PHI-free request logs. |

## 9. Main risks

The full risk register is in `docs/pilot/risk_register.md` (Phase 3). In brief:

- the synthetic-only model will not transfer to real patients (dataset shift);
- placeholder thresholds;
- no horizon;
- alert fatigue from the OR-combined alert rule;
- false negatives that the model cannot raise above the rule floor;
- unaudited outputs (defects 1 and 2);
- overreliance on a number that looks like a calibrated probability;
- PHI in free-text `patient_id`.
