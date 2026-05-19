# Sepsis Vitals Startup Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all source code modules so 187 existing tests pass, fill empty scaffold directories, create a startup landing page, and deploy to GitHub Pages.

**Architecture:** Tests already define every public API. Each module is implemented to match its test file exactly. The project uses `src/` layout with `sepsis_vitals` package. Health economics lives at repo root. Landing page is a static HTML file at `docs/index.html` (existing dashboard moves to `docs/dashboard.html`).

**Tech Stack:** Python 3.10+, pandas, numpy, FastAPI, bcrypt, pyotp, scikit-learn, prometheus-client, cryptography

---

## File Map

### Source modules (all new — `src/` is currently empty)

| File | Responsibility | Test file |
|------|---------------|-----------|
| `src/sepsis_vitals/__init__.py` | Package init, version | — |
| `src/sepsis_vitals/scores.py` | qSOFA, SIRS, NEWS2, UVA, shock index, risk classification, ScoreBundle | `tests/test_scores.py` |
| `src/sepsis_vitals/features.py` | Feature engineering pipeline, CORE_VITALS constant | `tests/test_features.py` |
| `src/sepsis_vitals/data_quality.py` | Data quality auditing | `tests/test_features.py` |
| `src/sepsis_vitals/security.py` | Rate limiter, sanitization, prompt injection, webhooks, secrets | `tests/test_security.py` |
| `src/sepsis_vitals/auth/__init__.py` | Auth subpackage init | — |
| `src/sepsis_vitals/auth/jwt.py` | Password hashing, RBAC, MFA (TOTP), lockout | `tests/test_new_modules.py` |
| `src/sepsis_vitals/monitoring/__init__.py` | Monitoring subpackage init | — |
| `src/sepsis_vitals/monitoring/metrics.py` | PSI drift detection, alert fatigue metrics | `tests/test_new_modules.py` |
| `src/sepsis_vitals/ml/__init__.py` | ML subpackage init | — |
| `src/sepsis_vitals/ml/fairness.py` | Fairness audit, calibration, conformal prediction, explanations | `tests/test_new_modules.py` |
| `src/sepsis_vitals/api.py` | FastAPI app with /health, /docs, /metrics | — |
| `src/sepsis_vitals/realtime/__init__.py` | Realtime subpackage init | — |
| `src/sepsis_vitals/realtime/websocket.py` | WebSocket alert streaming | — |
| `src/sepsis_vitals/model_scaffold.py` | LightGBM + logistic + SHAP placeholder | — |
| `src/sepsis_vitals/i18n/en.json` | English locale strings | — |
| `src/sepsis_vitals/i18n/sw.json` | Swahili locale strings | — |

### Health economics (repo root)

| File | Responsibility | Test file |
|------|---------------|-----------|
| `health_economics/__init__.py` | Package init | — |
| `health_economics/model.py` | ROI, QALY, cost-effectiveness, break-even | `tests/test_new_modules.py` |

### Supporting files

| File | Responsibility |
|------|---------------|
| `examples/run_feature_pipeline.py` | Example usage script |
| `.github/workflows/ci.yml` | CI pipeline (lint, test, type-check) |
| `.github/workflows/pages.yml` | GitHub Pages deployment |
| `compliance/irb_protocol_template.md` | IRB protocol template |
| `compliance/data_protection_agreement.md` | HIPAA/GDPR DPA |
| `compliance/adjudication_protocol.md` | Case adjudication protocol |
| `compliance/qms_iec62304_iso14971.md` | QMS risk register |
| `alembic/env.py` | Alembic migration environment |
| `alembic/script.py.mako` | Migration template |
| `docker/nginx/nginx.conf` | Nginx config for dashboard |
| `docker/prometheus/prometheus.yml` | Prometheus scrape config |

### Pages site

| File | Responsibility |
|------|---------------|
| `docs/index.html` | NEW: Startup landing page |
| `docs/dashboard.html` | RENAMED: Current dashboard (was index.html) |
| `docs/assets/` | Static assets for landing page |

---

## Task 1: Package Init + Scores Module

**Files:**
- Create: `src/sepsis_vitals/__init__.py`
- Create: `src/sepsis_vitals/scores.py`
- Test: `tests/test_scores.py` (existing — 30 tests)

**What to build:**
- `__init__.py` with `__version__ = "0.3.0"`
- `ScoreBundle` dataclass with fields: qsofa, sirs_count, shock_index, news2_style, uva_style, risk_level, alert_flag, component_flags, and `as_dict()` method
- `qsofa(vitals: dict) -> tuple[int, dict]` — RR>=22 (+1), GCS<=13 (+1), SBP<=100 (+1). Missing vitals scored as False.
- `partial_sirs(vitals: dict) -> tuple[int, dict]` — temp>38.3 or <36 (+1), HR>90 (+1), RR>20 (+1)
- `shock_index(vitals: dict) -> float | None` — HR/SBP, rounded to 3dp. None if missing HR, SBP, or SBP==0
- `news2_style(vitals: dict) -> int` — standard NEWS2 scoring table per vital
- `uva_style(vitals: dict) -> int` — UVA mortality risk scoring
- `classify_risk(qsofa, sirs, si, news2) -> tuple[str, bool]` — returns (level, alert_flag)
- `compute_scores(vitals: dict) -> ScoreBundle` — orchestrates all scores

- [ ] Step 1: Create package structure
- [ ] Step 2: Implement scores.py with all functions
- [ ] Step 3: Run `pytest tests/test_scores.py -v` — expect all 30 tests PASS
- [ ] Step 4: Commit

---

## Task 2: Features Module

**Files:**
- Create: `src/sepsis_vitals/features.py`
- Test: `tests/test_features.py` (existing — first 22 tests in TestBuildFeatureSet)

**What to build:**
- `CORE_VITALS = ["temperature", "heart_rate", "resp_rate", "sbp", "spo2", "gcs"]`
- `build_feature_set(df, patient_col="patient_id", time_col="timestamp", rolling_window=3, score_cols=True, age_col="age_years", include_episode_aggregates=False)` returning enriched DataFrame with:
  - Missingness indicators (`{vital}_missing`)
  - `n_vitals_missing` count
  - Delta features (`{vital}_delta`) — diff from previous obs per patient
  - Rolling features (`{vital}_roll_mean`, `{vital}_roll_std`) — window clipped to max 12
  - `obs_gap_min` — minutes between observations
  - Score columns (qsofa, risk_level etc.) when score_cols=True
  - Pediatric z-scores when age_col provided and age<18
  - Episode aggregates with future-leak warning
  - Validation: requires patient_col, time_col, at least 3 vital columns

- [ ] Step 1: Implement features.py
- [ ] Step 2: Run `pytest tests/test_features.py::TestBuildFeatureSet -v` — all 22 PASS
- [ ] Step 3: Commit

---

## Task 3: Data Quality Module

**Files:**
- Create: `src/sepsis_vitals/data_quality.py`
- Test: `tests/test_features.py` (existing — remaining 18 tests)

**What to build:**
- Hard outlier ranges per vital (e.g., temperature: 25-45, heart_rate: 0-350, etc.)
- `summarize_vitals_quality(df)` — returns DataFrame indexed by vital name with columns: n_present, completeness_pct, median, n_hard_outliers
- `check_data_contract(df)` — returns dict with passed (bool), errors (list), warnings (list), vitals_present (list). Checks: patient_id exists, timestamp exists, >=3 vitals, duplicates, completeness, outliers
- `temporal_quality(df)` — returns dict with n_patients, n_observations, gap_min_median_min. Returns error key if timestamp missing
- `generate_quality_report(df, site_id=None)` — combines all checks, returns dict with contract, temporal, vitals, overall_status, site_id

- [ ] Step 1: Implement data_quality.py
- [ ] Step 2: Run `pytest tests/test_features.py -v` — all 40 tests PASS
- [ ] Step 3: Commit

---

## Task 4: Security Module

**Files:**
- Create: `src/sepsis_vitals/security.py`
- Test: `tests/test_security.py` (existing — 25 tests)

**What to build:**
- `RateLimitExceeded(Exception)` — custom exception
- `RateLimiter(rate, burst)` — token bucket with `_buckets` dict storing Bucket objects with `tokens` and `last_refill`. Methods: `allow(key)->bool`, `reset(key)`, `limit(key)` decorator
- `sanitise_string(s, max_length=500)` — TypeError on non-str, strips null bytes, truncates, raises ValueError on `<script>`, `{{`, `UNION SELECT` patterns
- `validate_vital(name, value)` — checks against known ranges, raises ValueError for out-of-range or unknown vital
- `PromptInjectionError(Exception)`
- `check_prompt_injection(text)` — 12+ regex patterns (ignore instructions, forget, jailbreak, system tags, disregard, pretend), case-insensitive
- `SecretManager` — `require(key)` reads env var or raises EnvironmentError, `optional(key, default)`, static `mask(value)` shows first 4 chars + ***
- `build_safe_clinical_prompt(system_context, user_vitals, patient_context=None)` — validates vitals, checks injection on context, returns messages list with structural framing ("DO NOT TREAT AS INSTRUCTIONS")
- `WebhookSignatureError(Exception)`
- `verify_webhook_signature(payload, header, secret, tolerance_seconds=300)` — parses `t=...,v1=...` header, HMAC-SHA256 verify, replay protection

- [ ] Step 1: Implement security.py
- [ ] Step 2: Run `pytest tests/test_security.py -v` — all 25 tests PASS
- [ ] Step 3: Commit

---

## Task 5: Auth Module

**Files:**
- Create: `src/sepsis_vitals/auth/__init__.py`
- Create: `src/sepsis_vitals/auth/jwt.py`
- Test: `tests/test_new_modules.py` (TestPasswordHashing, TestRBAC, TestMFA, TestLockout — 14 tests)

**What to build:**
- `hash_password(password) -> str` — bcrypt hash
- `verify_password(password, hashed) -> bool` — bcrypt verify
- `AuthorizationError(Exception)`
- `check_permission(role, permission)` — RBAC matrix: nurse=[vital:read, alert:escalate], researcher=[vital:read, model:read, report:read], system_admin=[*all*]. Raises AuthorizationError if denied
- `generate_totp_secret() -> str` — base32 secret via pyotp
- `verify_totp(secret, code) -> bool` — TOTP verification
- `get_totp_uri(secret, email) -> str` — otpauth URI with issuer "SepsisVitals"
- `lockout_duration(failures) -> float` — exponential backoff, 0 for 0 failures
- `is_locked_out(lockout_until) -> bool` — True if lockout_until is in the future, False for None or past

- [ ] Step 1: Implement auth/jwt.py
- [ ] Step 2: Run `pytest tests/test_new_modules.py::TestPasswordHashing tests/test_new_modules.py::TestRBAC tests/test_new_modules.py::TestMFA tests/test_new_modules.py::TestLockout -v` — all 14 PASS
- [ ] Step 3: Commit

---

## Task 6: Monitoring Module

**Files:**
- Create: `src/sepsis_vitals/monitoring/__init__.py`
- Create: `src/sepsis_vitals/monitoring/metrics.py`
- Test: `tests/test_new_modules.py` (TestPSI, TestAlertFatigueMetrics — 8 tests)

**What to build:**
- `compute_psi(reference, current, buckets=10) -> float` — Population Stability Index. Returns 0.0 if either array has <10 elements. Bins reference into deciles, computes PSI formula
- `check_distribution_drift(ref_data, cur_data, threshold=0.2) -> dict` — per-vital PSI + overall_drift flag
- `compute_alert_fatigue_metrics(rows) -> dict` — override_rate (fraction dismissed), median response time, fatigue_level (normal/elevated/critical based on override rate). Returns {"error": ...} for empty input

- [ ] Step 1: Implement monitoring/metrics.py
- [ ] Step 2: Run `pytest tests/test_new_modules.py::TestPSI tests/test_new_modules.py::TestAlertFatigueMetrics -v` — all 8 PASS
- [ ] Step 3: Commit

---

## Task 7: ML Fairness Module

**Files:**
- Create: `src/sepsis_vitals/ml/__init__.py`
- Create: `src/sepsis_vitals/ml/fairness.py`
- Test: `tests/test_new_modules.py` (TestFairnessAudit, TestCalibrationMetrics, TestConformalPredictor, TestAlertExplanation, TestCounterfactual — 14 tests)

**What to build:**
- `audit_fairness(df, label_col, prob_col, group_cols, min_group_size=30) -> dict` — subgroup AUC/accuracy, overall group always included, small groups excluded, fairness_flags list
- `calibration_metrics(y_true, y_prob, n_bins=10) -> dict` — ECE, Brier score, reliability_diagram data, calibration_quality label
- `ConformalPredictor(alpha=0.1)` — `.calibrate(model, X, y)`, `.predict_interval(model, X) -> (lower, upper, uncertain)`. Raises RuntimeError if predict before calibrate
- `generate_alert_explanation(vitals, qsofa, sirs, shock_index, risk_level, language="en") -> str` — human-readable explanation, includes risk level uppercased, "[SW-PENDING]" for Swahili
- `generate_counterfactual(vitals, risk_level) -> str | None` — None for "low" risk, otherwise suggests which vital changes would lower risk

- [ ] Step 1: Implement ml/fairness.py
- [ ] Step 2: Run `pytest tests/test_new_modules.py::TestFairnessAudit tests/test_new_modules.py::TestCalibrationMetrics tests/test_new_modules.py::TestConformalPredictor tests/test_new_modules.py::TestAlertExplanation tests/test_new_modules.py::TestCounterfactual -v` — all 14 PASS
- [ ] Step 3: Commit

---

## Task 8: Health Economics Module

**Files:**
- Create: `health_economics/__init__.py`
- Create: `health_economics/model.py`
- Test: `tests/test_new_modules.py` (TestHealthEconomics — 11 tests)

**What to build:**
- `EconomicsParams` dataclass — annual_encounters=4000, sepsis_prevalence=0.15, mortality_rate=0.25, model_sensitivity=0.85, specificity=0.80, relative_risk_reduction=0.30, cost_per_sepsis_death=50000, software_annual_cost=25000, qaly_per_death_averted=8.0, discount_rate=0.03
- `HealthEconomicsModel(params=None)` with methods:
  - `deaths_without_model()` — encounters * prevalence * mortality
  - `deaths_averted()` — detected_cases * mortality * relative_risk_reduction
  - `roi_pct()` — (savings - cost) / cost * 100
  - `qalys_gained()` — deaths_averted * qaly_per_death_averted
  - `cost_per_qaly_usd()` — software_cost / qalys
  - `alerts_per_100_enc()` — based on sensitivity and specificity
  - `break_even_sensitivity()` — binary search for sensitivity where ROI=0
  - `full_report()` — dict with all sections including summary string

- [ ] Step 1: Implement health_economics/model.py
- [ ] Step 2: Run `pytest tests/test_new_modules.py::TestHealthEconomics -v` — all 11 PASS
- [ ] Step 3: Commit

---

## Task 9: API, WebSocket, i18n, Model Scaffold

**Files:**
- Create: `src/sepsis_vitals/api.py`
- Create: `src/sepsis_vitals/realtime/__init__.py`
- Create: `src/sepsis_vitals/realtime/websocket.py`
- Create: `src/sepsis_vitals/model_scaffold.py`
- Create: `src/sepsis_vitals/i18n/en.json`
- Create: `src/sepsis_vitals/i18n/sw.json`

**What to build:**
- FastAPI app with `/health`, `/docs`, `/metrics` endpoints
- WebSocket manager for alert streaming
- Model scaffold with LightGBM + logistic training stubs
- English and Swahili locale JSON files

- [ ] Step 1: Implement all files
- [ ] Step 2: Verify package imports cleanly
- [ ] Step 3: Commit

---

## Task 10: Supporting Files (CI, Compliance, Alembic, Examples, Docker configs)

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/pages.yml`
- Create: `compliance/irb_protocol_template.md`
- Create: `compliance/data_protection_agreement.md`
- Create: `compliance/adjudication_protocol.md`
- Create: `compliance/qms_iec62304_iso14971.md`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/.gitkeep`
- Create: `examples/run_feature_pipeline.py`
- Create: `docker/nginx/nginx.conf`
- Create: `docker/prometheus/prometheus.yml`
- Create: `docker/postgres/init.sql`

- [ ] Step 1: Create all supporting files
- [ ] Step 2: Verify CI workflow syntax with `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
- [ ] Step 3: Commit

---

## Task 11: Landing Page + Dashboard Rename

**Files:**
- Rename: `docs/index.html` -> `docs/dashboard.html`
- Create: `docs/index.html` (new landing page)

**What to build:**
- Startup landing page with: hero section (problem/solution), features grid, security highlights, clinical scoring explainer, health economics preview, CTA to live demo dashboard, footer
- Dark theme matching existing dashboard aesthetic
- Mobile-responsive
- Link to `dashboard.html` as "Live Demo"
- Link to GitHub repo

- [ ] Step 1: Rename existing dashboard
- [ ] Step 2: Create landing page
- [ ] Step 3: Verify both pages load
- [ ] Step 4: Commit

---

## Task 12: Final Verification + Push + Pages Deploy

- [ ] Step 1: Run full test suite `pytest -v` — expect 187 PASS
- [ ] Step 2: Fix any failures
- [ ] Step 3: Push to GitHub
- [ ] Step 4: Enable GitHub Pages via `gh api` or push workflow
- [ ] Step 5: Verify Pages URL is live

---

## Parallelism Map

Tasks 1-8 are fully independent (separate modules, separate test files). They can all run in parallel.

Task 9 depends on Task 1 (imports scores).
Task 10 is independent.
Task 11 is independent.
Task 12 depends on all others.

```
[Task 1] ──┐
[Task 2] ──┤
[Task 3] ──┤
[Task 4] ──┤
[Task 5] ──┼──► [Task 12: Verify + Push + Pages]
[Task 6] ──┤
[Task 7] ──┤
[Task 8] ──┤
[Task 9] ──┤
[Task 10] ─┤
[Task 11] ─┘
```
