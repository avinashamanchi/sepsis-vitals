# Sepsis Vitals — independent project review

Review date: 2026-10-06. Scope: the full repository at `codex/no-labs-ablation`
(`a0565a6`): backend, ML pipeline, clinical logic, frontend, docs,
compliance drafts, CI and deployment. Method: code reading, runtime
proof-of-concept requests against the API, and reproducible experiments on
the synthetic pipeline and the local MIMIC-IV demo dictionaries. Every claim
below cites a file, a command, or a reproducible report.

Fixes for the most serious findings ship with this review (see
[What this change fixes](#what-this-change-fixes)). Everything else is a
prioritized plan.

> **Stage 2 (2026-10-08):** every finding below was re-checked, 24 new issues
> were found, and most were fixed. The current status of each item is in
> [§7 Consolidated issue register](#7-stage-2-consolidated-issue-register).
> Sections 1-6 are the original stage-1 review, kept as a record.

---

## 1. Executive summary

**What the project is trying to achieve.** Sepsis Vitals wants to be a
*validation-first* sepsis early-warning platform. It does not sell an
unproven alert. It offers hospitals a paid engagement to find out whether a
sepsis signal is safe and useful in their population and workflow:
retrospective external validation first, then a silent-mode prospective
study. The success metric in `STARTUP_REVIEW.md` is the right one: *one
independent hospital dataset, one locked evaluation, one publishable result
that survives clinical scrutiny.* The code hints at a beachhead in
resource-constrained hospitals: Amharic/Swahili locales, Africa's Talking
SMS, offline support, and a "district hospital" no-labs ablation.

**Overall assessment.** The *positioning* is now honest and well judged. The
*evidence and infrastructure* behind it are not yet trustworthy enough to
run the validation engagement it promises. Four problems would each sink a
partner study or a journal review:

1. **Tenant isolation was broken** (fixed here). A nurse at hospital A could
   list, read, write into, and re-home hospital B's patients. Any user could
   switch their own tenant with `PUT /auth/me`. For a multi-site validation
   platform this is disqualifying.
2. **Clinical calculations had material errors** (fixed here).
   - NEWS2 scored a hypoxic patient on oxygen (SpO₂ 90%) as **0 instead of 5**.
   - The MIMIC loader read **platelet count as WBC** and **C-reactive protein
     as procalcitonin**. Procalcitonin is the model's #1 feature.
   - GCS eye/motor items were swapped, and partial GCS was summed as if it
     were a total.
   - The Sepsis-3 windows deviated from the published definition.
3. **The synthetic evidence does not show what it appears to.** In the
   generator, septic rows have *lower* heart rate and *higher* blood pressure
   than non-septic rows. Septic patients are 19 years older, and demographics
   alone give AUROC 0.72. The headline 0.90 AUROC measures recognising rows
   *after* labelled onset. Early-warning discrimination (pre-onset) is 0.70.
   The no-labs ablation inherits all of this.
4. **The live API scores differently from how the model was evaluated.** Each
   request is scored as a first observation, which drops AUROC 0.90 → 0.82
   and **halves mean predicted risk** (0.179 → 0.092). Live risk is
   systematically understated.

There is also a **strategic incoherence** to resolve before the first
partner conversation:
- the implied setting is an LMIC district hospital, with manual vitals and
  few labs;
- the development data is US ICU (MIMIC);
- the model's top features are procalcitonin and age, and procalcitonin is
  rarely available in that setting.

The strongest version of this company picks one setting and freezes an
intended use. It then makes the validation tooling the product, and shows
incremental value over the free scores it already implements (UVA, NEWS2,
qSOFA).

---

## 2. What is working well (preserve these)

- **Honest framing.**
  - The README, Evidence page, EULA and model card all say
    "investigational, not for patient care".
  - A sweep of every locale, page and the built site found no surviving
    "99% specificity", "HIPAA compliant", "SOC 2", "lives saved" or
    mortality claim. Every hit is a disclaimer, e.g.
    `frontend/src/pages/Evidence.tsx:39-41`.
- **Provenance metadata.** `models/model_metadata.json` records data source,
  generator, seed, intended use, limitations and regulatory status. Keep
  this habit and extend it (see §5).
- **Patient-level splits.** `generate_train_val_test` splits by patient
  (`src/sepsis_vitals/ml/synthetic_data.py:898-939`), and the ablation reuses
  identical splits across arms.
- **Sound security primitives.**
  - JWT verification pins the algorithm and requires `sub/exp/iat/type`
    (`auth/tokens.py:300-318`).
  - Production mode refuses to disable auth (`api.py:188-193`).
  - PII is encrypted with AES-256-GCM plus blind indexes (`security.py`).
  - WebSocket tokens travel in the subprotocol, not the URL.
  - The external-LLM copilot is double-gated, and frozen features
    (billing, bundles) are really unmounted.
- **Feature freezes.** Billing, bundles, copilot and forecasts are
  disabled by default. That shows founder discipline.
- **Comparator scores are already implemented.** qSOFA, NEWS2, SIRS, shock
  index and the UVA score (`scores.py`) are exactly the baselines a credible
  validation must beat. UVA was derived in sub-Saharan African inpatients,
  which suits the implied setting.
- **Breadth of tests.** 562 tests passed at baseline. Several earlier audit
  rounds left regression tests behind.
- **Compliance scaffolding.** IRB, adjudication, QMS (IEC 62304 / ISO 14971)
  and DPA drafts are useful starting templates.

---

## 3. Issues to fix

Severity: **Critical** = invalidates a core claim, or a patient-safety or
exploitable-security risk. **High** = materially misleading, or a real bug
in a core path. **Medium** = bounded. **Low** = hygiene. ✅ = fixed in this
change.

### 3.1 Security and privacy

| # | Sev | Issue | Evidence | Status |
|---|---|---|---|---|
| S1 | Critical | No tenant scoping on the patients, FHIR, alerts or monitor routers. The earlier "IDOR fix" (`d0478ae`) only touched `api.py` and `bundles/router.py`. | Proved at runtime: nurse A got `200` on `GET /patients/{B}/history`, `POST /patients/{B}/vitals` and `PUT /patients/{B}` with `site_id=A`, and `GET /fhir/Patient/<B's MRN>`. | ✅ |
| S2 | Critical | Self-service tenant switch: `PUT /auth/me {"site_id": ...}` wrote `User.site_id`, which is the tenant key. | `auth/router.py:526-527` (before the fix) | ✅ |
| S3 | High | `org_id=None` meant "allow all". Self-registered users get `org_id=None`; WebSocket connections without an org received every site's alerts. | `api.py:254-256`, `realtime/websocket.py:70-73` | ✅ fail-closed |
| S4 | High | Notification contacts were global: anyone could list or delete anyone's phone numbers, send SMS to any number via `/alerts/test`, and register arbitrary push endpoints, which the server then POSTs to (SSRF). | `alerts/router.py`, `alerts/dispatcher.py:286-350`, `alerts/push.py:56-82` | ✅ owner-scoped, admin-only test sends, push-host allowlist |
| S5 | Medium | Alert acks recorded as `"unknown"` (the code read `user["sub"]`, but the dict uses `id`), or attributed to a caller-supplied `user_id`. | `alerts/router.py:364`, `patients/router.py:154-166` | ✅ |
| S6 | High | Patient identity is unique across all sites (`external_id_hash unique=True`). A FHIR upsert from site A could overwrite site B's patient, and creating a patient leaks MRN existence. | `db.py:202`, `fhir/router.py:151-160` | ⚠️ partly: cross-site upserts now return a generic 409. A composite `(site_id, MRN)` key still needs a migration. |
| S7 | Medium | Behind nginx/ALB, `TRUSTED_PROXIES` is unset, so all clients share one rate-limit bucket and audit logs record the proxy IP. | `api.py:55-64`; not set in `docker-compose.yml` or `terraform/main.tf` | Open |
| S8 | Medium | MFA is modelled but never verified at login; lockout grows without a cap, so a known email can be locked out indefinitely. | No TOTP check in `auth/service.py` | Open |
| S9 | Medium | Token lifecycle: refresh-token replay detection is documented but not implemented; WebSockets outlive token expiry and revocation; reset tokens are reusable; break-glass tokens have no site and no enforced read-only scope. | `auth/service.py:419-455`, `api.py` websocket | Open |
| S10 | Medium | `docker-compose.yml` publishes Postgres 5432, Redis 6379, Prometheus 9090, Grafana 3001 and MLLP 2575 on all interfaces. | `docker/docker-compose.yml:38,60,95-97,117,135` | Open |
| S11 | Medium | Patient identifiers and contacts are written to logs and to unencrypted SQLite side stores (`models/alert_escalation.db`, `models/patient_state.db`). | `alerts/escalation.py:283-286`, `fhir/listener.py:1135-1140` | Open |
| S12 | Medium | Billing router (disabled by default): any user can open the billing portal for any organisation. | `billing/router.py:220-327` | Open, low urgency while frozen |
| S13 | Low | `auth/jwt.py` keeps a second, weaker JWT implementation and a SQLite `UserStore`, both importable. | `auth/jwt.py:141-341` | Open |

### 3.2 Clinical logic

| # | Sev | Issue | Evidence | Status |
|---|---|---|---|---|
| C1 | Critical | **NEWS2 oxygen handling inverted.** `on_supplemental_o2` switched to SpO₂ Scale 2 instead of adding 2 points, so SpO₂ 90% on oxygen scored **0** (correct: 3 + 2 = **5**). Scale 2 is only for a prescribed 88-92% target. No API input carried oxygen status, so the +2 was never applied anywhere. | `scores.py:101-155`; the old tests asserted the bug (`tests/test_security_fixes.py`) | ✅ RCP-2017 chart, new `spo2_scale2` flag, both flags exposed on `/score` and `/predict`, boundary tests |
| C2 | Critical | **MIMIC lab items mis-mapped.** `51265` (Platelet Count, mean 209.6 in the demo) was loaded as WBC. `50889` (C-reactive protein, mean 70.1 mg/L) was loaded as procalcitonin, the model's #1 feature. WBC is really `51301`/`51300`, and MIMIC-IV has no procalcitonin item. | `ml/mimic_loader.py:67-73`; checked against `d_labitems.csv.gz` and `labevents.csv.gz` | ✅ plus a dictionary-checked regression test |
| C3 | High | **GCS eye/motor items swapped**, and a partial GCS (e.g. verbal not charted when intubated) was summed and treated as a total. A falsely low GCS triggers qSOFA/NEWS2. | `ml/mimic_loader.py:62-64, 205-221` | ✅ requires all three components |
| C4 | Medium | **Sepsis-3 windows deviated from Seymour 2016.** The suspected-infection window was symmetric at ±72 h (should be: culture ≤24 h after antibiotics, or antibiotics ≤72 h after culture). The SOFA window was ±48 h (should be −48 h/+24 h). Both over-label sepsis. | `ml/sepsis3_labeler.py:332-350, 408-437` | ✅ (the legacy symmetric window is kept as an opt-in for sensitivity analysis) |
| C5 | Medium | SIRS temperature threshold was >38.3 °C; the ACCP/SCCM consensus is >38.0 °C. | `scores.py:59` | ✅ |
| C6 | Low | qSOFA uses GCS ≤13 for altered mentation. That is defensible (derivation cohort), but many tools use GCS <15, and the choice was undocumented. | `scores.py:28` | ✅ documented |
| C7 | Medium | NEWS2 has no single-parameter "red score" escalation, and consciousness is GCS-only (no ACVPU or new confusion). | `scores.py`, `classify_risk` | Open: product decision |

### 3.3 ML validity

These numbers come from `reports/synthetic_pipeline_audit.md`, generated by
`python scripts/audit_synthetic_pipeline.py` (6,000 patients, seed 7).

| # | Sev | Issue | Evidence |
|---|---|---|---|
| M1 | Critical | **The synthetic generator encodes "older = septic" and inverts physiology.** Septic rows have HR 80.5 vs 82.7, SBP 137.9 vs 128.0, and lactate 1.42 vs 1.43, compared with non-septic rows. Sick-non-septic patients are deliberately pushed 90% toward "sepsis-like" vitals, while septic trajectories are capped at about 70%. | Demographics plus comorbidities alone: **0.723** AUROC. Age alone, patient level: **0.752**. Vitals and scores only: 0.785. No-labs: 0.840. Full: 0.914. |
| M2 | High | **The headline metric is detection, not early warning.** Labels are positive only *after* onset, so row-level AUROC rewards recognising current sepsis. | Pre-onset rows of future-septic patients vs never-septic rows: **0.700**, against 0.904 headline (95% CI 0.894-0.914). |
| M3 | High | **Train/serve skew.** `SepsisPredictor._build_feature_vector` treats every request as a first observation: deltas, rolling std and observation gap are NaN, and the rolling mean equals the current value. | Live-style features: AUROC **0.824** and mean risk **0.092**, against 0.904 and 0.179 for training-style features. Live risk is understated. Documented at `ml/predictor.py`. |
| M4 | Medium | **Non-monotonic timestamps.** `hours_offset = i * rng.uniform(2, 6)` draws a new interval scale per row, so **32%** of consecutive observations go backwards in time. That corrupts deltas and `obs_gap_min`. | `ml/synthetic_data.py:586` |
| M5 | Medium | **MIMIC-demo metrics are in-sample.** Below 200 patients, `train.py` sets train = val = test, so the stored `val_*` metrics equal `train_*`, and the dual thresholds were chosen on the same rows. The artifact also predates fix C2. | `train.py:104-109`; flagged in `models/mimic-demo/model_metadata.json` `known_issues` ✅ |
| M6 | Medium | **No uncertainty in reported metrics.** The ablation and model card report point estimates only. | `reports/no_labs_ablation.md`; the new audit adds a patient-bootstrap CI |
| M7 | Medium | **TRIPOD+AI gaps.** Missing: calibration slope and intercept, decision curves, alert rate per patient-day, PPV at the operating point, lead-time distribution, and a comparison against UVA/NEWS2/qSOFA. | — |

### 3.4 Strategy, product and documentation

| # | Sev | Issue | Why it matters |
|---|---|---|---|
| P1 | High | **No single intended-use statement.** The README says "research and silent-mode", while the code serves LMIC SMS alerting, ICU-style monitoring, FHIR ingestion and a public demo. | Regulators, IRBs and hospital buyers all start from intended use. Without one, the validation protocol, risk file and success criteria cannot be written. Draft: [`compliance/intended_use_and_validation_plan.md`](compliance/intended_use_and_validation_plan.md). |
| P2 | High | **Setting/data/model mismatch.** The setting is an LMIC ward (manual vitals every 4-8 h, few labs). The data is a US ICU (MIMIC). The model's top feature is procalcitonin. | The model being built cannot be deployed where the company says it will work. A no-labs, vitals-only model benchmarked against UVA is the coherent product. |
| P3 | High | **Launch gates are not testable.** The 8 gates in `STARTUP_REVIEW.md` say "acceptable" without thresholds. | A paid validation engagement needs prespecified, numeric go/revise/stop criteria. Otherwise negative results get rationalised. |
| P4 | Medium | **Positioning lacks a buyer and deliverables.** "Paid validation platform" doesn't say who signs (CMIO, quality lead, ministry, NGO consortium) or what they receive. | Suggested deliverable: a locked SAP, a data-readiness report, a TRIPOD+AI validation report against local comparators, a silent-mode alert-burden report, and a go/revise/stop memo. |
| P5 | Medium | **Regulatory framing is too soft.** Under FDA's 2022 Clinical Decision Support guidance, software that alerts to a time-critical condition such as sepsis is treated as a device function, not exempt CDS. Silent mode is the right path. | Plan the regulatory pathway (and the LMIC national regulator) per market with counsel before any clinician-facing display. |
| P6 | Low | **Stale or misleading root docs.** `AUDIT_INSTRUCTIONS.md` is an old agent prompt that still says auth is Firebase. `STARTUP_REVIEW.md` reads as a changelog. | A clinical partner should land on README → intended use → evidence. |
| P7 | Low | **License.** MIT for investigational medical software is unusual. | The warranty disclaimer helps, but decide deliberately (e.g. keep the code MIT and keep trained models and clinical content under a separate, restrictive license). |

### 3.5 Architecture, frontend and DevEx

| # | Sev | Issue | Evidence |
|---|---|---|---|
| A1 | High | **The CI typecheck job was red.** mypy reported 1 error. | `model_scaffold.py:80` ✅ |
| A2 | Medium | **CI ran only on PRs into `main`.** Branch PRs went unchecked; there was no concurrency cancel and no pip cache. | `.github/workflows/ci.yml` ✅ |
| A3 | Medium | **The test suite takes about 9 minutes.** Eight MIMIC/CSV integration tests take about 470 of the 535 seconds (50-70 s each), each reloading the demo dataset. | `pytest --durations=15` |
| A4 | Medium | **God modules.** `api.py` (1,532 lines) and `fhir/listener.py` (1,576 lines). | — |
| A5 | Medium | **Dead code.** `state.py`, `ml/forecast.py`, `ml/ensemble.py` and `health_economics/` are imported only by tests. `auth/jwt.py` holds a legacy JWT. The untracked `files/` directory holds stale duplicates of bundle code. | `grep` import audit |
| A6 | Medium | **Dependency bloat.** `anthropic` is a *core* dependency although the copilot is frozen. `stripe`, `twilio`, `africastalking` and `pywebpush` sit in the base `api` extra. `python-jose` and `PyJWT` are both installed. There is no lockfile. | `pyproject.toml` |
| A7 | Medium | **The built site is committed to `docs/`** (`vite outDir: '../docs'`), and Pages deploys only when `docs/**` changes. The deployed site silently drifts from `frontend/src`. | `frontend/vite.config.ts`, `.github/workflows/pages.yml` |
| A8 | Medium | **Frontend/backend contract drift.** The dashboard expects `predictions_today`/`avg_response_min`; the API returns `recent_predictions`. The frontend also always sent `site_id=default`. | `frontend/src/lib/api.ts:190-196`, `patients/router.py:177-182`. Site parameter ✅; field names open. |
| A9 | Low | **Thin frontend tests.** There is one frontend test file. | `frontend/src/__tests__/` |
| A10 | Low | **`DriftMonitor`'s `asyncio.Event` is bound to the first event loop**, so a restarted app crashes the monitor task. | `monitoring/drift_monitor.py:170` ✅ |
| A11 | Low | **Missing repo basics.** No `SECURITY.md`, `CONTRIBUTING.md`, Dependabot, `pip-audit` or pre-commit. The local `.venv` points at an old path. | — |

---

## 4. Recommended improvements (prioritized)

### High priority — next 2-4 weeks

1. **Freeze the intended use and the beachhead** (P1, P2). Adopt or edit the
   draft in
   [`compliance/intended_use_and_validation_plan.md`](compliance/intended_use_and_validation_plan.md).

   *Recommendation:* adult medical-ward inpatients in district or regional
   hospitals, with manually charted vital signs, scored in silent mode.

   *Alternative:* the US/EU ward market (Epic/Cerner integration, CMS SEP-1
   pressure). Its competition is strong: the Epic Sepsis Model's external
   validation failure, TREWS, and an FDA-authorized diagnostic (Prenosis,
   2024). The LMIC thesis is more differentiated and better matches what is
   already built.
2. **Rebuild the synthetic generator, then regenerate every artifact in one
   change** (M1, M4).
   - Use monotonic timestamps: cumulative sums of per-interval draws.
   - Make septic trajectories physiologically *worse* than mimics on average.
   - Decouple age from the label, or match on it.
   - Make labs optional by setting.

   Then retrain, rerun `run_no_labs_ablation.py` and
   `audit_synthetic_pipeline.py`, and update the README numbers. Until then,
   treat every synthetic metric as a software smoke test.
3. **Remove the train/serve skew** (M3). Build live features from stored
   history with the same `prepare_features` used in training. Add a test that
   identical histories produce identical feature vectors in both paths, and
   an alert if feature drift exceeds a threshold.
4. **Redefine the evaluation target** (M2, M7). Predict onset within *N*
   hours. Exclude post-onset rows from scoring, or censor at onset. Report
   lead time, alerts per 100 patient-days, PPV and sensitivity at the
   prespecified operating point, calibration slope and intercept, and decision
   curves, each with patient-level bootstrap CIs. Always include UVA, NEWS2
   and qSOFA as comparators.
5. **Close the remaining security gaps before any second site** (S6-S11).
   - **Patient identity:** make the key `(site_id, external_id_hash)` (Alembic
     migration). Derive the FHIR ingest site from the client certificate or
     user.
   - **Auth hardening:** enforce MFA for clinical roles. Cap lockouts with
     exponential backoff. Rotate refresh tokens with replay detection.
     Re-validate WebSockets on expiry and revocation. Make reset tokens
     single-use. Bind break-glass to a site and make it read-only.
   - **Network:** set `TRUSTED_PROXIES` in compose and ECS, and bind compose
     ports to `127.0.0.1`.
   - **PHI at rest:** drop PHI from logs, and move the SQLite side stores into
     the encrypted primary DB.
6. **Make the launch gates numeric** (P3). The validation plan proposes
   starting thresholds for a clinical lead and statistician to ratify *before*
   any outcome data is seen.

### Medium priority — next 1-3 months

7. **Turn validation into the product** (P4).
   - Package the data-readiness checks (`data_quality.py`), the cohort builder
     (Sepsis-3 labeler), and a locked evaluation runner that emits a
     TRIPOD+AI-structured report.
   - Add silent-mode logging. `PredictionRecord` already stores
     `model_version`; add the feature vector, an input hash, display state
     ("silent") and outcome linkage.

   This is the deliverable a partner pays for, and it is reusable across
   models.
8. **Slim the codebase** (A5, A6). Delete the dead modules, `files/`,
   `AUDIT_INSTRUCTIONS.md` and the legacy JWT. Move `anthropic`, `stripe`,
   `twilio`, `africastalking` and `pywebpush` into opt-in extras. Drop
   `python-jose`. Add a lockfile (`uv lock` or `pip-tools`).
9. **Cut the test suite to under 2 minutes** (A3).
   - Load the MIMIC demo once in a session-scoped fixture.
   - Mark model-training tests `@pytest.mark.slow` and run them in a separate
     nightly CI job.
   - Train on 300-patient cohorts in unit tests.
10. **Split the god modules** (A4). Split `api.py` into `routers/score.py`,
    `routers/predict.py`, `routers/monitor.py`, `routers/simulator.py`,
    `routers/copilot.py` and `realtime/ws.py`, plus an `app.py` factory.
    Split `fhir/listener.py` into `mllp_server.py`, `hl7_parser.py`,
    `webhook.py` and `ingest.py`.
11. **Build the Pages site in CI** (A7). Set the Vite `outDir` to `dist`. Have
    the Pages workflow run `npm ci && npm run build` and upload
    `frontend/dist`. Delete the committed `docs/` build, which frees `docs/`
    for real documentation.
12. **Fix the frontend/backend contract** (A8). Generate TypeScript types
    from the OpenAPI schema, e.g. `openapi-typescript`. Add frontend tests for
    the risk display, the EULA gate, and the not-for-clinical-use banner on
    every clinical page.

### Low priority

13. Add `SECURITY.md` (with a disclosure contact), `CONTRIBUTING.md`,
    Dependabot, a `pip-audit` CI step and a pre-commit config (ruff + mypy).
14. Decide model/content licensing separately from code licensing (P7).
15. Reorganise the root: a README for each audience (clinical partner,
    engineer, reviewer); move `STARTUP_REVIEW.md` and this review into a
    `strategy/` folder once `docs/` is freed.
16. Add the NEWS2 single-parameter red score and an ACVPU input (C7), after
    clinical sign-off.

---

## 5. Improved version

### What this change fixes

| Area | Change | Files |
|---|---|---|
| Tenant isolation (S1-S5) | A new fail-closed policy module: only `system_admin` is unscoped, and every other role is restricted to its site. A user with no site gets 403. Cross-site lookups return 404. | `src/sepsis_vitals/auth/scope.py` (new) |
| | Applied to the patients, FHIR, alerts and monitor routers, the WebSocket stream, and dashboard aggregates. | `patients/router.py`, `patients/service.py`, `fhir/router.py`, `alerts/router.py`, `api.py` |
| | `PUT /auth/me` can no longer change a user's site. A new admin-only `PUT /auth/users/{id}/site` writes an audit log line. | `auth/router.py` |
| | Acks are attributed to the authenticated user. Contacts are owner-scoped. Test sends and delivery history are admin-only. Push endpoints must be https on known push services. | `alerts/router.py`, `patients/router.py` |
| | `/predict` and `/predict/batch` refuse to write predictions against another site's registered patient. 21 two-site integration tests. | `api.py`, `tests/test_tenant_isolation.py` (new) |
| NEWS2 (C1) | Corrected oxygen handling. `on_supplemental_o2` and `spo2_scale2` are accepted by `/score` and `/predict`. | `scores.py`, `api.py` |
| | Boundary tests for every NEWS2 band. | `tests/test_scores.py`, `tests/test_security_fixes.py` |
| SIRS / qSOFA (C5, C6) | SIRS temperature >38.0 °C; the qSOFA GCS choice is documented. | `scores.py` |
| MIMIC loader (C2, C3) | Correct WBC and lactate items, no fake procalcitonin, GCS components un-swapped, partial GCS no longer summed. | `ml/mimic_loader.py` |
| | Regression test against the local dictionaries. | `tests/test_mimic_itemids.py` (new) |
| Sepsis-3 labeler (C4) | Seymour 2016 asymmetric infection windows and the −48 h/+24 h SOFA window. | `ml/sepsis3_labeler.py`, `tests/test_sepsis3_labeler.py` |
| ML honesty (M1-M5) | A reproducible audit of generator artefacts, feature-group AUROCs, early-warning discrimination, train/serve skew and bootstrap CIs. | `ml/pipeline_audit.py`, `scripts/audit_synthetic_pipeline.py`, `reports/synthetic_pipeline_audit.{md,json}`, `tests/test_pipeline_audit.py` |
| | Known issues recorded on the MIMIC-demo artifact; the skew is documented in the predictor. | `models/mimic-demo/model_metadata.json`, `ml/predictor.py` |
| CI (A1, A2) | mypy fixed. CI runs on every PR, with concurrency cancel, pip cache, `--durations`, and actions aligned with `pages.yml`. | `model_scaffold.py`, `.github/workflows/ci.yml` |
| Reliability (A10) | `DriftMonitor` creates its stop event per start. | `monitoring/drift_monitor.py` |
| Strategy (P1, P3) | Draft intended-use statement, target population, numeric launch gates, and a statistical analysis plan skeleton. | `compliance/intended_use_and_validation_plan.md` (new) |
| README | Corrected the interpretation of the synthetic evidence; links to the audit and this review. | `README.md` |

**Behaviour changes to note.**
- Non-admin users without a site assignment now get 403 on patient
  endpoints. Assign sites with `PUT /auth/users/{id}/site`.
- `PUT /auth/me` with `site_id` returns 403 for non-admins.
- `/alerts/test` and `/alerts/history` are admin-only.
- `/patients/dashboard/stats` defaults to the caller's site. Administrators
  may pass `site_id` or omit it to get all sites.
- NEWS2 totals change for any caller that passed `on_supplemental_o2=True`.
  The old behaviour was clinically wrong.

### Rewritten positioning (proposed README opening)

> **Sepsis Vitals helps hospitals find out — before anyone acts on an
> alert — whether a sepsis early-warning signal is accurate, well-calibrated,
> and workable on their wards.** We run a locked retrospective validation on
> your data against the scores you could use for free (UVA, NEWS2, qSOFA),
> then a silent-mode study that measures lead time and alert burden without
> touching care. You get a go / revise / stop decision backed by a
> TRIPOD+AI-structured report. Our software is investigational and is never
> shown to treating clinicians during a study.

---

## 6. Questions and assumptions

| Question | Why it matters | Assumption used in this review |
|---|---|---|
| Which setting is the beachhead: LMIC district or regional wards, or US/EU hospitals? | It drives the model features (labs or no labs), the comparators, the regulator, the buyer and the data partner. | LMIC adult medical wards, inferred from the locales, SMS and no-labs work. |
| Who is the first paying customer: a hospital, a ministry, an NGO or research consortium, or a funder? | It sets the deliverables, the price and the contracting path. | A research-active hospital or academic consortium. |
| Is there a data partner with labelled ward data, including infection and organ-dysfunction outcomes? | Without one, the 90-day plan cannot start; MIMIC does not represent the setting. | Not yet. |
| Team composition: is there a clinician lead and a statistician? | The numeric gates and the SAP need their sign-off; the thresholds proposed here are placeholders. | Founder-led engineering team. |
| Are the committed synthetic artifacts used in any external material (pitch decks, grant applications)? | If so, the M1-M3 caveats must travel with them. | Repository use only. |
| Should `system_admin` remain the only cross-site role? | A multi-site study coordinator may need read-only cross-site access; that would be a new role, not a reason to loosen `org_id=None`. | Yes. |

### Verification performed for this change

- `pytest`: **632 passed** (562 before this change, plus 70 new), 0 failed, 535 s.
- `ruff check src tests scripts`: clean. `mypy src/sepsis_vitals`: clean.
  `bandit -r src -ll`: clean.
- **Frontend:** not built locally (node was unavailable in the review
  environment). The one-line `api.ts` change is covered by the CI `frontend`
  job.

---

## 7. Stage 2: consolidated issue register

Re-checked on 2026-10-08 against branch `review/independent-review-fixes`.
Each status was re-verified in this pass. Earlier claims were not assumed to
still hold.

**Status key**
- ✅ Fixed and verified locally (a test or command reproduces the original failure and now passes).
- 🔵 Implemented; verification depends on CI (frontend build, Docker build, real Postgres), because node, Docker and Postgres are unavailable locally.
- ◐ Partially fixed.
- ❓ Needs an owner decision.
- ⛔ Unresolved or blocked.

### 7.1 Stage-1 findings: current status

| # | Sev | Finding | Status | Evidence / acceptance test |
|---|---|---|---|---|
| S1 | Critical | No tenant scoping on patient, FHIR, alert, monitor and WebSocket paths | ✅ | `tests/test_tenant_isolation.py`: 36 two-site tests |
| S2 | Critical | `PUT /auth/me` let users switch their own site | ✅ | `test_user_cannot_switch_own_site` |
| S3 | High | `org_id=None` meant "allow all" | ✅ | `test_unassigned_user_sees_nothing`, `test_websocket_rejects_orgless_non_admin` |
| S4 | High | Global contacts, SMS relay, push-endpoint SSRF | ✅ | `test_notification_contacts_are_owner_scoped`, `test_test_alerts_and_delivery_history_are_admin_only`, `test_push_endpoint_allowlist` (7 cases); tests added in stage 2 |
| S5 | Medium | Alert acks recorded as `unknown` or a caller-supplied user | ✅ | `test_alert_ack_is_attributed_to_authenticated_user_and_scoped` |
| S6 | High | MRN unique across sites; FHIR upsert could overwrite another site's patient | ✅ locally / 🔵 Postgres | Composite `(site_id, external_id_hash)` key in the ORM and migration 003; per-site lookups; `test_same_mrn_may_exist_at_two_sites`. Real-Postgres check: `postgres-migrations` CI job |
| S7 | Medium | `TRUSTED_PROXIES` unset behind nginx/ALB | 🔵 | Set in compose (fixed subnet) and Terraform (VPC CIDR). Not runnable locally |
| S8 | Medium | MFA never enforced; lockout uncapped | ◐ | Lockout capped at 15 min (`test_lockout_is_capped_and_never_overflows`). MFA enforcement ❓ needs an enrolment UX decision |
| S9 | Medium | Token lifecycle gaps | ◐ | ✅ refresh replay revokes all sessions; ✅ single-use reset tokens; ✅ WebSocket closes at token expiry. Break-glass ❓ (N18) |
| S10 | Medium | Compose published DB, Redis, monitoring and MLLP on all interfaces | 🔵 | Bound to `127.0.0.1`. `docker compose` is not available locally |
| S11 | Medium | PHI in logs and plaintext SQLite stores | ◐ | ✅ `log_ref()` in listener, escalation and monitor logs; ✅ reset flow no longer logs email. ⛔ `models/patient_state.db` and `models/alert_escalation.db` remain plaintext |
| S12 | Medium | Billing org IDOR (billing frozen) | ⛔ | Unchanged; fix before enabling billing |
| S13 | Low | Legacy JWT implementation in `auth/jwt.py` | ❓ | Dead-code removal awaits your approval (A5) |
| C1 | Critical | NEWS2 oxygen handling | ✅ | `TestNEWS2Scale2`, band tests in `test_scores.py` |
| C2 | Critical | MIMIC WBC/procalcitonin item IDs | ✅ | `tests/test_mimic_itemids.py` (the dictionary check runs only where the demo data exists; see N17) |
| C3 | High | GCS items swapped; partial GCS summed | ✅ | `test_gcs_components_are_not_swapped` |
| C4 | Medium | Sepsis-3 windows | ✅ | `TestSeymourWindows` |
| C5 | Medium | SIRS temperature >38.3 °C | ✅ | `test_sirs_temperature_threshold` |
| C6 | Low | qSOFA GCS cut-off undocumented | ✅ | Documented in `scores.py` |
| C7 | Medium | No NEWS2 red score or ACVPU | ❓ | Needs clinical sign-off |
| M1 | Critical | Generator ties the label to age and inverts physiology | ❓ | Still true. New evidence N19. Redesign and retraining await your decision |
| M2 | High | Headline AUROC measures detection, not early warning | ❓ | Audit reports pre-onset AUROC 0.725. Changing the prediction target is a product decision |
| M3 | High | Train/serve skew | ✅ | Live inference uses the training pipeline plus recorded history (`tests/test_inference_parity.py`). Unregistered IDs are still scored as first observations, by design |
| M4 | Medium | 32% of synthetic timestamps ran backwards | ✅ | `test_synthetic_timestamps_are_strictly_increasing`. For the same seed all other generated values are unchanged |
| M5 | Medium | MIMIC-demo metrics in-sample; pre-fix mappings | ◐ | Recorded in `known_issues`; artifact not retrained |
| M6 | Medium | No uncertainty in reported metrics | ◐ | Audit reports a patient-bootstrap CI; ablation report does not |
| M7 | Medium | TRIPOD+AI evaluation gaps | ⛔ | Plan in `compliance/intended_use_and_validation_plan.md` |
| P1-P5, P7 | — | Intended use, beachhead, numeric gates, buyer, regulatory path, licensing | ❓ | Drafts exist; these are owner decisions |
| P6 | Low | Stale `AUDIT_INSTRUCTIONS.md` at repo root | ❓ | Deletion awaits your approval |
| A1 | High | CI typecheck red (mypy) | ✅ | CI typecheck passed on PR #4 |
| A2 | Medium | CI only ran on PRs into `main` | ✅ | CI ran on PR #4 |
| A3 | Medium | Slow test suite | ◐ | MIMIC/integration tests share one session fixture: about 415 s down to 125 s locally. Model-training tests are still slow |
| A4 | Medium | God modules (`api.py`, `fhir/listener.py`) | ⛔ | Not attempted; a large refactor is out of scope for a fix pass |
| A5 | Medium | Dead code (`state.py`, `ml/forecast.py`, `ml/ensemble.py`, `health_economics/`, legacy JWT) | ❓ | Removal awaits your approval |
| A6 | Medium | Dependency bloat | ◐ | ✅ added the missing Postgres driver and dropped unused `asyncpg`. ⛔ `anthropic` is still a core dependency; payment/SMS SDKs remain in `api` |
| A7 | Medium | Built site committed to `docs/` | ❓ | Changing the Pages deployment needs your approval |
| A8 | Medium | Frontend/backend contract drift | 🔵 | Dashboard, Analytics and Patients now use real fields. Verified by CI lint/test/build only |
| A9 | Low | Thin frontend tests | ⛔ | Not addressed |
| A10 | Low | `DriftMonitor` event bound to a stale loop | ✅ | No more "different event loop" errors in TestClient restarts |
| A11 | Low | No `SECURITY.md`, Dependabot or pip-audit | ⛔ | Not addressed |

### 7.2 New issues found in stage 2

| # | Sev | Issue (affected area) | Evidence | Status / verification |
|---|---|---|---|---|
| N1 | High | **Frontend dependencies have 12 known vulnerabilities (2 critical, 8 high)**: tinypool via vitest, react-router 7.12-7.18.1, postcss, nanoid, fast-uri, source-map-js. CI on `main` has been red since July. | `npm audit --audit-level=high` step in CI run 37511478654 | ⛔ Blocked. `npm` is blocked in this workspace, so the lockfile cannot be regenerated here (commands in the PR description) |
| N2 | High | **Plaintext GitHub OAuth token in `.claude/settings.local.json`** (local file, gitignored) | Allow-rule strings contain a `gho_` token. `git log --all -S` finds no commit | ⛔ Your action: revoke and rotate the token |
| N3 | Critical | **Alembic migrations missing ORM columns**: `users.email_hash`, `patients.external_id_hash`, `vitals.dbp/map/lactate/wbc/procalcitonin`. Login fails on a migrated database | Offline DDL diffed against ORM metadata | ✅ Migration 003 plus `tests/test_migrations.py` (fails without 003, passes with it). 🔵 Real Postgres in CI |
| N4 | Critical | **Encrypted columns overflow `VARCHAR(64)`**: a 15-character MRN encrypts to 64+ chars; `totp_secret` also overflows | Measured with `FieldEncryptor` | ✅ Migration 003 widens to TEXT (`test_encrypted_columns_are_not_length_limited`). 🔵 CI inserts a 32-char MRN on Postgres |
| N5 | Critical | **`docker/postgres/init.sql` created a stale schema**, so the compose stack could not log in | `init.sql` users table has no `email_hash`; `create_all` never adds columns | 🔵 `init.sql` is now extensions only; the container runs `alembic upgrade head` via `docker/entrypoint.sh` |
| N6 | Critical | **No synchronous Postgres driver installed** by `pip install .[api]`, so the API and Alembic cannot connect | `create_engine('postgresql://…')` raised `ModuleNotFoundError: psycopg` | ✅ `psycopg[binary]` added; URLs normalised (`test_database_urls_use_installed_sync_driver`). 🔵 CI Postgres job |
| N7 | Critical | **Frontend API calls never reached the backend**: no `/api` prefix, a Vite proxy with no rewrite, and nginx proxying only `/api/`, so login broke in dev and compose. A trailing-slash redirect also escaped the prefix | `api.ts` used `BASE=''`; `vite.config.ts`; `nginx.conf` | 🔵 Default base `/api`, the Vite proxy strips it, the WebSocket URL resolves relative bases. CI build only; not exercised end-to-end |
| N8 | High | **Patients list showed fabricated demo patients in live mode**, and rendered unobserved patients as "low" risk with vitals of 0 | `useState(DEMO_PATIENTS)`, `?? 'low'`, `?? 0` | ✅ backend: `GET /patients` returns latest observation or null (`test_patient_list_reports_latest_observation_or_null`). 🔵 frontend shows "—" and "Not scored", plus error and empty states |
| N9 | High | **Dashboard and Analytics showed invented numbers in live mode**: placeholder counts, `predictions_today × 7` (a field that does not exist), synthetic AUROC defaults | `Analytics.tsx`, `Dashboard.tsx` | 🔵 Live mode shows "—" until loaded; Analytics uses the real weekly-trends and risk-distribution endpoints |
| N10 | High | **Password reset was unfinished**: email delivery was a `TODO`; no UI accepted the token; the requester's email was logged | `auth/router.py`; no confirm form in `Login.tsx` | ✅ backend: SMTP mailer, token in the URL fragment, sent in the background, uniform response, no email in logs (`test_reset_*`). 🔵 frontend set-new-password form |
| N11 | Medium | **Login lockout overflow**: `float(2**n)` raises after about 1,025 failures (500 on login); `failed_attempts` SMALLINT can overflow | `auth/jwt.py` | ✅ capped; counter bounded |
| N12 | High | **Docker API image could not serve predictions or run migrations**: only `[api]` installed (no scikit-learn); `pip install … \|\| true` hid failures; no Alembic entrypoint; no `libgomp1` | `docker/Dockerfile` | 🔵 `[api,ml]`, the failure surfaces, entrypoint migrations, `libgomp1`. CI `docker` job builds and imports. Model artifacts are still not in the image ❓ |
| N13 | Low | Compose required `ANTHROPIC_API_KEY` although the copilot is frozen | `docker-compose.yml` | 🔵 now optional |
| N14 | Medium | **Test suite wrote to the developer's `./sepsis_vitals.db`** | DB modification time changed on every run | ✅ `tests/conftest.py` uses a temporary DB; modification time unchanged after the full run |
| N15 | Medium | **Order-dependent monitor tests** (`asyncio.get_event_loop()`) failed after other async tests | 6 failures reproduced on the *original* code in a different order | ✅ `asyncio.run`; passes in both orders |
| N16 | Medium | **`GET /patients/alerts` was unreachable**: shadowed by `/patients/{patient_id}`, returned 404 | Reproduced with TestClient | ✅ Route order fixed (`test_active_alert_list_is_routable`) |
| N17 | Medium | **MIMIC tests always skip in CI** (demo data is gitignored), so CI never exercises the MIMIC loader | `HAS_DEMO_DATA` guards; CI logs | ❓ Options: download the open-access demo in CI, or add a small MIMIC-format fixture |
| N18 | Medium | **Break-glass access is non-functional**: its token's user does not exist and it has no site | `auth/service.py` break-glass issuance | ❓ Fail-closed today; needs a redesign (DB-backed, site-bound, read-only) or removal |
| N19 | High | **The committed model assigns ~0.7% sepsis probability to a patient with HR 118, RR 24, SBP 96, lactate 2.4**, whom the rule-based scores call critical | Probe in stage 2; consequence of M1 | ❓ Requires the generator redesign and retraining (M1) |
| N20 | Low | Deprecated Pydantic `.dict()` in the predict endpoints | Test warnings | ✅ `model_dump()` |
| N21 | Low | History rows without timestamps became `NaT` and broke ordering | mypy `arg-type` error | ✅ skipped |
| N22 | Low | `generate_validation_report.py` writes into `docs/`, which the next frontend build wipes | default `--output docs/...` | ⛔ Not addressed |
| N23 | Low | Revocation has 1-second granularity, so a login in the same second as a password reset can be revoked | `TokenBlacklist.is_revoked` uses `<=` on integer seconds | ⛔ Not addressed (rare) |
| N24 | Low | `asyncpg` declared but never used | grep | ✅ removed |


### 7.3 Verification performed in stage 2 (local)

| Check | Result |
|---|---|
| `pytest` (full suite) | **667 passed, 0 failed**, 239 s (stage 1: 632 passed in 535 s) |
| `ruff check src tests scripts` | clean |
| `mypy src/sepsis_vitals` | clean (it caught a real edge case, N21) |
| `bandit -r src -ll` | no medium/high findings |
| Drift test fails without migration 003 | confirmed: missing columns, VARCHAR types and global MRN uniqueness all detected |
| Pre-existing failures reproduced on the original code before fixing | N15 (6 order-dependent failures), N16 (404 on `/patients/alerts`) |
| Local dev DB untouched by the test run | modification time unchanged |

Not verifiable locally: anything needing node (frontend lint/test/build), Docker, `docker compose`, or PostgreSQL. These are covered by the CI jobs `frontend`, `docker` and `postgres-migrations` on PR #4.
