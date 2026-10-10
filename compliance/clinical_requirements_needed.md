# Decisions and evidence needed before the open clinical items can move

**Status: requirements request, not a specification.** This document lists
what clinicians, the study team and the project owner have to decide or
supply. Nothing here proposes a clinical threshold, horizon, policy or
acceptance criterion. Where the code needs a value today, it uses an
engineering placeholder; those are listed in the last section so they are
not mistaken for clinical decisions.

Until these items are resolved:
- the software is limited to research use on synthetic or authorized
  retrospective data;
- every model output carries `clinical_use: "not-permitted"`;
- no validation status counts as clinically approved
  (`CLINICALLY_APPROVED_STATUSES` is empty).

## Requirements by blocker

| ID | Decision or evidence needed | From whom | What it unblocks | How it is limited today |
|---|---|---|---|---|
| **P1** Intended use | One intended-use statement covering: setting (ward type, country or level of care); users; the decision the output supports; whether output is shown to clinicians (silent mode or not); the population, including exclusions. Draft: `compliance/intended_use_and_validation_plan.md`. | Clinical lead and project owner, then regulatory counsel | Validation protocol, risk file, the choices for M2 and P3 | API and UI label everything research-only; Predict states "Clinical use: not permitted" |
| **M2** Prediction target and horizon | (a) The label definition: Sepsis-3 (which SOFA rise, which infection criterion and window) or another reference standard. (b) The early-warning horizon: hours before onset. (c) How rows after onset are handled. | Clinical lead and study statistician | Retraining with `label_mode="onset_within_horizon"` (implemented and tested; the horizon is a required argument with no default) | Committed model is "current state" detection on synthetic data; the profile report uses 12 h only as a labelled mechanics demonstration |
| **C7** NEWS2 red score and ACVPU | (a) Whether the single-parameter red score (any parameter scoring 3) should escalate the risk level, and how. (b) Whether consciousness is captured as ACVPU (including new confusion) instead of, or as well as, GCS, and how each maps to the score. | Clinical lead (RCP NEWS2 users) | Implementing both in `scores.py` with tests from the approved specification | Every score response includes `news2_limitations` stating both gaps |
| **P3** Acceptance and alerting criteria | (a) Prespecified go/revise/stop gates: which metrics, which thresholds, on which data. (b) The alerting operating point: the target sensitivity, specificity, PPV or alert burden, and for whom. (c) The minimum calibration needed before a probability may be shown. | Clinical lead, study statistician, project owner | Replacing the placeholders listed below; a locked statistical analysis plan | Placeholders are documented below; nothing is described as validated |
| **N18** Emergency access | A break-glass policy: who may use it, under what conditions, scope (read-only? which data?), duration, required justification, approval, alerting, and after-the-fact review. | Clinical governance and privacy officer | Redesigning break-glass (DB-backed, site-bound, time-limited, audited) | `POST /auth/break-glass` always returns 403 and is audited |
| **M5** MIMIC re-evaluation | Authorized PhysioNet credentials and a data-use agreement for MIMIC-IV. Agreement on cohort definition and on whether results are reported in-sample or with a held-out split. | Project owner (data access); study statistician | Retraining and evaluating the MIMIC pipeline after the label fix (N27) | MIMIC-demo metadata lists `known_issues`; loader logic is tested only on a synthetic MIMIC-format fixture |
| **M7** Representative validation | Data from the intended setting, with ethics approval and a data-sharing agreement, plus a protocol (TRIPOD+AI): temporal or external split, subgroups, calibration, decision-curve or alert-burden analysis, missing-data handling. | Site PI, IRB or ethics committee, study statistician | Any statement about performance in patients | All reported numbers are labelled synthetic engineering evidence; bootstrap intervals are explicitly conditional on synthetic test data |

## Engineering placeholders currently in the code (not clinical decisions)

| Where | Value | Origin | Replaced by |
|---|---|---|---|
| `ml/predictor.py`: model level cut-offs | "moderate" at the 95%-specificity threshold, "high" at 99%, "critical" at 99% + 0.15 | Chosen on synthetic validation data by the trainer, plus an arbitrary offset | P3 operating point, on representative data |
| `ml/predictor.py`: alert | rule alert, or model level high/critical, or model output > 0.6 | Engineering default | P3 alerting criteria |
| Combined `risk_level` | higher of the rule-based level and the model level | Safety convention so the model cannot lower a rule flag; it is ordinal, not a probability (documented in `PredictionResponse`) | Clinician review of whether and how the two signals should combine |
| `scores.py`: rule-based levels | NEWS2-style banding, qSOFA, partial SIRS, shock index as implemented | Published score definitions, with the gaps listed in C7 | C7 decisions |
| `schemas.VitalsInput` and FHIR ingestion | Plausibility bounds (for example HR 0-350, SBP 30-300) | Input validation against impossible values | Not a clinical threshold; review with the data-quality plan |
| `ml/profile_evaluation.py` | Operating point at specificity 0.90 | Fixed point for comparing synthetic profiles | P3 |
| Profile report | 12 h horizon | Demonstrates label mechanics only | M2 |

## Confirmed visibly limited

Checked in this stage and covered by tests that run in CI:

- **Backend:**
  - `/model/status` and every prediction report `clinical_use: "not-permitted"` and `clinically_ready: false` (`tests/test_model_artifacts.py`).
  - No API method changes this: POST, PUT and PATCH on `/model/status` return 405 in the compose E2E run.
  - Score responses carry `news2_limitations`.
- **Frontend:**
  - Predict shows the synthetic-data warning, keeps the rule and model levels apart, and states "Clinical use: not permitted". No control enables clinical use (`Predict.test.tsx`; E2E).
  - Live mode shows no demo patients or metrics (`Dashboard.test.tsx`, `PatientDetail.test.tsx`; E2E).
  - The EULA gate precedes the application.
- **Reports:** every report states its claim scope (synthetic engineering evidence) and its limitations. Bootstrap intervals state that they exclude training variability and say nothing about patients.
- **Deployment configuration:**
  - The copilot, billing and treatment bundles are off unless explicitly enabled.
  - Break-glass always returns 403.
  - MFA enforcement is off until the owner turns it on.
  - The model is mounted read-only and its manifest validation status is `synthetic-development`.
