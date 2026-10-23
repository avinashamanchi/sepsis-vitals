# Intended use, target population, and validation plan (DRAFT)

Status: **draft for owner, clinical-lead and statistician sign-off.** Values in
⟨angle brackets⟩ are proposals, not decisions. Freeze this document — version,
date and signatures — **before** any outcome data from a partner site is
examined. Changes after that point must be logged as protocol amendments.

---

## 1. Intended use (investigational)

> Sepsis Vitals is investigational software that computes established
> early-warning scores (UVA, NEWS2, qSOFA) and an experimental
> machine-learning risk estimate from routinely charted vital signs of adult
> inpatients. It is intended **only** for retrospective analysis and for
> prospective **silent-mode** evaluation under an approved research protocol.
> During silent mode its outputs are not shown to treating clinicians and
> are not used for diagnosis, treatment, triage, or escalation of care.

| Element | Proposed value |
|---|---|
| Users | Research staff and study statisticians. **Not** bedside clinicians during silent mode. |
| Patients | Adults (≥18 years) admitted to ⟨general medical wards⟩ of ⟨district / regional hospitals in the partner country⟩. |
| Exclusions | ⟨Paediatrics, obstetrics, ICU admissions, comfort-care-only patients, stays under 12 h⟩ |
| Inputs | Manually charted vital signs: temperature, heart rate, respiratory rate, SBP/DBP, SpO₂, oxygen use, and consciousness (GCS or ACVPU). Plus age and sex. Labs are optional and never required. |
| Output | A risk estimate and score values, logged with model version, input hash and timestamp. |
| Prediction target | Sepsis-3 onset (Seymour 2016 suspected infection plus a SOFA rise of ≥2, or a site-adapted ⟨mSOFA⟩) within ⟨24 h⟩ after the prediction time. Post-onset observations are excluded from scoring. |
| Not intended for | Autonomous alerting, treatment recommendations, paediatric use, ICU monitoring, or any use outside an approved protocol. |

**Regulatory note.** Under FDA's 2022 *Clinical Decision Support Software*
guidance, software that alerts clinicians to a time-critical condition such
as sepsis is generally a device software function, not exempt CDS. Silent
mode keeps outputs away from clinical decisions. Before *any*
clinician-facing display, qualified counsel must assess the pathway for each
market, including the partner country's national regulator.

---

## 2. Why this population and these inputs

- **Labs:** the implied beachhead (Amharic/Swahili locales, SMS delivery,
  offline support) is ward care with intermittent manual observations and
  limited labs. A model that needs procalcitonin cannot be deployed there.
- **Comparators:** UVA was derived in sub-Saharan African adult inpatients.
  NEWS2 and qSOFA are widely known. A new model must show *incremental* value
  over scores a hospital can already use for free.
- **Development data:** MIMIC-IV (US ICU) and the synthetic generator are
  software fixtures only. They must not inform performance claims for this
  population (see `reports/synthetic_pipeline_audit.md`).

---

## 3. Launch gates with measurable criteria

Each gate needs a named owner and a prespecified pass, revise or stop
decision. Thresholds are ⟨proposals⟩ to be ratified before any data is seen.

| Gate | Evidence | Pass criterion (proposal) | Stop criterion (proposal) |
|---|---|---|---|
| G1 Intended use frozen | This document, signed | Signed by owner, clinical lead and statistician | — |
| G2 Data readiness | Data-quality report on the partner extract | ≥⟨80%⟩ of admissions have at least ⟨2⟩ complete vital-sign sets in the first 24 h. Timestamp and unit checks pass. Outcome ascertainment rate ≥⟨90%⟩ | Outcome cannot be ascertained reliably |
| G3 Retrospective discrimination | Locked evaluation on held-out partner data | AUROC for onset within 24 h, lower 95% CI bound ≥ ⟨UVA AUROC + 0.03⟩ | Point estimate ≤ UVA |
| G4 Calibration | Calibration curve, slope and intercept | Slope ⟨0.8–1.2⟩, calibration-in-the-large ⟨±0.05⟩ after at most one prespecified recalibration | Slope < ⟨0.6⟩ |
| G5 Subgroups | Age band, sex, HIV status where recorded, ward, site | No subgroup AUROC more than ⟨0.05⟩ below overall, and every subgroup CI overlaps the overall estimate | Systematic under-performance in a subgroup |
| G6 Silent-mode burden | ≥⟨8⟩ weeks prospective silent logging | ≤⟨5⟩ alerts per 100 patient-days at the operating point; median lead time ≥⟨6 h⟩; PPV ≥⟨15%⟩ | Burden above ⟨2×⟩ the target |
| G7 Human factors | Formative usability sessions | ≥⟨80%⟩ of participants interpret risk, limitations and "not for clinical use" correctly | Repeated critical misinterpretation |
| G8 Safety and operations | Risk file (ISO 14971), downtime/rollback drill, incident owner | All high hazards mitigated; rollback demonstrated | — |

---

## 4. Statistical analysis plan (skeleton)

1. **Unit of analysis.** Admission; predictions are made at each charted
   observation. The primary metric is computed at the admission level using
   the first prediction that crosses the threshold, plus an observation-level
   sensitivity analysis.
2. **Primary outcome.** Sepsis-3 onset within 24 h of the prediction time
   (§1). Post-onset rows are excluded.
3. **Primary metric.** AUROC with a patient-level bootstrap 95% CI
   (2,000 resamples). The UVA comparison uses a paired bootstrap of the
   difference.
4. **Secondary metrics.**
   - AUPRC, calibration slope and intercept, and Brier score.
   - Sensitivity, specificity, PPV and NPV at the prespecified operating
     point.
   - Alerts per 100 patient-days and lead-time distribution.
   - Decision-curve net benefit.
5. **Operating point.** Chosen on development data only, targeting
   ⟨≤5 alerts per 100 patient-days⟩, and locked before validation.
6. **Missing data.**
   - Report missingness by variable and by site.
   - Primary analysis: last-observation-carried-forward within ⟨8 h⟩, then
     the model's own imputation.
   - Sensitivity analysis: complete cases.
7. **Comparators.** UVA, NEWS2 (with oxygen and Scale 2 recorded) and qSOFA,
   computed from the same observations.
8. **Reporting.** TRIPOD+AI checklist. All prespecified analyses are
   reported, including negative results.

---

## 5. Silent-mode data to log (per prediction)

`prediction_id`, `model_version`, `feature_vector` (or its hash plus a
stored copy), `input_hash`, `observation_time`, `prediction_time`,
`display_state = "silent"`, `site_id`, `ward`, and later-linked outcome
fields (`sepsis3_onset_time`, `death`, `icu_transfer`). `PredictionRecord` in
`src/sepsis_vitals/db.py` already stores `model_version`. The other fields
need adding.

---

## 6. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Owner / founder | | | |
| Clinical lead | | | |
| Statistician | | | |
| Regulatory advisor | | | |
