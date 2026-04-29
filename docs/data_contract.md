# Data Contract

This contract describes the minimum data needed to move from Phase 0 scoping to
Phase 1 dataset construction.

## Required Encounter Fields

| Column | Type | Notes |
| --- | --- | --- |
| `patient_id` | string | Stable de-identified patient key. |
| `timestamp` | datetime | Time of vital sign observation. |
| `temperature` | float | Celsius. Convert Fahrenheit before ingestion. |
| `heart_rate` | float | Beats per minute. |
| `resp_rate` | float | Breaths per minute. |
| `sbp` | float | Systolic blood pressure, mmHg. |
| `spo2` | float | Peripheral oxygen saturation, percent. |
| `gcs` | float | Glasgow Coma Scale, 3-15. |

The feature pipeline can run with at least 3 of the 6 vitals, but the study
should measure missingness explicitly at every site.

## Recommended Fields

| Column | Type | Notes |
| --- | --- | --- |
| `encounter_id` | string | Separates multiple encounters for the same patient. |
| `site_id` | string | Hospital/site identifier for split and fairness analysis. |
| `age_years` | float | Needed for pediatric threshold features. |
| `sex` | string | Optional subgroup/fairness field. |
| `department` | string | ED, triage, ward, outpatient, etc. |
| `disposition` | string | Discharged, admitted, transferred, died. |
| `antibiotic_time` | datetime | Candidate suspected-infection proxy. |
| `culture_order_time` | datetime | Candidate suspected-infection proxy where available. |
| `clinician_recognition_time` | datetime | Time sepsis was clinically recognized. |
| `adjudicated_sepsis` | int | Final label after clinical review. |

## Site Feasibility Questions

- Are vitals paper-based or electronic?
- Are repeat vitals timestamped?
- Is GCS recorded, or is AVPU used instead?
- Is SpO2 available at triage?
- Are antibiotics, cultures, transfer, discharge, and death recorded?
- Can de-identified data leave the site?
- Must data remain in-country?
- Which local REC/IRB and national clearance bodies apply?

## Leakage Rules

- Never compute features using readings after the prediction time.
- Keep `include_episode_agg=False` for time-indexed prediction.
- If episode aggregates are used for a one-row-per-encounter baseline, compute
  them only from the allowed observation window.
- Split by site and time where possible; do not let later records from the same
  clinical episode leak into training.

## First Extract QA

Run `summarize_vitals_quality()` on any sample file from a potential site before
modeling. The report should be attached to the Phase 0 site feasibility memo.

Minimum fields to inspect:

- `n_rows`
- `n_patients`
- per-vital `missing_rate`
- per-vital `implausible_count`
- `rows_with_all_six_vitals_rate`
- `rows_with_all_six_plausible_vitals_rate`
