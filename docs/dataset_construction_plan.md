# Dataset Construction Plan

## Objective

Build a de-identified, adjudicated vitals-only sepsis dataset from partner
district hospitals, then combine it with public ICU data filtered down to the
same six-vital inference constraint.

## Target Sample

- 10,000+ retrospective patient episodes.
- Gate: at least 8,000 labeled episodes.
- Gate: core vital missingness below 30% after site-specific workflow review.
- Stratify by hospital site, department, age group, and time.

## Core Fields

- Temperature.
- Heart rate.
- Respiratory rate.
- Systolic blood pressure.
- SpO2.
- GCS or a documented mapping from AVPU to mental status.
- Age.
- Sex.
- Site.
- Encounter and observation timestamps.

## Labeling

Use Sepsis-3 as the conceptual standard, but document when full SOFA is not
available. Ambiguous cases should be clinician-adjudicated.

Recommended adjudication queue:

- Suspected infection.
- Antibiotic order or administration.
- Culture order where available.
- Hypotension, altered mentation, hypoxia, renal dysfunction where available.
- Disposition: death, transfer, ICU referral, admission, discharge.

## De-Identification

- Remove names, phone numbers, direct IDs, bed numbers, and raw addresses.
- Keep a site-owned linkage key at the hospital.
- Date-shift if required by local governance.
- Keep country/site codes coarse enough to avoid re-identification.

## Quality Reports

Run `summarize_vitals_quality()` on every extract and attach:

- Missingness per vital.
- Implausible range counts.
- Complete six-vital row rate.
- Complete plausible six-vital row rate.
- Number of patients and rows.
