# Model Card Draft

## Model

Vitals-only sepsis risk model for low-resource hospitals.

## Intended Use

Support nurse and clinician escalation decisions in district hospitals after
prospective validation and local ethics approval.

## Not Intended For

- Direct autonomous diagnosis.
- ICU-only deployment without local validation.
- Replacement of clinical judgment.
- Pediatric deployment unless pediatric validation is complete.

## Inputs

- Temperature.
- Heart rate.
- Respiratory rate.
- Systolic blood pressure.
- SpO2.
- GCS.
- Engineered missingness, trajectory, and clinical-score features.

## Known Limitations

- Labels may be modified Sepsis-3 where full SOFA is unavailable.
- Missingness patterns may differ by site.
- MIMIC pre-training does not prove LMIC validity.
- qSOFA may be an imperfect comparator but remains deployable.

## Required Reporting

- AUROC.
- AUPRC.
- Sensitivity/specificity.
- Brier score and calibration curve.
- Site-level performance.
- Adult/pediatric subgroup performance if both are included.
- Alert burden per 100 encounters.
