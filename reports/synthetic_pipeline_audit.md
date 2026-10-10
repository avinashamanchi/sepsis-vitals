# Synthetic pipeline audit

Synthetic development evidence only. It describes the hand-authored generator and the
committed development model, not performance in any patient population.

Cohort: 6,000 generated patients, seed 7, 13,497 held-out rows.

## 1. Generator artefacts

- 0% of consecutive observations go backwards in time (was 32% before the 2026-10 fix to the timestamp accumulation).
- Rows labelled septic are not more abnormal on average: heart rate 80.54 vs 82.66, SBP 137.91 vs 127.97, lactate 1.42 vs 1.43 (septic vs non-septic rows).
- Septic patients are older (mean age 67.2 vs 48.1); age alone gives patient-level AUROC 0.752.

## 2. Where discrimination comes from

Same learner (HistGradientBoosting), same patient-level split, different feature groups:

| Feature group | Features | Held-out row AUROC |
| --- | ---: | ---: |
| demographics and comorbidities | 6 | 0.723 |
| vitals and scores only | 36 | 0.778 |
| no labs | 45 | 0.833 |
| full | 58 | 0.913 |

## 3. Committed model (`models/sepsis_model.joblib`)

| Check | Value |
| --- | ---: |
| Row AUROC, training-style features (95% CI, patient bootstrap) | 0.902 (0.890-0.912) |
| Row AUROC, each row scored without history | 0.824 |
| Mean predicted risk, with history vs without history | 0.161 vs 0.092 |
| Observed positive row rate | 0.197 |
| Pre-onset rows of future-septic patients vs never-septic rows (early warning) | 0.725 |
| Patient-level AUROC (max risk over stay) | 0.865 |

## Interpretation

- The headline synthetic AUROC measures recognising rows *after* labelled onset. Early-warning
  ability (pre-onset rows) is much weaker and is the quantity a sepsis early-warning claim needs.
- A large share of discrimination is available from demographics, which the generator ties to
  the label by construction. The no-labs ablation inherits this.
- Scoring without history under-predicts risk relative to training. The API now passes
  recorded history for registered patients (tests/test_inference_parity.py); unregistered
  IDs are still scored as first observations.
- None of these numbers should be quoted as product performance.

Reproduce: `python scripts/audit_synthetic_pipeline.py`
