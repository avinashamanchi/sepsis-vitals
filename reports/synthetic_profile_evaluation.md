# Synthetic generator profiles: evaluation

**Engineering evidence only.** These numbers describe a hand-authored generator, not
patients, and do not support any performance claim. The prediction horizon below
(12 h) only demonstrates the label mechanics; choosing a horizon is a
clinical decision for the study team.

4,000 generated patients per profile, seed 7. Split: patient-level and temporal
(earliest 60% of admissions train, next 15% validate, latest 25% test). Operating point:
specificity 0.90 on validation (an engineering convention, not a clinical threshold).

| Profile | Label | AUROC (95% CI) | AUPRC | Brier | Cal. slope | Cal. intercept | Sens @ op. | PPV @ op. | AUROC no labs (95% CI) | AUROC demographics only (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | current_state | 0.911 (0.901-0.919) | 0.761 | 0.089 | 0.99 | 0.04 | 0.761 | 0.671 | 0.821 (0.800-0.841) | 0.730 (0.703-0.756) |
| decoupled | current_state | 0.879 (0.865-0.891) | 0.633 | 0.075 | 0.97 | 0.07 | 0.700 | 0.545 | 0.631 (0.615-0.649) | 0.507 (0.462-0.548) |
| decoupled | onset within 12 h | 0.734 (0.711-0.759) | 0.198 | 0.040 | 0.97 | -0.04 | 0.412 | 0.166 | 0.574 (0.546-0.603) | 0.490 (0.453-0.528) |

## Subgroup AUROC (test split)

| Profile | age 18-44 | age 45-64 | age 65+ | sex F | sex M |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy (current) | 0.801 | 0.882 | 0.864 | 0.912 | 0.910 |
| decoupled (current) | 0.886 | 0.878 | 0.874 | 0.882 | 0.876 |
| decoupled (horizon) | 0.729 | 0.737 | 0.735 | 0.716 | 0.750 |

## Reading the table

- Intervals: paired patient-level bootstrap (percentile, 95%, 1000 replicates, seed 7); the model, its no-labs evaluation and the demographics-only probe are scored on the same resampled test patients. They cover sampling variability of this synthetic test split given the fitted models, not training variability, and are not evidence about patients.
- *Demographics only* measures how much of the label the generator ties to age and
  comorbidity. In the decoupled profile it should fall towards 0.5.
- *Onset within horizon* scores only pre-onset rows: the early-warning question.
  It is expected to be much harder than recognising rows after onset.
- None of these models is a candidate for clinical use. The committed artifact
  (models/manifest.json) is still the legacy synthetic model.

Reproduce: `python scripts/evaluate_synthetic_profiles.py --horizon-hours 12 --patients 4000 --seed 7`
