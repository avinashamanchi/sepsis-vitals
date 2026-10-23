# Model card: vital-signs model 3.0.0 (committed artifact)

**Not approved for clinical use. Validation status: `synthetic-development`.**
The model was trained on synthetic data only. Its role in the prototype is
to exercise the software and the silent-mode workflow. The **published
scores** (UVA, qSOFA, NEWS2-style) are the primary decision-support output
(see `docs/low_resource_deployment.md`).

| Item | Value |
|---|---|
| Artifact | `models/sepsis_model.joblib`, sha256 `f540f9bf349ee164c91ae50383aa2166bedf6f43f1ddd0de389821e05c2c6d4c` (`models/manifest.json`) |
| Algorithm | scikit-learn HistGradientBoostingClassifier with **monotonic constraints** (`trainer.clinical_monotonic_constraints`), sigmoid calibration, and a split-conformal predictor (90% coverage target), all fitted on synthetic data; scikit-learn 1.6.1 |
| Inputs | **Vital signs only:** temperature, heart rate, respiratory rate, SBP, DBP, MAP, SpO₂, GCS, their trends (last 4 observations), age and comorbidity flags. 45 features. No laboratory values. |
| Constraints | The output can only rise with heart rate, respiratory rate, instability and the early-warning scores, and only rise as SBP, DBP, MAP, SpO₂ or GCS fall. Temperature is unconstrained, because fever and hypothermia both matter. |
| Training data | 20,000 synthetic patients, generator profile `physiologic` (onset-anchored septic deterioration; no age or comorbidity shortcut), seed 42, prevalence 0.15 |
| Label | Current state: "this observation is at or after the generated onset". **No prediction horizon** (a clinical decision, R3). |
| Output | A score in [0, 1]. It is **not a probability for patients**. |
| Thresholds | Placeholder bands (`placeholder-bands-v1`), not clinically derived |
| Explanation | Per-patient median-substitution attribution; not causal |

## Performance (synthetic only)

Source: `reports/committed_model_evaluation.md`. Fresh synthetic cohort:
physiologic profile, 4,000 patients, seed 20261015. Intervals are 95%
patient-cluster bootstrap.

- AUROC 0.693 (0.679–0.708); AUPRC 0.316 (0.286–0.344). Age alone: 0.490.
- Calibration on this cohort: slope 0.99, intercept −0.03 (mean output 0.108 against an observed rate of 0.107).
- **Behaviour checks: passed.** Fever, tachycardia (HR > 90), tachypnoea (RR ≥ 22) and hypotension (SBP ≤ 100) each raise the output on average. HR, RR and SBP lower it in at most 4% of observations; fever, which is unconstrained, in 25%.

These numbers describe the generator, not patients. Vital-sign scores on
real data reach similar ranges: the published UVA score had AUC 0.77 for
in-hospital mortality (Moore 2017). That is context, not evidence about
this model.

## Known limitations

1. **Synthetic-only training.** The generator's physiology is written by hand. Real performance is unknown.
2. **Modest discrimination.** The model adds few alerts beyond the rule-based scores; most alerts come from the rules.
3. **First observations.** Unregistered patients are scored without history, so they carry less information.
4. **No uncertainty for patients.** The conformal label set describes coverage on synthetic calibration data, not on patients.

## History

The previous artifact (GradientBoosting 2.0.0, legacy generator, with labs,
sha256 `93dedb4f…`) failed every behaviour check. Its output fell with
fever, tachycardia, tachypnoea and hypotension, because in its training data
septic observations were *less* abnormal in vital signs, and procalcitonin
and age carried the signal (finding N53). It was replaced on 2026-10-22,
is archived locally under `models/archive/93dedb4ff761/`, and can be
recovered from git history (`git show 3475fcf:models/sepsis_model.joblib`).

## Intended use

Software development, demonstrations, and silent-mode data collection with
synthetic or approved data. Any clinical use requires local retraining and
validation (`docs/roadmap_post_merge.md`) and requirements R1–R10.
