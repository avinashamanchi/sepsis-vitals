# Model card: GradientBoosting 2.0.0 (committed artifact)

**Not approved for clinical use. Validation status: `synthetic-development`.**

| Item | Value |
|---|---|
| Artifact | `models/sepsis_model.joblib`, sha256 `93dedb4ff761b6502a44c238802e9373260dd84b36aa0659cd22fcbf90412cf1` (`models/manifest.json`) |
| Feature schema | 58 features, sha256 `713c9d45436a26808b97c67a9df3a6da0415c445d44360ef39798b3734e1bad5`; preprocessing (pipeline) version 2.0.0 |
| Algorithm | scikit-learn GradientBoostingClassifier with sigmoid calibration on synthetic validation data; scikit-learn 1.6.1 |
| Training data | 20,000 synthetic patients (legacy generator, seed 42, prevalence 0.15), trained on 2026-06-20. **No real patient data.** |
| Label | Current state: "this observation is at or after the generated sepsis onset". **No prediction horizon.** |
| Inputs | Vital signs, three labs, age, five comorbidity flags, history (last 4 observations); see `docs/inference_contract.md` |
| Output | A score in [0, 1]. It is **not a probability for patients**. |
| Thresholds | Placeholder bands (`placeholder-bands-v1`), not clinically derived |
| Uncertainty | None available (no conformal predictor ships) |
| Explanation | Per-patient median-substitution attribution; not causal |

## Performance (synthetic only)

Source: `reports/committed_model_evaluation.md`. The cohort is a fresh
synthetic one (4,000 patients, seed 20261015); intervals are 95%
patient-cluster bootstrap.

- AUROC 0.909 (0.903–0.914); AUPRC 0.720 (0.706–0.733).
- Age alone: AUROC 0.781 (0.768–0.794).
- Calibration on this cohort: slope 1.00, intercept 0.24.

These numbers describe the generator, not patients.

## Known limitations and failure modes

1. **Synthetic-only training (M1, N53).** In the training generator, septic
   observations differ from non-septic ones mainly in procalcitonin and
   age. Their vital signs are no more abnormal. The model's output
   *falls* with fever in 97% of febrile observations, and with tachycardia
   in 85% of tachycardic ones. The model's behaviour on vital signs is
   therefore not clinically meaningful.
2. **Feature importance concentrates on two inputs.** procalcitonin rolling
   mean 0.39 and age 0.32. Labs are missing in about 25% of synthetic
   observations, and in most real ward observations.
3. **First observations.** Unregistered patients are scored without
   history, which differs from training (reports/synthetic_pipeline_audit.md).
4. **Generator drift.** The generator was changed after training (M4), so
   current synthetic cohorts are close to, not identical with, the training
   distribution.

## Intended use

Software development, demonstration and pipeline testing with synthetic
data. Silent-mode evaluation on real data would first need a model trained
on approved data, and requirements R1–R10 (`docs/pilot/clinical_requirements.md`).

## Replacement

The model is replaced only with separate approval. `retrain.py` refuses to
overwrite it without `--replace` and archives the old one first. A new
artifact must:

- get a new manifest;
- state its validation status;
- pass `check_feature_compatibility`.
