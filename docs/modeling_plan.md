# Modeling Plan

## Phase 1 Baselines

1. qSOFA.
2. Partial SIRS without WBC.
3. Logistic regression on six raw vitals.
4. Logistic regression on engineered features.
5. XGBoost or LightGBM on engineered features.

## Primary Model

The first serious model should be gradient boosting with:

- Native missing-value handling.
- Probability calibration.
- Site-aware validation.
- SHAP explanations for the top alert driver.
- Fixed clinical operating thresholds chosen on validation data only.

## Primary Endpoint

AUROC for sepsis prediction 4 hours before clinical recognition.

## Secondary Endpoints

- AUPRC.
- Sensitivity at matched specificity against qSOFA.
- Specificity at site-agreed alert burden.
- Calibration slope/intercept and Brier score.
- Alert rate per 100 encounters.
- Subgroup performance by site, age group, sex, and department.

## Validation Splits

Preferred split hierarchy:

1. External site holdout.
2. Temporal holdout within site.
3. Patient-level random split only for early prototyping.

## Explainability Output

For each alert, the deployed model should expose:

- Risk tier: low, medium, high.
- Top contributing feature.
- Last reading time.
- Missing high-signal inputs, if any.

No raw probability should be shown in the nurse-facing V1 workflow.
