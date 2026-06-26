"""
sepsis_vitals.train
~~~~~~~~~~~~~~~~~~~
One-command training pipeline for the sepsis identification model.

Usage:
    python -m sepsis_vitals.train
    python -m sepsis_vitals.train --patients 20000 --output models/v2

Generates synthetic clinical data, trains 5 model types with hyperparameter
optimization, evaluates on held-out test set, computes SHAP explanations,
and saves the best model with a full evaluation report.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train sepsis identification model"
    )
    parser.add_argument(
        "--patients", type=int, default=10000,
        help="Number of synthetic patients to generate (default: 10000)"
    )
    parser.add_argument(
        "--prevalence", type=float, default=0.15,
        help="Sepsis prevalence in dataset (default: 0.15)"
    )
    parser.add_argument(
        "--output", type=str, default="models",
        help="Output directory for model artifacts (default: models)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP computation (faster training)"
    )
    parser.add_argument(
        "--data-source", type=str, default="synthetic",
        choices=["synthetic", "mimic-demo"],
        help="Data source: 'synthetic' (default) or 'mimic-demo'"
    )
    parser.add_argument(
        "--max-patients", type=int, default=None,
        help="Limit number of patients (for faster iteration)"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use ensemble training (requires >=500 patients)"
    )

    opts = parser.parse_args(args)

    print("=" * 70)
    print("  SEPSIS VITALS — AUTONOMOUS MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"\n  Configuration:")
    print(f"    Data source:      {opts.data_source}")
    print(f"    Patients:         {opts.max_patients or opts.patients:,}")
    print(f"    Sepsis prevalence:{opts.prevalence:.0%}")
    print(f"    CV folds:         {opts.cv_folds}")
    print(f"    Output:           {opts.output}")
    print(f"    Seed:             {opts.seed}")

    total_start = time.time()

    # ── Step 1: Load data ────────────────────────────────────────────────

    if opts.data_source == "mimic-demo":
        print("\n" + "─" * 70)
        print("  STEP 1: Loading MIMIC-IV Demo data with Sepsis-3 labels")
        print("─" * 70)

        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        full_df = loader.build_training_dataset(max_patients=opts.max_patients)

        n_patients = full_df["patient_id"].nunique()
        print(f"\n  Loaded {len(full_df)} observations from {n_patients} patients")
        print(f"  Sepsis prevalence: {full_df['sepsis_label'].mean():.1%}")

        if "label_source" in full_df.columns:
            print(f"  Label sources: {full_df['label_source'].value_counts().to_dict()}")

        # For small datasets, don't split -- use LOPOCV instead
        if n_patients < 200:
            print(f"\n  Small dataset ({n_patients} patients) -> using LOPOCV evaluation")
            use_lopocv = True
            train_df = full_df
            val_df = full_df
            test_df = full_df
        else:
            use_lopocv = False
            patient_ids = full_df["patient_id"].unique()
            np.random.seed(opts.seed)
            np.random.shuffle(patient_ids)
            n_train = int(0.7 * len(patient_ids))
            n_val = int(0.15 * len(patient_ids))

            train_pids = set(patient_ids[:n_train])
            val_pids = set(patient_ids[n_train:n_train + n_val])
            test_pids = set(patient_ids[n_train + n_val:])

            train_df = full_df[full_df["patient_id"].isin(train_pids)]
            val_df = full_df[full_df["patient_id"].isin(val_pids)]
            test_df = full_df[full_df["patient_id"].isin(test_pids)]
    else:
        use_lopocv = False
        print("\n" + "─" * 70)
        print("  STEP 1: Generating clinically-grounded synthetic data")
        print("─" * 70)

        from sepsis_vitals.ml.synthetic_data import generate_train_val_test

        train_df, val_df, test_df = generate_train_val_test(
            n_patients=opts.patients,
            sepsis_prevalence=opts.prevalence,
            seed=opts.seed,
        )

    print(f"\n  Train: {len(train_df):,} observations ({train_df['patient_id'].nunique():,} patients)")
    print(f"  Val:   {len(val_df):,} observations ({val_df['patient_id'].nunique():,} patients)")
    print(f"  Test:  {len(test_df):,} observations ({test_df['patient_id'].nunique():,} patients)")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df):,} observations")

    sepsis_rate_train = train_df["sepsis_label"].mean()
    sepsis_rate_test = test_df["sepsis_label"].mean()
    print(f"\n  Sepsis rate — Train: {sepsis_rate_train:.1%} | Test: {sepsis_rate_test:.1%}")

    # ── Step 2: Feature engineering ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 2: Feature engineering")
    print("─" * 70)

    from sepsis_vitals.ml.trainer import prepare_features

    train_features, feature_cols = prepare_features(train_df)
    val_features, _ = prepare_features(val_df)
    test_features, _ = prepare_features(test_df)

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Feature list: {', '.join(feature_cols[:10])}...")

    # Extract X, y arrays
    X_train = train_features[feature_cols].values.astype(np.float64)
    y_train = train_features["sepsis_label"].values.astype(int)
    X_val = val_features[feature_cols].values.astype(np.float64)
    y_val = val_features["sepsis_label"].values.astype(int)
    X_test = test_features[feature_cols].values.astype(np.float64)
    y_test = test_features["sepsis_label"].values.astype(int)

    # Handle NaN values
    nan_mask_train = np.isnan(X_train)
    nan_mask_val = np.isnan(X_val)
    nan_mask_test = np.isnan(X_test)

    # Compute column medians from training set for imputation
    col_medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[nan_mask_train[:, j], j] = col_medians[j]
        X_val[nan_mask_val[:, j], j] = col_medians[j]
        X_test[nan_mask_test[:, j], j] = col_medians[j]

    # Save imputation medians for inference
    imputation_medians = dict(zip(feature_cols, col_medians))

    print(f"\n  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  NaN imputed with training medians")

    # ── Step 3: Train all models ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 3: Training 5 models with hyperparameter optimization")
    print("─" * 70)

    from sepsis_vitals.ml.trainer import (
        calibrate_model,
        compute_shap_values,
        generate_evaluation_report,
        save_model,
        select_best_model,
        train_all_models,
    )

    results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_cols,
        n_cv_folds=opts.cv_folds,
    )

    # ── Step 4: Select and calibrate best model ──────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 4: Selecting and calibrating best model")
    print("─" * 70)

    best = select_best_model(results)
    print(f"\n  Best model: {best.name}")
    print(f"  Val AUROC:  {best.metrics['val_auroc']:.4f}")

    # Calibrate
    if best.scaler is not None:
        X_val_scaled = best.scaler.transform(X_val)
    else:
        X_val_scaled = X_val

    best = calibrate_model(best, X_val_scaled, y_val)
    print(f"  Calibrated: Yes (Platt scaling)")

    # ── Dual operating points ───────────────────────────────────────────
    from sepsis_vitals.ml.trainer import compute_dual_thresholds

    y_prob_val = best.model.predict_proba(
        best.scaler.transform(X_val) if best.scaler else X_val
    )[:, 1]
    dual_thresholds = compute_dual_thresholds(y_val, y_prob_val)

    print(f"\n  Dual operating points:")
    print(f"    Continuous (99% spec): threshold={dual_thresholds['continuous']['threshold']:.3f}, "
          f"sensitivity={dual_thresholds['continuous']['sensitivity']:.3f}")
    print(f"    On-demand  (95% spec): threshold={dual_thresholds['on_demand']['threshold']:.3f}, "
          f"sensitivity={dual_thresholds['on_demand']['sensitivity']:.3f}")

    # ── Step 5: SHAP explanations ────────────────────────────────────────
    shap_importance = {}
    if not opts.skip_shap:
        print("\n" + "─" * 70)
        print("  STEP 5: Computing SHAP explanations")
        print("─" * 70)

        try:
            shap_values, shap_importance = compute_shap_values(
                model=best.model,
                X=X_test if best.scaler is None else best.scaler.transform(X_test),
                feature_names=feature_cols,
                max_samples=min(1000, len(X_test)),
            )
            print(f"\n  SHAP values computed for {min(1000, len(X_test))} samples")
            print(f"\n  Top 5 SHAP features:")
            for i, (feat, imp) in enumerate(list(shap_importance.items())[:5]):
                print(f"    {i+1}. {feat}: {imp:.4f}")
        except Exception as e:
            print(f"\n  SHAP computation failed: {e}")
            print("  Using feature importance from model instead")
            shap_importance = best.feature_importance or {}
    else:
        print("\n  Skipping SHAP (--skip-shap)")
        shap_importance = best.feature_importance or {}

    # ── Step 6: Generate evaluation report ───────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 6: Generating evaluation report")
    print("─" * 70)

    if best.scaler is not None:
        X_test_final = best.scaler.transform(X_test)
    else:
        X_test_final = X_test

    report_path = generate_evaluation_report(
        results=results,
        best=best,
        X_test=X_test_final,
        y_test=y_test,
        feature_names=feature_cols,
        shap_importance=shap_importance,
        output_dir=opts.output,
    )
    print(f"\n  Report saved to {report_path}")

    # ── Step 7: Save model ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 7: Saving model artifacts")
    print("─" * 70)

    model_path = save_model(
        result=best,
        feature_names=feature_cols,
        output_dir=opts.output,
    )

    # Save imputation medians
    import json
    medians_file = f"{opts.output}/imputation_medians.json"
    with open(medians_file, "w") as f:
        json.dump(
            {k: float(v) if not np.isnan(v) else 0.0 for k, v in imputation_medians.items()},
            f, indent=2,
        )
    print(f"    - imputation_medians.json")

    # Add dual thresholds to model metadata
    metadata_file = f"{opts.output}/model_metadata.json"
    with open(metadata_file) as f:
        meta = json.load(f)
    meta["dual_thresholds"] = dual_thresholds
    with open(metadata_file, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"    - dual_thresholds added to model_metadata.json")

    # ── LOPOCV evaluation (for small MIMIC datasets) ────────────────────
    if opts.data_source == "mimic-demo" and use_lopocv:
        print("\n" + "─" * 70)
        print("  LOPOCV EVALUATION")
        print("─" * 70)

        from sepsis_vitals.ml.trainer import lopocv_evaluate

        lopocv_results = lopocv_evaluate(
            train_features, feature_cols,
            model_type="gradient_boosting" if "GradientBoosting" in best.name else "logistic",
        )

        print(f"    LOPOCV AUROC: {lopocv_results.get('auroc', 'N/A')}")
        print(f"    Patients evaluated: {lopocv_results['n_evaluated']}/{lopocv_results['n_patients']}")

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Best Model:       {best.name}")
    print(f"  Val AUROC:        {best.metrics['val_auroc']:.4f}")
    print(f"  Val Sensitivity:  {best.metrics['val_recall']:.4f}")
    print(f"  Val Specificity:  {best.metrics['val_specificity']:.4f}")
    print(f"  Calibrated:       Yes")
    print(f"  Total Time:       {total_time:.1f}s")
    print(f"  Output Directory: {opts.output}/")
    print(f"\n  To use the model:")
    print(f"    from sepsis_vitals.ml.predictor import SepsisPredictor")
    print(f"    predictor = SepsisPredictor('{opts.output}')")
    print(f"    result = predictor.predict(vitals)")
    print(f"\n" + "=" * 70)

    return best


if __name__ == "__main__":
    main()
