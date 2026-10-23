#!/usr/bin/env python3
"""
retrain.py — Unified retraining pipeline for sepsis identification model.

Supports two data sources:
  1. Hand-authored heuristic synthetic data — for development and baseline metrics
  2. MIMIC-IV (real ICU data)      — for clinical validation and pitch deck

Usage:
    # Synthetic baseline (default)
    python retrain.py --source synthetic --patients 50000

    # MIMIC-IV clinical validation
    python retrain.py --source mimic --mimic-path /data/mimic-iv/

    # Quick iteration
    python retrain.py --source mimic --mimic-path /data/mimic-iv/ --max-patients 5000 --skip-shap
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Version tracking
MODEL_VERSION = "2.0.0"
PIPELINE_VERSION = "2.0.0"


def _banner(text: str, char: str = "─") -> None:
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}")


def load_synthetic_data(
    n_patients: int,
    prevalence: float,
    seed: int,
    **generator_options: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate synthetic training data (legacy generator unless options are given)."""
    from sepsis_vitals.ml.synthetic_data import generate_train_val_test

    train_df, val_df, test_df = generate_train_val_test(
        n_patients=n_patients,
        sepsis_prevalence=prevalence,
        seed=seed,
        **generator_options,
    )

    provenance = {
        "source": "synthetic",
        "generator": "Hand-authored heuristic synthetic generator (sepsis_vitals.ml.synthetic_data)",
        "n_patients": n_patients,
        "sepsis_prevalence": prevalence,
        "seed": seed,
        "generator_options": generator_options,
        "clinical_validation": "NONE — synthetic data only. Requires MIMIC-IV or institutional EHR validation.",
        "regulatory_note": "NOT suitable for clinical claims. Research use only.",
    }

    return train_df, val_df, test_df, provenance


def load_mimic_data(
    mimic_path: str,
    max_patients: Optional[int] = None,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and split MIMIC-IV data for training."""
    from sepsis_vitals.ml.mimic_loader import MIMICLoader

    loader = MIMICLoader(mimic_path)
    df = loader.build_training_dataset(max_patients=max_patients)

    if len(df) == 0:
        raise ValueError("No data loaded from MIMIC-IV. Check file paths and data integrity.")

    # Patient-level split to prevent data leakage
    patient_ids = df["patient_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    n_test = int(len(patient_ids) * test_fraction)
    n_val = int(len(patient_ids) * val_fraction)

    test_ids = set(patient_ids[:n_test])
    val_ids = set(patient_ids[n_test:n_test + n_val])
    train_ids = set(patient_ids[n_test + n_val:])

    train_df = df[df["patient_id"].isin(train_ids)].copy()
    val_df = df[df["patient_id"].isin(val_ids)].copy()
    test_df = df[df["patient_id"].isin(test_ids)].copy()

    sepsis_prev = df["sepsis_label"].mean()

    provenance = {
        "source": "MIMIC-IV",
        "path": mimic_path,
        "total_stays": len(df),
        "total_patients": len(patient_ids),
        "sepsis_prevalence": round(float(sepsis_prev), 4),
        "train_patients": len(train_ids),
        "val_patients": len(val_ids),
        "test_patients": len(test_ids),
        "clinical_validation": "Retrospective validation on MIMIC-IV ICU cohort (Beth Israel Deaconess Medical Center).",
        "regulatory_note": "Retrospective single-center study. Multi-center prospective validation required for regulatory submission.",
    }

    return train_df, val_df, test_df, provenance


def run_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    provenance: Dict[str, Any],
    output_dir: str = "models",
    cv_folds: int = 5,
    skip_shap: bool = False,
    feature_set: str = "full",
) -> Dict[str, Any]:
    """Execute the full training pipeline."""
    from sepsis_vitals.ml.trainer import (
        calibrate_model,
        compute_shap_values,
        generate_evaluation_report,
        prepare_features,
        save_model,
        select_best_model,
        train_all_models,
    )

    total_start = time.time()
    source = provenance["source"]
    run_provenance = {
        **provenance,
        "feature_set": feature_set,
        "lab_features_available": feature_set == "full",
    }

    print("=" * 70)
    print("  SEPSIS VITALS — MODEL RETRAINING PIPELINE v" + PIPELINE_VERSION)
    print("=" * 70)
    print(f"\n  Data source:        {source}")
    if source == "MIMIC-IV":
        print(f"  MIMIC path:         {provenance.get('path', 'N/A')}")
        print(f"  Total stays:        {provenance.get('total_stays', 'N/A'):,}")
    else:
        print(f"  Patients:           {provenance.get('n_patients', 'N/A'):,}")
    print(f"  Model version:      {MODEL_VERSION}")
    print(f"  Feature set:        {feature_set}")
    print(f"  Output:             {output_dir}")

    # ── Step 1: Data summary ────────────────────────────────────────────
    _banner("STEP 1: Data summary")

    print(f"\n  Train: {len(train_df):,} observations ({train_df['patient_id'].nunique():,} patients)")
    print(f"  Val:   {len(val_df):,} observations ({val_df['patient_id'].nunique():,} patients)")
    print(f"  Test:  {len(test_df):,} observations ({test_df['patient_id'].nunique():,} patients)")

    sepsis_train = train_df["sepsis_label"].mean()
    sepsis_test = test_df["sepsis_label"].mean()
    print(f"\n  Sepsis rate — Train: {sepsis_train:.1%} | Test: {sepsis_test:.1%}")

    # ── Step 2: Feature engineering ─────────────────────────────────────
    _banner("STEP 2: Feature engineering")

    train_features, feature_cols = prepare_features(train_df, feature_set=feature_set)
    val_features, _ = prepare_features(val_df, feature_set=feature_set)
    test_features, _ = prepare_features(test_df, feature_set=feature_set)

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Sample:   {', '.join(feature_cols[:8])}...")

    X_train = train_features[feature_cols].values.astype(np.float64)
    y_train = train_features["sepsis_label"].values.astype(int)
    X_val = val_features[feature_cols].values.astype(np.float64)
    y_val = val_features["sepsis_label"].values.astype(int)
    X_test = test_features[feature_cols].values.astype(np.float64)
    y_test = test_features["sepsis_label"].values.astype(int)

    # NaN imputation with training medians
    col_medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        nan_mask = np.isnan(X_train[:, j])
        X_train[nan_mask, j] = col_medians[j]
        nan_mask = np.isnan(X_val[:, j])
        X_val[nan_mask, j] = col_medians[j]
        nan_mask = np.isnan(X_test[:, j])
        X_test[nan_mask, j] = col_medians[j]

    # Handle any remaining NaN from entirely-NaN columns
    col_medians = np.nan_to_num(col_medians, nan=0.0)

    imputation_medians = {k: float(v) for k, v in zip(feature_cols, col_medians)}

    print(f"\n  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # ── Step 3: Train all models ────────────────────────────────────────
    _banner("STEP 3: Training models with hyperparameter optimization")

    results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_cols,
        n_cv_folds=cv_folds,
    )

    # ── Step 4: Select and calibrate ────────────────────────────────────
    _banner("STEP 4: Selecting and calibrating best model")

    best = select_best_model(results)
    print(f"\n  Best model: {best.name}")
    print(f"  Val AUROC:  {best.metrics['val_auroc']:.4f}")

    X_val_scaled = best.scaler.transform(X_val) if best.scaler else X_val
    best = calibrate_model(best, X_val_scaled, y_val)
    print("  Calibrated: Yes (Platt scaling)")

    # ── Step 5: SHAP explanations ───────────────────────────────────────
    shap_importance = {}
    if not skip_shap:
        _banner("STEP 5: Computing SHAP explanations")
        try:
            X_shap = best.scaler.transform(X_test) if best.scaler else X_test
            _, shap_importance = compute_shap_values(
                model=best.model,
                X=X_shap,
                feature_names=feature_cols,
                max_samples=min(1000, len(X_test)),
            )
            print("\n  Top 5 SHAP features:")
            for i, (feat, imp) in enumerate(list(shap_importance.items())[:5]):
                print(f"    {i+1}. {feat}: {imp:.4f}")
        except Exception as e:
            print(f"\n  SHAP failed: {e} — using model feature importance")
            shap_importance = best.feature_importance or {}
    else:
        print("\n  Skipping SHAP (--skip-shap)")
        shap_importance = best.feature_importance or {}

    # ── Step 6: Evaluation report ───────────────────────────────────────
    _banner("STEP 6: Generating evaluation report")

    # generate_evaluation_report applies best.scaler itself: pass unscaled
    # features (passing scaled ones scaled them twice for scaler-based models).
    report_path = generate_evaluation_report(
        results=results,
        best=best,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_cols,
        shap_importance=shap_importance,
        output_dir=output_dir,
    )

    # Load the generated report to augment it
    with open(report_path) as f:
        report = json.load(f)

    # Inject provenance and version into report
    report["data_provenance"] = run_provenance
    report["feature_set"] = feature_set
    report["feature_names"] = feature_cols
    report["model_version"] = MODEL_VERSION
    report["pipeline_version"] = PIPELINE_VERSION

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report saved to {report_path}")

    # ── Step 7: Save model artifacts ────────────────────────────────────
    _banner("STEP 7: Saving model artifacts")

    save_model(result=best, feature_names=feature_cols, output_dir=output_dir)

    # Save imputation medians
    medians_file = Path(output_dir) / "imputation_medians.json"
    with open(medians_file, "w") as f:
        json.dump(imputation_medians, f, indent=2)

    # ── Step 8: Update model_metadata.json with provenance ──────────────
    _banner("STEP 8: Updating model metadata with data provenance")

    metadata_file = Path(output_dir) / "model_metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Inject version and provenance
    metadata["version"] = MODEL_VERSION
    metadata["data_provenance"] = run_provenance
    metadata["feature_set"] = feature_set
    metadata["retrained_at"] = datetime.now(timezone.utc).isoformat()
    metadata["pipeline_version"] = PIPELINE_VERSION

    # Update model card
    card = metadata.get("model_card", {})
    card["version"] = MODEL_VERSION

    if source == "MIMIC-IV":
        card["training_data"] = (
            f"MIMIC-IV v2.2+ ({provenance.get('total_stays', 'N/A')} ICU stays, "
            f"{provenance.get('total_patients', 'N/A')} patients, "
            f"{provenance.get('sepsis_prevalence', 0):.1%} sepsis prevalence). "
            f"Beth Israel Deaconess Medical Center. "
            f"Retrospective cohort with Sepsis-3 labels derived from ICD-10 codes."
        )
        card["regulatory_status"] = (
            "Research use only. Retrospective single-center validation completed. "
            "Not FDA-cleared, not CE-marked. Multi-center prospective validation "
            "and 510(k)/De Novo submission required before clinical use."
        )
        card["limitations"] = (
            "Single-center retrospective study (BIDMC). Performance may not generalize "
            "to other populations, EHR systems, or care settings. Sepsis labels derived "
            "from ICD-10 codes (not prospective clinical adjudication). "
            "Fairness audit across demographic subgroups required before deployment."
        )
    else:
        card["training_data"] = (
            f"Synthetic dataset ({provenance.get('n_patients', 'N/A'):,} patients) "
            f"generated from hand-authored distributions partially informed by public "
            f"clinical reference ranges. "
            f"NOT trained on real patient data. "
            f"Clinical validation on MIMIC-IV or institutional EHR data is REQUIRED."
        )
        card["regulatory_status"] = (
            "Research use only. Not FDA-cleared, not CE-marked. "
            "Must complete 510(k) or De Novo pathway before clinical use."
        )
        card["limitations"] = (
            "Trained on synthetic data only. Performance metrics are pre-validation "
            "estimates and may not generalize to real clinical populations. "
            "MIMIC-IV or institutional EHR validation REQUIRED. "
            "Fairness audit across demographic subgroups not yet completed."
        )

    metadata["model_card"] = card

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  Metadata updated: {metadata_file}")

    # Manifest last, once every artifact is final (the API refuses models without one).
    from sepsis_vitals.ml.artifacts import write_manifest
    source = metadata.get("data_provenance", {}).get("source", "unknown")
    write_manifest(
        Path(output_dir),
        "synthetic-development" if source == "synthetic" else "unvalidated",
    )
    print(f"  Manifest written: {Path(output_dir) / 'manifest.json'}")

    # ── Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - total_start

    test_metrics = report.get("test_set_performance", {})

    print("\n" + "=" * 70)
    print("  RETRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Data Source:      {source}")
    print(f"  Feature Set:      {feature_set}")
    print(f"  Model Version:    {MODEL_VERSION}")
    print(f"  Best Model:       {best.name}")
    print(f"  Test AUROC:       {test_metrics.get('test_auroc', 0):.4f}")
    print(f"  Test Sensitivity: {test_metrics.get('test_recall', 0):.4f}")
    print(f"  Test Specificity: {test_metrics.get('test_specificity', 0):.4f}")
    print(f"  Test PPV:         {test_metrics.get('test_ppv', 0):.4f}")
    print(f"  Test NPV:         {test_metrics.get('test_npv', 0):.4f}")
    print(f"  Calibration ECE:  {report.get('calibration', {}).get('ece', 0):.4f}")
    print(f"  Total Time:       {total_time:.1f}s")
    print(f"  Output:           {output_dir}/")
    print("\n" + "=" * 70)

    return {
        "best_model": best.name,
        "test_metrics": test_metrics,
        "report": report,
        "provenance": run_provenance,
        "feature_set": feature_set,
        "feature_cols": feature_cols,
        "shap_importance": shap_importance,
        "total_time": total_time,
        # Held-out predictions of the final (calibrated) model, row-aligned with
        # test_df, for paired uncertainty estimates (sepsis_vitals.ml.uncertainty).
        "test_predictions": {
            "patient_id": test_features["patient_id"].tolist(),
            "y_true": y_test.tolist(),
            "y_prob": best.model.predict_proba(
                best.scaler.transform(X_test) if best.scaler else X_test
            )[:, 1].tolist(),
        },
    }


def archive_existing_model(output_dir: Path) -> Path:
    """Copy the model in *output_dir* (per its manifest) to archive/<sha>/.

    Restoring means copying the archived files back; the manifest pins every
    file's SHA-256, so a partial or mixed restore is refused at load time.
    """
    import shutil

    manifest = json.loads((output_dir / "manifest.json").read_text())
    sha = manifest["artifacts"]["sepsis_model.joblib"]["sha256"][:12]
    dest = output_dir / "archive" / sha
    dest.mkdir(parents=True, exist_ok=True)
    for name in list(manifest["artifacts"]) + ["manifest.json"]:
        if (output_dir / name).exists() and not (dest / name).exists():
            shutil.copy2(output_dir / name, dest / name)
    return dest


def main(argv: list | None = None):
    parser = argparse.ArgumentParser(
        description="Retrain sepsis identification model (synthetic or MIMIC-IV)"
    )
    parser.add_argument(
        "--source", choices=["synthetic", "mimic"], default="synthetic",
        help="Data source: 'synthetic' or 'mimic' (default: synthetic)"
    )
    parser.add_argument(
        "--mimic-path", type=str, default=None,
        help="Path to MIMIC-IV root directory (required if --source mimic)"
    )
    parser.add_argument(
        "--patients", type=int, default=50000,
        help="Number of synthetic patients (default: 50000)"
    )
    parser.add_argument(
        "--max-patients", type=int, default=None,
        help="Max MIMIC patients to load (default: all)"
    )
    parser.add_argument(
        "--prevalence", type=float, default=0.15,
        help="Sepsis prevalence for synthetic data (default: 0.15)"
    )
    parser.add_argument(
        "--output", type=str, default="models",
        help="Output directory (default: models)"
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
        help="Skip SHAP computation (faster)"
    )
    parser.add_argument(
        "--feature-set", choices=["full", "no_labs"], default="full",
        help="Feature set: 'full' or 'no_labs' ablation (default: full)"
    )
    parser.add_argument(
        "--generator-profile", choices=["legacy", "decoupled"], default="legacy",
        help="Synthetic generator profile (sepsis_vitals.ml.profile_evaluation.PROFILES)",
    )
    parser.add_argument(
        "--label-mode", choices=["current_state", "onset_within_horizon"], default="current_state",
        help="Training label; onset_within_horizon needs --horizon-hours (a clinical choice, no default)",
    )
    parser.add_argument("--horizon-hours", type=float, default=None)
    parser.add_argument(
        "--replace", action="store_true",
        help="Allow replacing a model already in --output (it is archived to <output>/archive/<sha>/ first)",
    )

    opts = parser.parse_args(argv)

    if opts.label_mode == "onset_within_horizon" and opts.horizon_hours is None:
        parser.error("--label-mode onset_within_horizon requires --horizon-hours (a clinical decision)")
    output = Path(opts.output)
    if (output / "manifest.json").exists():
        if not opts.replace:
            parser.error(f"{output} already holds a model (manifest.json); "
                         "pass --replace to archive it and write a new one")
        print(f"Archived the current model to {archive_existing_model(output)}")

    if opts.source == "mimic":
        if opts.mimic_path is None:
            parser.error("--mimic-path is required when --source is 'mimic'")
        train_df, val_df, test_df, provenance = load_mimic_data(
            mimic_path=opts.mimic_path,
            max_patients=opts.max_patients,
            seed=opts.seed,
        )
    else:
        from sepsis_vitals.ml.profile_evaluation import PROFILES

        options = dict(PROFILES[opts.generator_profile])
        if opts.label_mode != "current_state":
            options.update(label_mode=opts.label_mode, horizon_hours=opts.horizon_hours)
        train_df, val_df, test_df, provenance = load_synthetic_data(
            n_patients=opts.patients,
            prevalence=opts.prevalence,
            seed=opts.seed,
            **options,
        )

    result = run_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        provenance=provenance,
        output_dir=opts.output,
        cv_folds=opts.cv_folds,
        skip_shap=opts.skip_shap,
        feature_set=opts.feature_set,
    )

    return result


if __name__ == "__main__":
    main()
