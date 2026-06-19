"""
sepsis_vitals.ml.trainer
~~~~~~~~~~~~~~~~~~~~~~~~
Full ML training pipeline for sepsis identification.

Trains 5 model types, performs hyperparameter optimization via cross-validation,
evaluates on held-out test set, generates SHAP explanations, and saves the best
model with a comprehensive evaluation report.

Models:
1. LightGBM (gradient boosting)
2. XGBoost (gradient boosting)
3. Random Forest
4. Gradient Boosting Classifier (sklearn)
5. Logistic Regression (baseline)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sepsis_vitals.features import CORE_VITALS, build_feature_set
from sepsis_vitals.ml.fairness import calibration_metrics
from sepsis_vitals.model_scaffold import ModelCard

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature columns for ML (constructed by build_feature_set + extras)
# ---------------------------------------------------------------------------

# Base vitals used as features
BASE_VITAL_FEATURES = [
    "temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "gcs", "map",
]

# Engineered features from build_feature_set
DELTA_FEATURES = [f"{v}_delta" for v in CORE_VITALS]
ROLLING_MEAN_FEATURES = [f"{v}_roll_mean" for v in CORE_VITALS]
ROLLING_STD_FEATURES = [f"{v}_roll_std" for v in CORE_VITALS]
MISSING_FEATURES = [f"{v}_missing" for v in CORE_VITALS] + ["n_vitals_missing"]

# Lab value features
LAB_FEATURES = ["lactate", "wbc", "procalcitonin"]
LAB_DERIVED = [
    "lactate_delta", "wbc_delta", "procalcitonin_delta",
    "lactate_roll_mean", "wbc_roll_mean", "procalcitonin_roll_mean",
    "lactate_missing", "wbc_missing", "procalcitonin_missing",
    "n_labs_missing",
]

# Clinical scores as features
SCORE_FEATURES = ["qsofa", "news2_computed", "sirs_computed", "shock_index_computed"]

# Demographic / comorbidity features
DEMOGRAPHIC_FEATURES = ["age_years"]
COMORBIDITY_FEATURES = ["has_hypertension", "has_diabetes", "has_ckd", "has_copd", "has_heart_failure"]

# Temporal features
TEMPORAL_FEATURES = ["obs_gap_min"]


def _compute_additional_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional clinical scores as features."""
    from sepsis_vitals.scores import news2_style, partial_sirs, shock_index

    out = df.copy()

    news2_vals = []
    sirs_vals = []
    si_vals = []

    for _, row in out.iterrows():
        vitals = {}
        for v in BASE_VITAL_FEATURES:
            if v in row and pd.notna(row[v]):
                vitals[v] = row[v]

        news2_vals.append(news2_style(vitals))
        sirs_count, _ = partial_sirs(vitals)
        sirs_vals.append(sirs_count)
        si_vals.append(shock_index(vitals))

    out["news2_computed"] = news2_vals
    out["sirs_computed"] = sirs_vals
    out["shock_index_computed"] = [v if v is not None else np.nan for v in si_vals]

    return out


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Run full feature engineering pipeline and return (features_df, feature_columns).

    This is the canonical feature preparation function used for both training
    and inference.
    """
    # Build feature set (adds deltas, rolling stats, missingness, qsofa)
    features = build_feature_set(
        df,
        patient_col="patient_id",
        time_col="timestamp",
        rolling_window=3,
        score_cols=True,
        age_col="age_years",
    )

    # Add additional clinical scores
    features = _compute_additional_scores(features)

    # Determine which feature columns are available
    feature_cols = []

    for col in BASE_VITAL_FEATURES:
        if col in features.columns:
            feature_cols.append(col)

    for col in DELTA_FEATURES + ROLLING_MEAN_FEATURES + ROLLING_STD_FEATURES:
        if col in features.columns:
            feature_cols.append(col)

    for col in MISSING_FEATURES:
        if col in features.columns:
            # Convert boolean to int
            features[col] = features[col].astype(int)
            feature_cols.append(col)

    for col in LAB_FEATURES + LAB_DERIVED:
        if col in features.columns:
            if features[col].dtype == bool:
                features[col] = features[col].astype(int)
            feature_cols.append(col)

    for col in SCORE_FEATURES:
        if col in features.columns:
            feature_cols.append(col)

    for col in DEMOGRAPHIC_FEATURES + COMORBIDITY_FEATURES:
        if col in features.columns:
            feature_cols.append(col)

    for col in TEMPORAL_FEATURES:
        if col in features.columns:
            feature_cols.append(col)

    # Convert risk_level to numeric if present
    if "risk_level" in features.columns:
        risk_map = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
        features["risk_level_numeric"] = features["risk_level"].map(risk_map).fillna(0)
        feature_cols.append("risk_level_numeric")

    # Remove duplicates while preserving order
    seen = set()
    unique_cols = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    feature_cols = unique_cols

    return features, feature_cols


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Results from training a single model."""
    name: str
    model: Any
    scaler: Optional[StandardScaler]
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    training_time: float
    card: ModelCard
    is_calibrated: bool = False


def _get_model_configs() -> Dict[str, Dict[str, Any]]:
    """Return model configurations with hyperparameter search spaces."""
    configs = {}

    # LightGBM (optional — requires libomp on macOS)
    try:
        import lightgbm as lgb
        configs["LightGBM"] = {
            "model_class": lgb.LGBMClassifier,
            "param_grid": [
                {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                 "num_leaves": 31, "min_child_samples": 20, "subsample": 0.8,
                 "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
                 "random_state": 42, "verbose": -1, "n_jobs": -1},
                {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03,
                 "num_leaves": 63, "min_child_samples": 10, "subsample": 0.9,
                 "colsample_bytree": 0.9, "reg_alpha": 0.05, "reg_lambda": 0.5,
                 "random_state": 42, "verbose": -1, "n_jobs": -1},
                {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.02,
                 "num_leaves": 15, "min_child_samples": 30, "subsample": 0.7,
                 "colsample_bytree": 0.7, "reg_alpha": 0.2, "reg_lambda": 2.0,
                 "random_state": 42, "verbose": -1, "n_jobs": -1},
            ],
            "needs_scaling": False,
            "tree_model": True,
        }
    except (ImportError, OSError):
        print("  [INFO] LightGBM unavailable — skipping")

    # XGBoost
    try:
        import xgboost as xgb
        configs["XGBoost"] = {
            "model_class": xgb.XGBClassifier,
            "param_grid": [
                {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                 "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
                 "reg_lambda": 1.0, "random_state": 42, "eval_metric": "logloss",
                 "use_label_encoder": False, "n_jobs": -1},
                {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03,
                 "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.05,
                 "reg_lambda": 0.5, "random_state": 42, "eval_metric": "logloss",
                 "use_label_encoder": False, "n_jobs": -1},
                {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.02,
                 "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.2,
                 "reg_lambda": 2.0, "random_state": 42, "eval_metric": "logloss",
                 "use_label_encoder": False, "n_jobs": -1},
            ],
            "needs_scaling": False,
            "tree_model": True,
        }
    except Exception:
        print("  [INFO] XGBoost unavailable — skipping")

    configs["RandomForest"] = {
        "model_class": RandomForestClassifier,
        "param_grid": [
            {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5,
             "min_samples_split": 10, "max_features": "sqrt",
             "random_state": 42, "n_jobs": -1},
            {"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 3,
             "min_samples_split": 5, "max_features": "sqrt",
             "random_state": 42, "n_jobs": -1},
        ],
        "needs_scaling": False,
        "tree_model": True,
    }

    configs["GradientBoosting"] = {
        "model_class": GradientBoostingClassifier,
        "param_grid": [
            {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05,
             "min_samples_leaf": 20, "subsample": 0.5, "random_state": 42},
        ],
        "needs_scaling": False,
        "tree_model": True,
    }

    configs["LogisticRegression"] = {
        "model_class": LogisticRegression,
        "param_grid": [
            {"C": 1.0, "max_iter": 2000, "random_state": 42, "solver": "lbfgs"},
            {"C": 0.1, "max_iter": 2000, "random_state": 42, "solver": "lbfgs"},
            {"C": 10.0, "max_iter": 2000, "random_state": 42, "solver": "lbfgs"},
        ],
        "needs_scaling": True,
        "tree_model": False,
    }

    return configs


def _evaluate_model(
    model: Any, X: np.ndarray, y: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    """Compute comprehensive metrics for a model."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}
    pfx = f"{prefix}_" if prefix else ""

    metrics[f"{pfx}auroc"] = float(roc_auc_score(y, y_prob))
    metrics[f"{pfx}auprc"] = float(average_precision_score(y, y_prob))
    metrics[f"{pfx}accuracy"] = float(accuracy_score(y, y_pred))
    metrics[f"{pfx}precision"] = float(precision_score(y, y_pred, zero_division=0))
    metrics[f"{pfx}recall"] = float(recall_score(y, y_pred, zero_division=0))
    metrics[f"{pfx}f1"] = float(f1_score(y, y_pred, zero_division=0))
    metrics[f"{pfx}brier"] = float(brier_score_loss(y, y_prob))

    # Sensitivity at high-specificity operating points
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    for target_spec in [0.90, 0.95]:
        target_fpr = 1 - target_spec
        idx = np.searchsorted(fpr, target_fpr)
        if idx < len(tpr):
            metrics[f"{pfx}sensitivity_at_{int(target_spec*100)}spec"] = float(tpr[idx])
        else:
            metrics[f"{pfx}sensitivity_at_{int(target_spec*100)}spec"] = float(tpr[-1])

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics[f"{pfx}true_positives"] = int(tp)
    metrics[f"{pfx}true_negatives"] = int(tn)
    metrics[f"{pfx}false_positives"] = int(fp)
    metrics[f"{pfx}false_negatives"] = int(fn)
    metrics[f"{pfx}specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics[f"{pfx}npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    metrics[f"{pfx}ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return metrics


def _get_feature_importance(model: Any, feature_names: List[str], is_tree: bool) -> Dict[str, float]:
    """Extract feature importance from a model."""
    if is_tree and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))
    return {}


def train_single_model(
    name: str,
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_cv_folds: int = 5,
) -> ModelResult:
    """Train a single model with hyperparameter selection via cross-validation."""
    print(f"\n  Training {name}...")
    start = time.time()

    model_class = config["model_class"]
    param_grid = config["param_grid"]
    needs_scaling = config["needs_scaling"]
    is_tree = config["tree_model"]

    scaler = None
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()

    if needs_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_scaled)
        X_val_scaled = scaler.transform(X_val_scaled)

    # Cross-validation for hyperparameter selection
    best_score = -1.0
    best_params = param_grid[0]

    if len(param_grid) > 1:
        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

        for params in param_grid:
            cv_scores = []
            for train_idx, val_idx in skf.split(X_train_scaled, y_train):
                X_cv_train = X_train_scaled[train_idx]
                y_cv_train = y_train[train_idx]
                X_cv_val = X_train_scaled[val_idx]
                y_cv_val = y_train[val_idx]

                model = model_class(**params)
                model.fit(X_cv_train, y_cv_train)
                y_prob = model.predict_proba(X_cv_val)[:, 1]

                try:
                    auc = roc_auc_score(y_cv_val, y_prob)
                    cv_scores.append(auc)
                except ValueError:
                    cv_scores.append(0.0)

            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        print(f"    Best CV AUROC: {best_score:.4f}")
    else:
        best_params = param_grid[0]

    # Train final model with best params on full training set
    model = model_class(**best_params)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_metrics = _evaluate_model(model, X_train_scaled, y_train, prefix="train")
    val_metrics = _evaluate_model(model, X_val_scaled, y_val, prefix="val")
    metrics = {**train_metrics, **val_metrics}

    # Feature importance
    feat_imp = _get_feature_importance(model, feature_names, is_tree)

    elapsed = time.time() - start

    print(f"    Train AUROC: {metrics['train_auroc']:.4f} | Val AUROC: {metrics['val_auroc']:.4f}")
    print(f"    Val Sensitivity: {metrics['val_recall']:.4f} | Val Specificity: {metrics['val_specificity']:.4f}")
    print(f"    Training time: {elapsed:.1f}s")

    card = ModelCard(
        name=f"{name}-SepsisVitals",
        version="1.0.0",
        description=f"{name} model trained on synthetic clinical data for sepsis identification",
        metrics=metrics,
        training_data=f"Synthetic dataset calibrated to NHANES population distributions",
        fairness_notes="Trained with demographic features; fairness audit recommended",
    )

    return ModelResult(
        name=name,
        model=model,
        scaler=scaler,
        metrics=metrics,
        feature_importance=feat_imp,
        training_time=elapsed,
        card=card,
    )


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_cv_folds: int = 5,
) -> List[ModelResult]:
    """Train all 5 model types and return results."""
    configs = _get_model_configs()
    results = []

    for name, config in configs.items():
        result = train_single_model(
            name=name,
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            n_cv_folds=n_cv_folds,
        )
        results.append(result)

    return results


def select_best_model(results: List[ModelResult]) -> ModelResult:
    """Select the best model based on validation AUROC, with clinical sensitivity as tiebreaker."""
    return max(
        results,
        key=lambda r: (
            r.metrics.get("val_auroc", 0),
            r.metrics.get("val_recall", 0),
            -r.metrics.get("val_brier", 1),
        ),
    )


def calibrate_model(
    result: ModelResult,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> ModelResult:
    """Apply Platt scaling calibration to improve probability estimates."""
    calibrated = CalibratedClassifierCV(result.model, cv="prefit", method="sigmoid")
    calibrated.fit(X_val, y_val)

    result.model = calibrated
    result.is_calibrated = True
    return result


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 1000,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute SHAP values for model interpretability.

    Returns (shap_values, mean_abs_shap_importance).
    """
    import shap

    if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
        # Tree-based model
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model, X[:min(100, len(X))])
    elif hasattr(model, "calibrated_classifiers_"):
        # Calibrated model — use the base estimator
        base = model.calibrated_classifiers_[0].estimator
        try:
            explainer = shap.TreeExplainer(base)
        except Exception:
            explainer = shap.Explainer(base, X[:min(100, len(X))])
    else:
        explainer = shap.Explainer(model, X[:min(100, len(X))])

    X_sample = X[:max_samples] if len(X) > max_samples else X
    shap_values = explainer.shap_values(X_sample)

    # Handle multi-output SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (sepsis)

    # Mean absolute SHAP importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True,
    ))

    return shap_values, importance


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    result: ModelResult,
    feature_names: List[str],
    output_dir: str = "models",
) -> str:
    """Save trained model, scaler, and metadata to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = output_path / "sepsis_model.joblib"
    joblib.dump(result.model, model_file)

    # Save scaler if used
    if result.scaler is not None:
        scaler_file = output_path / "scaler.joblib"
        joblib.dump(result.scaler, scaler_file)

    # Save metadata
    metadata = {
        "model_name": result.name,
        "version": result.card.version,
        "feature_names": feature_names,
        "needs_scaling": result.scaler is not None,
        "is_calibrated": result.is_calibrated,
        "metrics": {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in result.metrics.items()},
        "feature_importance": {k: round(v, 4) for k, v in
                                (result.feature_importance or {}).items()},
        "training_time_seconds": round(result.training_time, 1),
        "model_card": result.card.to_dict(),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    metadata_file = output_path / "model_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n  Model saved to {output_path}/")
    print(f"    - sepsis_model.joblib ({model_file.stat().st_size / 1024:.0f} KB)")
    print(f"    - model_metadata.json")
    if result.scaler is not None:
        print(f"    - scaler.joblib")

    return str(output_path)


def generate_evaluation_report(
    results: List[ModelResult],
    best: ModelResult,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    shap_importance: Dict[str, float],
    output_dir: str = "models",
) -> str:
    """Generate a comprehensive evaluation report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Test set metrics for the best model
    if best.scaler is not None:
        X_test_scaled = best.scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    test_metrics = _evaluate_model(best.model, X_test_scaled, y_test, prefix="test")

    # Calibration metrics
    y_prob = best.model.predict_proba(X_test_scaled)[:, 1]
    cal_metrics = calibration_metrics(y_test, y_prob)

    # Build report
    report = {
        "report_type": "Sepsis Model Evaluation Report",
        "generated_at": pd.Timestamp.now().isoformat(),
        "best_model": {
            "name": best.name,
            "version": best.card.version,
            "is_calibrated": best.is_calibrated,
        },
        "model_comparison": [],
        "test_set_performance": test_metrics,
        "calibration": {
            "ece": cal_metrics["ece"],
            "brier_score": cal_metrics["brier_score"],
            "quality": cal_metrics["calibration_quality"],
        },
        "shap_feature_importance": {k: round(v, 4) for k, v in
                                     list(shap_importance.items())[:20]},
        "clinical_performance": {
            "sensitivity": test_metrics["test_recall"],
            "specificity": test_metrics["test_specificity"],
            "ppv": test_metrics["test_ppv"],
            "npv": test_metrics["test_npv"],
            "sensitivity_at_90_specificity": test_metrics.get("test_sensitivity_at_90spec", 0),
            "sensitivity_at_95_specificity": test_metrics.get("test_sensitivity_at_95spec", 0),
        },
        "roc_curve": None,
        "pr_curve": None,
    }

    # Model comparison table
    for r in results:
        report["model_comparison"].append({
            "name": r.name,
            "val_auroc": r.metrics.get("val_auroc", 0),
            "val_auprc": r.metrics.get("val_auprc", 0),
            "val_sensitivity": r.metrics.get("val_recall", 0),
            "val_specificity": r.metrics.get("val_specificity", 0),
            "val_f1": r.metrics.get("val_f1", 0),
            "training_time_s": r.training_time,
        })

    # ROC curve data
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    report["roc_curve"] = {
        "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//100)]],
        "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//100)]],
    }

    # PR curve data
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)
    report["pr_curve"] = {
        "precision": [round(float(x), 4) for x in prec[::max(1, len(prec)//100)]],
        "recall": [round(float(x), 4) for x in rec[::max(1, len(rec)//100)]],
    }

    # Save report
    report_file = output_path / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Also generate a human-readable text report
    text_report = _generate_text_report(report, results, best)
    text_file = output_path / "evaluation_report.txt"
    with open(text_file, "w") as f:
        f.write(text_report)

    return str(report_file)


def _generate_text_report(
    report: Dict,
    results: List[ModelResult],
    best: ModelResult,
) -> str:
    """Generate human-readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  SEPSIS VITALS — MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"\n  Generated: {report['generated_at']}")
    lines.append(f"  Best Model: {best.name} (v{best.card.version})")
    lines.append(f"  Calibrated: {'Yes' if best.is_calibrated else 'No'}")

    lines.append("\n" + "─" * 70)
    lines.append("  MODEL COMPARISON (Validation Set)")
    lines.append("─" * 70)
    lines.append(f"  {'Model':<20} {'AUROC':>8} {'AUPRC':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'Time':>8}")
    lines.append("  " + "─" * 60)
    for comp in report["model_comparison"]:
        marker = " ***" if comp["name"] == best.name else ""
        lines.append(
            f"  {comp['name']:<20} "
            f"{comp['val_auroc']:>8.4f} "
            f"{comp['val_auprc']:>8.4f} "
            f"{comp['val_sensitivity']:>8.4f} "
            f"{comp['val_specificity']:>8.4f} "
            f"{comp['val_f1']:>8.4f} "
            f"{comp['training_time_s']:>7.1f}s"
            f"{marker}"
        )

    test = report["test_set_performance"]
    lines.append("\n" + "─" * 70)
    lines.append("  TEST SET PERFORMANCE (Best Model)")
    lines.append("─" * 70)
    lines.append(f"  AUROC:              {test['test_auroc']:.4f}")
    lines.append(f"  AUPRC:              {test['test_auprc']:.4f}")
    lines.append(f"  Accuracy:           {test['test_accuracy']:.4f}")
    lines.append(f"  Sensitivity:        {test['test_recall']:.4f}")
    lines.append(f"  Specificity:        {test['test_specificity']:.4f}")
    lines.append(f"  PPV:                {test['test_ppv']:.4f}")
    lines.append(f"  NPV:                {test['test_npv']:.4f}")
    lines.append(f"  F1 Score:           {test['test_f1']:.4f}")
    lines.append(f"  Brier Score:        {test['test_brier']:.4f}")

    clin = report["clinical_performance"]
    lines.append(f"\n  Sensitivity @ 90% Specificity: {clin['sensitivity_at_90_specificity']:.4f}")
    lines.append(f"  Sensitivity @ 95% Specificity: {clin['sensitivity_at_95_specificity']:.4f}")

    cal = report["calibration"]
    lines.append(f"\n  Calibration ECE:    {cal['ece']:.4f} ({cal['quality']})")
    lines.append(f"  Brier Score:        {cal['brier_score']:.4f}")

    lines.append(f"\n  Confusion Matrix:")
    lines.append(f"    TP: {test['test_true_positives']:>6}  FP: {test['test_false_positives']:>6}")
    lines.append(f"    FN: {test['test_false_negatives']:>6}  TN: {test['test_true_negatives']:>6}")

    lines.append("\n" + "─" * 70)
    lines.append("  TOP FEATURES (SHAP Importance)")
    lines.append("─" * 70)
    for i, (feat, imp) in enumerate(list(report["shap_feature_importance"].items())[:15]):
        bar = "█" * int(imp / max(report["shap_feature_importance"].values()) * 30)
        lines.append(f"  {i+1:>2}. {feat:<30} {imp:.4f}  {bar}")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)
