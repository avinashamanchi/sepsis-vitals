"""
model_scaffold.py – Phase 1 model development scaffold.

Implements the "boring first model" strategy from the design principles:
  1. Logistic regression baseline with calibration curve
  2. LightGBM with Platt scaling
  3. SHAP feature importance
  4. Cross-site leave-one-site-out validation
  5. Threshold search targeting sensitivity ≥ 0.85 at < 60 alerts/100 enc
  6. Model card metadata export

All heavy dependencies (sklearn, lightgbm, shap) are imported lazily —
the module remains importable even if they are not installed, which keeps
the core sepsis_vitals package lightweight.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Feature column registry
# ──────────────────────────────────────────────────────────────────────────────

RAW_VITALS = [
    "temperature", "heart_rate", "resp_rate", "sbp", "spo2", "gcs",
]

MISSINGNESS_FEATURES = [f"{v}_missing" for v in RAW_VITALS]

DELTA_FEATURES = [
    f"{v}_delta"     for v in RAW_VITALS
] + [
    f"{v}_delta_abs" for v in RAW_VITALS
]

ROLLING_FEATURES = [
    f"{v}_{stat}"
    for v in RAW_VITALS
    for stat in ("roll_mean", "roll_std", "roll_min", "roll_max")
]

SCORE_FEATURES = [
    "qsofa", "sirs_count", "shock_index", "news2_style", "uva_style",
]

META_FEATURES = ["n_vitals_missing", "n_vitals_present", "obs_gap_min"]

DEFAULT_FEATURES = (
    RAW_VITALS
    + MISSINGNESS_FEATURES
    + DELTA_FEATURES
    + ROLLING_FEATURES
    + SCORE_FEATURES
    + META_FEATURES
)


def get_feature_columns(df: pd.DataFrame, subset: list[str] = None) -> list[str]:
    """Return feature columns that are actually present in df."""
    cols = subset or DEFAULT_FEATURES
    return [c for c in cols if c in df.columns]


# ──────────────────────────────────────────────────────────────────────────────
# Model card
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCard:
    """Structured model card (inspired by Mitchell et al., 2019)."""
    model_name:         str = "sepsis-vitals-v1"
    model_type:         str = "gradient_boosted_trees"
    version:            str = "0.1.0"
    trained_at:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    training_sites:     list[str] = field(default_factory=list)
    validation_sites:   list[str] = field(default_factory=list)
    n_train:            int = 0
    n_val:              int = 0
    n_positives_train:  int = 0
    prevalence_train:   float = 0.0
    features_used:      list[str] = field(default_factory=list)
    label_definition:   str = "Sepsis within 6h of first alert window (clinician adjudicated)"
    auroc:              Optional[float] = None
    auprc:              Optional[float] = None
    sensitivity:        Optional[float] = None
    specificity:        Optional[float] = None
    ppv:                Optional[float] = None
    npv:                Optional[float] = None
    decision_threshold: Optional[float] = None
    alerts_per_100_enc: Optional[float] = None
    calibration_brier:  Optional[float] = None
    intended_use:       str = (
        "Early nurse escalation support in LMIC district hospitals. "
        "Not for diagnosis. Not a substitute for clinical judgment."
    )
    known_limitations:  list[str] = field(default_factory=lambda: [
        "Trained on retrospective data — prospective validation required before deployment.",
        "Pediatric performance not validated; Phoenix criteria not used for labelling.",
        "SpO2 reliability varies with device quality in low-resource settings.",
        "Model may underperform at sites with missingness patterns different from training.",
    ])
    bias_notes:         str = ""
    contact:            str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelCard":
        data = json.loads(Path(path).read_text())
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────────────
# Train / evaluate helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require(pkg: str, extra: str = ""):
    """Lazy import with helpful error message."""
    try:
        import importlib
        return importlib.import_module(pkg)
    except ImportError:
        hint = f"  pip install {extra or pkg}" 
        raise ImportError(
            f"Package '{pkg}' is required for model training.\n{hint}"
        )


def prepare_Xy(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str] | None = None,
    drop_na_label: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features X and labels y from a labelled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (output of build_feature_set).
    label_col : str
        Binary outcome column (1 = sepsis, 0 = non-sepsis).
    feature_cols : list[str] | None
        If None, uses get_feature_columns(df).
    drop_na_label : bool
        Drop rows where label is missing.

    Returns
    -------
    (X, y) tuple of DataFrame and Series.
    """
    if drop_na_label:
        df = df[df[label_col].notna()].copy()

    feat_cols = get_feature_columns(df, feature_cols)
    X = df[feat_cols].copy()
    y = df[label_col].astype(int)

    # Fill remaining NaN with column medians — missingness already encoded
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def train_logistic_baseline(X_train, y_train, calibrate: bool = True):
    """
    Fit a calibrated logistic regression baseline.

    Returns (pipeline, calibrated_pipeline_or_None).
    """
    sklearn = _require("sklearn", "scikit-learn>=1.4")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    if calibrate:
        cal = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        cal.fit(X_train, y_train)
        return pipe, cal

    return pipe, None


def train_lgbm(X_train, y_train, calibrate: bool = True, params: dict | None = None):
    """
    Fit a LightGBM classifier with optional Platt calibration.

    Returns (lgbm_model, calibrated_model_or_None).
    """
    lgb = _require("lightgbm", "lightgbm>=4.0")
    sklearn = _require("sklearn", "scikit-learn>=1.4")
    from sklearn.calibration import CalibratedClassifierCV

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    default_params = {
        "n_estimators":     400,
        "max_depth":        6,
        "num_leaves":       31,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": pos_weight,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    if calibrate:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        cal.fit(X_train, y_train)
        return model, cal

    return model, None


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute standard classification metrics.

    Returns dict with auroc, auprc, and threshold-independent stats.
    """
    sklearn = _require("sklearn", "scikit-learn>=1.4")
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss
    )

    probs = model.predict_proba(X_test)[:, 1]
    auroc = round(float(roc_auc_score(y_test, probs)), 4)
    auprc = round(float(average_precision_score(y_test, probs)), 4)
    brier = round(float(brier_score_loss(y_test, probs)), 4)

    return {"auroc": auroc, "auprc": auprc, "calibration_brier": brier}


def find_threshold(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_sensitivity: float = 0.85,
    max_alerts_per_100: float = 60.0,
    n_thresholds: int = 200,
) -> dict:
    """
    Search for the lowest threshold that achieves min_sensitivity without
    exceeding the alert burden cap.

    Returns dict with threshold, sensitivity, specificity, ppv, npv,
    alerts_per_100_enc, and feasible flag.
    """
    sklearn = _require("sklearn", "scikit-learn>=1.4")
    from sklearn.metrics import confusion_matrix

    probs = model.predict_proba(X_val)[:, 1]
    best  = None

    for thresh in np.linspace(0.05, 0.95, n_thresholds):
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        alerts_per_100 = (tp + fp) / len(y_val) * 100

        if sens >= min_sensitivity and alerts_per_100 <= max_alerts_per_100:
            if best is None or thresh > best["threshold"]:
                best = {
                    "threshold":        round(float(thresh), 3),
                    "sensitivity":      round(sens, 3),
                    "specificity":      round(spec, 3),
                    "ppv":              round(ppv, 3),
                    "npv":              round(npv, 3),
                    "alerts_per_100_enc": round(alerts_per_100, 1),
                    "feasible":         True,
                }

    return best or {
        "threshold": 0.5, "feasible": False,
        "note": f"No threshold met sensitivity≥{min_sensitivity} "
                f"and alerts≤{max_alerts_per_100}/100 simultaneously.",
    }


def compute_shap(model, X: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
    """
    Compute mean |SHAP| values for feature importance.

    Returns a DataFrame sorted by importance descending.
    Skips gracefully if shap is not installed.
    """
    try:
        import shap
    except ImportError:
        warnings.warn("shap not installed — skipping SHAP analysis. pip install shap")
        return pd.DataFrame()

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    importance = pd.DataFrame({
        "feature":    X.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance.head(max_display)


# ──────────────────────────────────────────────────────────────────────────────
# Leave-one-site-out cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def leave_one_site_out(
    df: pd.DataFrame,
    label_col: str,
    site_col: str = "site_id",
    model_fn=None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run leave-one-site-out cross-validation.

    For each site: train on all other sites, evaluate on held-out site.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered + labelled DataFrame.
    label_col : str
        Binary label column.
    site_col : str
        Column identifying partner sites.
    model_fn : callable | None
        Function(X_train, y_train) -> fitted model.
        Defaults to train_lgbm with calibration.
    feature_cols : list[str] | None
        Feature columns to use.

    Returns
    -------
    DataFrame with one row per site, columns:
        site, n_train, n_val, auroc, auprc, brier
    """
    if site_col not in df.columns:
        raise ValueError(f"site_col='{site_col}' not found. Add site_id to the DataFrame.")

    if model_fn is None:
        def model_fn(X_tr, y_tr):
            _, cal = train_lgbm(X_tr, y_tr, calibrate=True)
            return cal

    sites   = df[site_col].unique()
    records = []

    for held_out in sites:
        train_df = df[df[site_col] != held_out]
        val_df   = df[df[site_col] == held_out]

        X_tr, y_tr = prepare_Xy(train_df, label_col, feature_cols)
        X_val, y_val = prepare_Xy(val_df, label_col, feature_cols)

        if len(y_val.unique()) < 2:
            warnings.warn(f"Site '{held_out}' has only one class in validation — skipping.")
            continue

        model = model_fn(X_tr, y_tr)
        metrics = evaluate_model(model, X_val, y_val)

        records.append({
            "site":    held_out,
            "n_train": len(y_tr),
            "n_val":   len(y_val),
            **metrics,
        })

    return pd.DataFrame(records)
