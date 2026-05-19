"""
sepsis_vitals.ml.fairness
~~~~~~~~~~~~~~~~~~~~~~~~~
ML fairness auditing, calibration metrics, conformal prediction,
alert explanation, and counterfactual generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# audit_fairness
# ---------------------------------------------------------------------------

def audit_fairness(
    df: pd.DataFrame,
    label_col: str,
    prob_col: str,
    group_cols: List[str],
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """Compute fairness metrics across demographic subgroups.

    Returns
    -------
    dict with keys:
        subgroups      - list of dicts, each with group_name, n, auc, accuracy
        fairness_flags - list of strings describing large performance gaps
    """
    subgroups: List[Dict[str, Any]] = []

    # --- overall ---
    y_true = df[label_col].values
    y_prob = df[prob_col].values
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        overall_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        overall_auc = float("nan")

    overall_acc = float(accuracy_score(y_true, y_pred))

    subgroups.append({
        "group_name": "overall",
        "n": len(df),
        "auc": overall_auc,
        "accuracy": overall_acc,
    })

    # --- per-group metrics ---
    for col in group_cols:
        for value, grp in df.groupby(col):
            if len(grp) < min_group_size:
                continue

            yt = grp[label_col].values
            yp = grp[prob_col].values
            ypr = (yp >= 0.5).astype(int)

            try:
                auc = float(roc_auc_score(yt, yp))
            except ValueError:
                auc = float("nan")

            acc = float(accuracy_score(yt, ypr))

            subgroups.append({
                "group_name": f"{col}={value}",
                "n": len(grp),
                "auc": auc,
                "accuracy": acc,
            })

    # --- fairness flags ---
    fairness_flags: List[str] = []

    for col in group_cols:
        col_entries = [s for s in subgroups if s["group_name"].startswith(f"{col}=")]
        if len(col_entries) < 2:
            continue

        aucs = [s["auc"] for s in col_entries if not np.isnan(s["auc"])]
        accs = [s["accuracy"] for s in col_entries]

        if aucs:
            auc_gap = max(aucs) - min(aucs)
            if auc_gap > 0.05:
                fairness_flags.append(
                    f"AUC gap of {auc_gap:.3f} across {col} subgroups"
                )

        if accs:
            acc_gap = max(accs) - min(accs)
            if acc_gap > 0.05:
                fairness_flags.append(
                    f"Accuracy gap of {acc_gap:.3f} across {col} subgroups"
                )

    return {"subgroups": subgroups, "fairness_flags": fairness_flags}


# ---------------------------------------------------------------------------
# calibration_metrics
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute calibration metrics.

    Returns
    -------
    dict with keys: ece, brier_score, reliability_diagram, calibration_quality
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Brier score
    bs = float(brier_score_loss(y_true, y_prob))

    # Reliability diagram + ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability: List[Dict[str, Any]] = []
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            reliability.append({
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": 0,
                "avg_predicted": None,
                "avg_observed": None,
            })
            continue

        avg_pred = float(y_prob[mask].mean())
        avg_obs = float(y_true[mask].mean())
        reliability.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "count": n_bin,
            "avg_predicted": avg_pred,
            "avg_observed": avg_obs,
        })
        ece += (n_bin / total) * abs(avg_pred - avg_obs)

    ece = float(ece)

    if ece < 0.05:
        quality = "excellent"
    elif ece < 0.10:
        quality = "good"
    elif ece < 0.15:
        quality = "moderate"
    else:
        quality = "poor"

    return {
        "ece": ece,
        "brier_score": bs,
        "reliability_diagram": reliability,
        "calibration_quality": quality,
    }


# ---------------------------------------------------------------------------
# ConformalPredictor
# ---------------------------------------------------------------------------

class ConformalPredictor:
    """Split conformal prediction for binary classifiers."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._calibrated = False
        self._quantile: Optional[float] = None

    def calibrate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute nonconformity scores on calibration set."""
        proba = model.predict_proba(X)[:, 1]
        y_arr = np.asarray(y, dtype=float)

        # Nonconformity score: 1 - predicted prob for the true class
        scores = np.where(y_arr == 1, 1 - proba, proba)

        # Quantile at (1-alpha) adjusted for finite-sample validity
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._quantile = float(np.quantile(scores, q_level))
        self._calibrated = True

    def predict_interval(
        self, model: Any, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, upper, uncertain) arrays."""
        if not self._calibrated:
            raise RuntimeError(
                "Must call calibrate() before predict_interval."
            )

        proba = model.predict_proba(X)[:, 1]
        q = self._quantile

        # Prediction set: classes whose nonconformity score <= quantile
        # For class 1: score = 1 - proba  =>  include if 1 - proba <= q  => proba >= 1 - q
        # For class 0: score = proba      =>  include if proba <= q
        include_1 = proba >= (1 - q)
        include_0 = proba <= q

        lower = np.where(include_0, 0.0, proba)
        upper = np.where(include_1, 1.0, proba)

        # Ensure lower <= upper
        lower = np.minimum(lower, upper)

        # Uncertain if both classes are in the prediction set
        uncertain = include_0 & include_1

        return lower, upper, uncertain


# ---------------------------------------------------------------------------
# generate_alert_explanation
# ---------------------------------------------------------------------------

def generate_alert_explanation(
    vitals: Dict[str, Any],
    qsofa: int,
    sirs: int,
    shock_index: Optional[float],
    risk_level: str,
    language: str = "en",
) -> str:
    """Generate a human-readable explanation for a sepsis alert."""
    risk_upper = risk_level.upper()

    parts: List[str] = []
    parts.append(f"Sepsis risk level: {risk_upper}.")

    # Vital sign details
    vital_details = []
    if "resp_rate" in vitals:
        vital_details.append(f"respiratory rate {vitals['resp_rate']}")
    if "sbp" in vitals:
        vital_details.append(f"systolic BP {vitals['sbp']}")
    if "gcs" in vitals:
        vital_details.append(f"GCS {vitals['gcs']}")
    if "heart_rate" in vitals:
        vital_details.append(f"heart rate {vitals['heart_rate']}")
    if "temperature" in vitals:
        vital_details.append(f"temperature {vitals['temperature']}")

    if vital_details:
        parts.append("Contributing vitals: " + ", ".join(vital_details) + ".")

    parts.append(f"qSOFA score: {qsofa}; SIRS criteria met: {sirs}.")

    if shock_index is not None:
        parts.append(f"Shock index: {shock_index:.2f}.")

    explanation = " ".join(parts)

    if language != "en":
        explanation = f"[SW-PENDING] {explanation}"

    return explanation


# ---------------------------------------------------------------------------
# generate_counterfactual
# ---------------------------------------------------------------------------

# Normal ranges for common vitals
_NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "resp_rate": (12.0, 20.0),
    "sbp": (100.0, 140.0),
    "gcs": (15.0, 15.0),
    "heart_rate": (60.0, 100.0),
    "temperature": (36.1, 37.2),
}


def generate_counterfactual(
    vitals: Dict[str, Any],
    risk_level: str,
) -> Optional[str]:
    """Suggest vital-sign changes that would lower sepsis risk.

    Returns None if risk_level is 'low'.
    """
    if risk_level == "low":
        return None

    suggestions: List[str] = []

    for name, value in vitals.items():
        if name not in _NORMAL_RANGES:
            continue
        lo, hi = _NORMAL_RANGES[name]
        val = float(value)
        if val > hi:
            suggestions.append(
                f"reduce {name} from {val:.0f} to ~{hi:.0f}"
            )
        elif val < lo:
            suggestions.append(
                f"raise {name} from {val:.0f} to ~{lo:.0f}"
            )

    if not suggestions:
        return f"Risk level is {risk_level.upper()}; no single vital change identified to lower risk."

    return (
        "To potentially lower the risk, consider: "
        + "; ".join(suggestions)
        + "."
    )
