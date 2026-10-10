"""Compare synthetic-generator profiles with a fixed evaluation protocol.

Engineering evidence only: every number here describes the hand-authored
generator, never patients. The protocol exists so that generator changes
(decoupling age, fixing physiology, changing the label definition) can be
judged with the same yardstick the clinical validation plan prescribes:
discrimination with uncertainty, calibration, operating-point behaviour,
missing-data behaviour, subgroups, a leakage probe and a temporal split.

The operating point (specificity 0.90 on the validation split) is an
engineering convention for comparing profiles, not a proposed clinical
threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DEMOGRAPHIC_COLS = [
    "age_years", "has_hypertension", "has_diabetes", "has_ckd", "has_copd", "has_heart_failure",
]
LAB_PREFIXES = ("lactate", "wbc", "procalcitonin")
TARGET_SPECIFICITY = 0.90  # engineering convention only

PROFILES: Dict[str, Dict[str, Any]] = {
    "legacy": {},
    "decoupled": {
        "age_effect": 0.0, "comorbidity_effect": 0.0,
        "septic_blend_ceiling": 1.0, "mimic_blend_ceiling": 0.6,
    },
}


def _auc(y: Any, p: Any) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def calibration(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Slope and intercept of logistic recalibration on logit(p)."""
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return {
        "slope": round(float(fit.coef_[0][0]), 3),
        "intercept": round(float(fit.intercept_[0]), 3),
        "in_the_large": round(float(np.mean(y) - np.mean(p)), 4),
    }


def _fit(train: pd.DataFrame, cols: List[str]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(
        train[cols].astype(float), train["sepsis_label"]
    )


def evaluate_profile(
    name: str,
    n_patients: int = 4000,
    seed: int = 7,
    label_mode: str = "current_state",
    horizon_hours: Optional[float] = None,
    n_bootstrap: int = 1000,
    **generator_options: Any,
) -> Dict[str, Any]:
    from sepsis_vitals.ml.synthetic_data import generate_dataset
    from sepsis_vitals.ml.trainer import prepare_features

    df = generate_dataset(
        n_patients=n_patients, seed=seed, label_mode=label_mode,
        horizon_hours=horizon_hours, include_onset_time=True, **generator_options,
    )
    start = df.groupby("patient_id")["timestamp"].min()

    # Patient-level split, ordered by admission time: train on the earliest
    # 60%, validate on the next 15%, test on the latest 25% (temporal).
    order = start.sort_values().index.to_numpy()
    n_train, n_val = int(len(order) * 0.60), int(len(order) * 0.15)
    split = {pid: "train" for pid in order[:n_train]}
    split.update({pid: "val" for pid in order[n_train:n_train + n_val]})
    split.update({pid: "test" for pid in order[n_train + n_val:]})

    features, cols = prepare_features(df)
    features["split"] = features["patient_id"].map(split)
    train, val, test = (features[features["split"] == s] for s in ("train", "val", "test"))

    model = _fit(train, cols)
    p_val = model.predict_proba(val[cols].astype(float))[:, 1]
    p_test = model.predict_proba(test[cols].astype(float))[:, 1]
    y_val, y_test = val["sepsis_label"].to_numpy(), test["sepsis_label"].to_numpy()

    negatives = np.sort(p_val[y_val == 0])
    threshold = float(negatives[int(np.ceil(TARGET_SPECIFICITY * len(negatives))) - 1]) if len(negatives) else 0.5
    flagged = p_test > threshold
    tp, fp = int((flagged & (y_test == 1)).sum()), int((flagged & (y_test == 0)).sum())
    fn, tn = int((~flagged & (y_test == 1)).sum()), int((~flagged & (y_test == 0)).sum())

    no_labs = test[cols].astype(float).copy()
    for c in cols:
        if c.startswith(LAB_PREFIXES):
            no_labs[c] = np.nan
        if c.endswith("_missing") and c.startswith(LAB_PREFIXES):
            no_labs[c] = 1.0
        if c == "n_labs_missing":
            no_labs[c] = 3.0
    p_nolabs = model.predict_proba(no_labs)[:, 1]

    demo_cols = [c for c in DEMOGRAPHIC_COLS if c in cols]
    p_demo = _fit(train, demo_cols).predict_proba(test[demo_cols].astype(float))[:, 1]

    # Paired patient-level bootstrap: the model, its no-labs evaluation and the
    # demographics-only probe are scored on the same resampled patients.
    from sepsis_vitals.ml.uncertainty import paired_cluster_bootstrap

    uncertainty = paired_cluster_bootstrap(
        test["patient_id"].to_numpy(), y_test,
        {"model": p_test, "no_labs": p_nolabs, "demographics_only": p_demo},
        reference="model", metrics=("auroc",), n_boot=n_bootstrap, seed=seed,
    )
    age_band = pd.cut(test["age_years"], [0, 44, 64, 200], labels=["18-44", "45-64", "65+"])
    sex = df.groupby("patient_id")["sex"].first() if "sex" in df else None
    subgroups: Dict[str, Any] = {}
    for label, mask in [(f"age {b}", (age_band == b).to_numpy()) for b in ["18-44", "45-64", "65+"]]:
        subgroups[label] = {"n_rows": int(mask.sum()), "auroc": _round(_auc(y_test[mask], p_test[mask]))}
    if sex is not None:
        test_sex = test["patient_id"].map(sex).to_numpy()
        for s in ("F", "M"):
            mask = test_sex == s
            subgroups[f"sex {s}"] = {"n_rows": int(mask.sum()), "auroc": _round(_auc(y_test[mask], p_test[mask]))}

    return {
        "profile": name,
        "label_mode": label_mode,
        "horizon_hours": horizon_hours,
        "generator_options": generator_options,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "positive_rate_test": round(float(y_test.mean()), 4),
        "auroc": _round(_auc(y_test, p_test)),
        "auroc_95ci": uncertainty["arms"]["model"]["auroc"]["ci"],
        "auprc": _round(_average_precision(y_test, p_test)),
        "brier": round(float(np.mean((p_test - y_test) ** 2)), 4),
        "calibration": calibration(y_test, p_test),
        "operating_point": {
            "threshold_from_validation": round(threshold, 4),
            "sensitivity": _round(tp / (tp + fn) if tp + fn else None),
            "specificity": _round(tn / (tn + fp) if tn + fp else None),
            "ppv": _round(tp / (tp + fp) if tp + fp else None),
            "flag_rate": round(float(flagged.mean()), 4),
        },
        "auroc_without_labs": _round(_auc(y_test, p_nolabs)),
        "auroc_without_labs_95ci": uncertainty["arms"]["no_labs"]["auroc"]["ci"],
        "auroc_demographics_only": _round(_auc(y_test, p_demo)),
        "auroc_demographics_only_95ci": uncertainty["arms"]["demographics_only"]["auroc"]["ci"],
        "uncertainty": uncertainty,
        "subgroups": subgroups,
        "split": "patient-level, temporal by admission time (60/15/25)",
    }


def _average_precision(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)
