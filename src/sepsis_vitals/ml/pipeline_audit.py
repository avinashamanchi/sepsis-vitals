"""Diagnostics that test what the synthetic development pipeline can and cannot show.

The headline synthetic AUROC is easy to over-read. These checks quantify:

* generator artefacts (non-monotonic timestamps, physiology of labelled rows);
* how much discrimination comes from demographics alone versus vital signs;
* whether the committed model separates *future* sepsis (pre-onset rows) from
  never-septic patients, i.e. early warning rather than detection of the
  current state;
* train/serve skew: the API scores single observations with no deltas,
  rolling statistics, or observation gap, unlike the training features.

Everything here is synthetic development evidence, never clinical evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DEMOGRAPHIC_COLS = [
    "age_years", "has_hypertension", "has_diabetes", "has_ckd", "has_copd", "has_heart_failure",
]
_VITAL_PREFIXES = ("temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "gcs", "map")
_SCORE_COLS = ("qsofa", "news2_computed", "sirs_computed", "shock_index_computed")
_LAB_PREFIXES = ("lactate", "wbc", "procalcitonin")


def _auroc(y: Any, p: Any) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def generator_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """Describe the raw generator output (before feature engineering)."""
    ordered = df.sort_index()
    gaps = ordered.groupby("patient_id")["timestamp"].diff().dt.total_seconds()
    ever = df.groupby("patient_id")["sepsis_label"].transform("max")
    cols = ["heart_rate", "resp_rate", "sbp", "temperature", "spo2", "lactate", "procalcitonin"]
    by_label = df.groupby("sepsis_label")[cols].mean().round(2)
    patients = df.groupby("patient_id").agg(age=("age_years", "first"), y=("sepsis_label", "max"))
    return {
        "fraction_negative_time_gaps": round(float((gaps.dropna() < 0).mean()), 3),
        "row_mean_by_label": {
            str(label): {k: float(v) for k, v in row.items()}
            for label, row in by_label.iterrows()
        },
        "mean_age_ever_septic": round(float(df.loc[ever == 1, "age_years"].mean()), 1),
        "mean_age_never_septic": round(float(df.loc[ever == 0, "age_years"].mean()), 1),
        "patient_level_auroc_age_alone": round(_auroc(patients["y"].values, patients["age"].values), 3),
    }


def feature_groups(cols: List[str]) -> Dict[str, List[str]]:
    """Partition model features into the groups compared by the audit."""
    vitals = [c for c in cols if c.startswith(_VITAL_PREFIXES) or c in _SCORE_COLS]
    no_labs = [c for c in cols if not c.startswith(_LAB_PREFIXES) and c != "n_labs_missing"]
    return {
        "demographics_and_comorbidities": [c for c in DEMOGRAPHIC_COLS if c in cols],
        "vitals_and_scores_only": vitals,
        "no_labs": no_labs,
        "full": list(cols),
    }


def feature_group_aurocs(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    """Held-out row-level AUROC of a fixed learner trained on each feature group."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    out: Dict[str, Any] = {}
    for name, group in feature_groups(cols).items():
        model = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        model.fit(train[group].astype(float), train["sepsis_label"])
        p = model.predict_proba(test[group].astype(float))[:, 1]
        out[name] = {"n_features": len(group), "test_row_auroc": round(_auroc(test["sepsis_label"].values, p), 3)}
    return out


def inference_style(features: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Rebuild features the way ``SepsisPredictor._build_feature_vector`` does."""
    X = features[feature_names].astype(float).copy()
    for c in feature_names:
        if c.endswith("_delta") or c.endswith("_roll_std"):
            X[c] = np.nan
        elif c.endswith("_roll_mean") and c.replace("_roll_mean", "") in features:
            X[c] = features[c.replace("_roll_mean", "")].astype(float)
    if "obs_gap_min" in X:
        X["obs_gap_min"] = np.nan
    return X


def committed_model_checks(
    test: pd.DataFrame,
    model: Any,
    feature_names: List[str],
    medians: Dict[str, float],
    n_bootstrap: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Score the committed artifact on a fresh synthetic cohort."""
    def impute(X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna({k: v for k, v in medians.items() if k in X.columns}).fillna(0)

    y = test["sepsis_label"].to_numpy()
    p = model.predict_proba(impute(test[feature_names].astype(float)))[:, 1]
    p_inf = model.predict_proba(impute(inference_style(test, feature_names)))[:, 1]

    ever = test.groupby("patient_id")["sepsis_label"].transform("max").to_numpy()
    first_pos = test[test["sepsis_label"] == 1].groupby("patient_id")["timestamp"].min()
    onset = test["patient_id"].map(first_pos)
    pre = (ever == 1) & (y == 0) & (test["timestamp"] < onset).to_numpy()
    never = ever == 0
    pre_auc = _auroc(np.r_[np.ones(pre.sum()), np.zeros(never.sum())], np.r_[p[pre], p[never]])

    rng = np.random.default_rng(seed)
    pids = test["patient_id"].unique()
    rows_by_pid = test.groupby("patient_id").indices
    boots = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([rows_by_pid[pid] for pid in rng.choice(pids, len(pids))])
        if 0 < y[idx].sum() < len(idx):
            boots.append(_auroc(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))

    patient = pd.DataFrame({"pid": test["patient_id"].values, "p": p, "y": ever}).groupby("pid").max()
    return {
        "row_auroc_training_style_features": round(_auroc(y, p), 3),
        "row_auroc_95ci_patient_bootstrap": [round(float(lo), 3), round(float(hi), 3)],
        "row_auroc_inference_style_features": round(_auroc(y, p_inf), 3),
        "mean_predicted_risk_training_style": round(float(p.mean()), 3),
        "mean_predicted_risk_inference_style": round(float(p_inf.mean()), 3),
        "observed_positive_row_rate": round(float(y.mean()), 3),
        "pre_onset_vs_never_septic_auroc": round(pre_auc, 3),
        "n_pre_onset_rows": int(pre.sum()),
        "patient_level_auroc_max_risk": round(_auroc(patient["y"].values, patient["p"].values), 3),
    }


def run_audit(
    n_patients: int = 6000,
    seed: int = 7,
    model_dir: Optional[str] = "models",
    n_bootstrap: int = 200,
) -> Dict[str, Any]:
    """Run every diagnostic on a fresh synthetic cohort."""
    from sepsis_vitals.ml.synthetic_data import generate_dataset, generate_train_val_test
    from sepsis_vitals.ml.trainer import prepare_features

    raw = generate_dataset(n_patients=min(n_patients, 3000), seed=seed)
    train, _, test = generate_train_val_test(n_patients=n_patients, seed=seed)
    Xtr, cols = prepare_features(train)
    Xte, _ = prepare_features(test)

    report: Dict[str, Any] = {
        "experiment": "synthetic_pipeline_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "synthetic development evidence only",
        "cohort": {"n_patients": n_patients, "seed": seed, "test_rows": int(len(Xte))},
        "generator": generator_diagnostics(raw),
        "feature_groups": feature_group_aurocs(Xtr, Xte, cols),
    }
    if model_dir is not None and (Path(model_dir) / "sepsis_model.joblib").exists():
        import joblib

        meta = json.loads((Path(model_dir) / "model_metadata.json").read_text())
        medians = json.loads((Path(model_dir) / "imputation_medians.json").read_text())
        # joblib unpickles: only point model_dir at the repository's own,
        # version-controlled artifacts, never at downloaded or user-supplied files.
        model = joblib.load(Path(model_dir) / "sepsis_model.joblib")
        report["committed_model"] = committed_model_checks(
            Xte, model, meta["feature_names"], medians, n_bootstrap=n_bootstrap, seed=seed
        )
    return report
