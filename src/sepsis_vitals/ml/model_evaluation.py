"""Evaluation of the committed model artifact on a fresh synthetic cohort.

Engineering evidence only. The cohort comes from the same hand-written
generator family the model was trained on, so every number here describes
the model's behaviour on that generator. None of it estimates performance in
patients.

The artifact is scored exactly as the API scores it:

- the training feature pipeline;
- training-median imputation;
- the artifact's scaler, if any;
- the threshold policy in ``ml/decision_policy.py``;
- the rule-based scores.

Every observation is scored with its full history; at the API, unregistered
patients are scored as first observations instead (see
reports/synthetic_pipeline_audit.md).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

LAB_COLUMNS = ("lactate", "wbc", "procalcitonin")
DCA_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)

# Thresholds at which the probes below are read. They are the SIRS and qSOFA
# criteria already implemented in sepsis_vitals.scores, not new clinical
# cut-offs. Each probe selects observations with that finding.
FEVER_ABOVE = 38.0
TACHYCARDIA_ABOVE = 90.0
BEHAVIOUR_PROBES = (
    ("temperature > 38", "temperature", lambda v: v > FEVER_ABOVE),
    ("heart_rate > 90", "heart_rate", lambda v: v > TACHYCARDIA_ABOVE),
    ("resp_rate >= 22", "resp_rate", lambda v: v >= 22),
    ("sbp <= 100", "sbp", lambda v: v <= 100),
)
# Engineering gate (not a clinical criterion): a finding must lower the
# model output in fewer than half of the observations that have it, and
# raise it on average. A flat model fails (mean change 0). The committed
# legacy model fails every probe (N53).
MAX_SHARE_LOWERED = 0.5


def behaviour_gate(probes: Dict[str, Any]) -> Dict[str, Any]:
    """Pass or fail per probe, and overall."""
    results = {
        name: bool(p["share_where_value_lowered_output"] < MAX_SHARE_LOWERED
                   and p["mean_change_in_output"] > 0)
        for name, p in probes.items()
    }
    return {"rule": f"share lowered < {MAX_SHARE_LOWERED} and mean change > 0",
            "results": results, "passed": bool(results) and all(results.values())}


def _auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def _ap(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def score_frame(predictor: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Model output for every row of a multi-patient frame (full history)."""
    from sepsis_vitals.ml.trainer import prepare_features

    frame = df.copy()
    for col in ("on_supplemental_o2", "spo2_scale2"):
        if col not in frame:
            frame[col] = False
    features, _ = prepare_features(frame)
    X = features.reindex(columns=predictor.feature_names).to_numpy(dtype=np.float64)
    medians = np.array([predictor._imputation_medians.get(n, 0.0) for n in predictor.feature_names])
    X = np.where(np.isnan(X), medians, X)
    if predictor.scaler is not None:
        X = predictor.scaler.transform(X)
    out = features[["patient_id", "timestamp"]].copy()
    out["p"] = predictor.model.predict_proba(X)[:, 1]
    return out


def _counterfactual(predictor: Any, df: pd.DataFrame, column: str) -> np.ndarray:
    """Model output with one measured input replaced by its training median."""
    altered = df.copy()
    reference = predictor._imputation_medians[column]
    altered[column] = altered[column].where(altered[column].isna(), reference)
    return score_frame(predictor, altered)["p"].to_numpy()


def _confusion(flag: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    return {
        "sensitivity": _r(tp / (tp + fn) if tp + fn else None),
        "specificity": _r(tn / (tn + fp) if tn + fp else None),
        "ppv": _r(tp / (tp + fp) if tp + fp else None),
        "flag_rate": _r(flag.mean()),
    }


def _alert_burden(name: str, flag: np.ndarray, data: pd.DataFrame, patient_days: float) -> Dict[str, Any]:
    y = data["sepsis_label"].to_numpy()
    flagged = data.assign(flag=flag)
    septic = flagged[flagged["septic_patient"]]
    non_septic = flagged[~flagged["septic_patient"]]
    before_onset = septic[septic["timestamp"] < septic["sepsis_onset_time"]]
    after_onset = septic[septic["timestamp"] >= septic["sepsis_onset_time"]]
    return {
        "rule": name,
        **_confusion(flag, y),
        "septic_patients_flagged_at_or_after_onset": _r(after_onset.groupby("patient_id")["flag"].any().mean()),
        "septic_patients_flagged_before_onset": _r(before_onset.groupby("patient_id")["flag"].any().mean())
        if len(before_onset) else None,
        "non_septic_patients_ever_flagged": _r(non_septic.groupby("patient_id")["flag"].any().mean()),
        "flagged_observations_per_patient_day": _r(flag.sum() / patient_days, 2),
    }


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds=DCA_THRESHOLDS) -> List[Dict[str, float]]:
    """Net benefit of acting on the model at each threshold, against acting on all (row level)."""
    n = len(y)
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        flag = p >= t
        tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
        weight = t / (1 - t)
        rows.append({
            "threshold": t,
            "net_benefit_model": round(float(tp / n - fp / n * weight), 4),
            "net_benefit_all": round(float(prevalence - (1 - prevalence) * weight), 4),
        })
    return rows


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    """Observed rate against mean output, by decile of the output."""
    order = pd.qcut(p, q=bins, labels=False, duplicates="drop")
    table = pd.DataFrame({"bin": order, "y": y, "p": p}).groupby("bin")
    return [
        {"mean_output": round(float(g["p"].mean()), 4), "observed_rate": round(float(g["y"].mean()), 4),
         "n_rows": int(len(g))}
        for _, g in table
    ]


def evaluate_committed_model(
    model_dir: str = "models",
    n_patients: int = 4000,
    seed: int = 20261015,
    prevalence: float = 0.15,
    n_boot: int = 1000,
    generator_profile: str = "legacy",
) -> Dict[str, Any]:
    import tempfile

    from sepsis_vitals.ml.decision_policy import model_level, threshold_policy
    from sepsis_vitals.ml.predictor import SepsisPredictor
    from sepsis_vitals.ml.profile_evaluation import PROFILES, calibration
    from sepsis_vitals.ml.provenance import dataset_fingerprint
    from sepsis_vitals.ml.synthetic_data import generate_dataset
    from sepsis_vitals.ml.uncertainty import paired_cluster_bootstrap
    from sepsis_vitals.scores import compute_scores

    predictor = SepsisPredictor(model_dir=model_dir, state_dir=tempfile.mkdtemp(prefix="eval-state-"))
    predictor.load()
    df = generate_dataset(n_patients=n_patients, sepsis_prevalence=prevalence, seed=seed,
                          include_onset_time=True, **PROFILES[generator_profile])
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    scored = score_frame(predictor, df)
    data = df.merge(scored, on=["patient_id", "timestamp"], how="inner", validate="one_to_one")
    data["septic_patient"] = data["sepsis_onset_time"].notna()
    y = data["sepsis_label"].to_numpy().astype(int)
    p = data["p"].to_numpy()
    groups = data["patient_id"].to_numpy()

    # Cohort description
    span = data.groupby("patient_id")["timestamp"].agg(lambda s: (s.max() - s.min()).total_seconds())
    patient_days = float(span.sum() / 86400) or 1.0
    vital_cols = ("temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "gcs", "map")
    cohort = {
        "patients": int(data["patient_id"].nunique()),
        "observations": int(len(data)),
        "septic_patients": int(data.groupby("patient_id")["septic_patient"].first().sum()),
        "positive_rows": int(y.sum()),
        "row_positive_rate": _r(y.mean()),
        "observations_per_patient_median": float(data.groupby("patient_id").size().median()),
        "patient_days": round(patient_days, 1),
        "missing_fraction": {c: _r(data[c].isna().mean(), 3) for c in vital_cols + LAB_COLUMNS},
        "age_median": float(data.groupby("patient_id")["age_years"].first().median()),
        "sex": {str(k): int(v) for k, v in data.groupby("patient_id")["sex"].first().value_counts().items()},
    }

    # Discrimination, with age alone as a probe of demographic dependence
    age = data["age_years"].to_numpy().astype(float)
    boot = paired_cluster_bootstrap(groups, y, {"model": p, "age_only": age}, reference="model",
                                    metrics=("auroc", "auprc"), n_boot=n_boot, seed=seed % 100000)

    # Calibration
    calib = calibration(y, p)

    # Alerting: placeholder policy, model alone and the combined rule
    policy = threshold_policy(predictor.dual_thresholds)
    rules = [compute_scores({k: v for k, v in row.items() if pd.notna(v)})
             for row in data[list(vital_cols + LAB_COLUMNS)].to_dict("records")]
    rule_alert = np.array([bool(s.alert_flag) for s in rules])
    rule_level = [s.risk_level for s in rules]
    order = ["low", "moderate", "high", "critical"]
    m_level = [model_level(x, policy) for x in p]
    combined_level = [max(a, b, key=order.index) for a, b in zip(rule_level, m_level)]
    combined_alert = rule_alert | np.isin(combined_level, ["high", "critical"]) | (p > policy["alert_cutoff"])
    burden = [
        _alert_burden(f"model output >= {c}", p >= c, data, patient_days)
        for c in sorted(set(policy["model_bands"].values()) | {policy["alert_cutoff"]})
    ]
    burden.append(_alert_burden("rule-based alert alone", rule_alert, data, patient_days))
    burden.append(_alert_burden("deployed alert rule (rules OR level OR cut-off)", combined_alert, data, patient_days))
    alert_sources = {
        "alerts": int(combined_alert.sum()),
        "rules_only": int((combined_alert & rule_alert & (p <= policy["alert_cutoff"])
                           & ~np.isin(m_level, ["high", "critical"])).sum()),
        "model_only": int((combined_alert & ~rule_alert & ~np.isin(rule_level, ["high", "critical"])).sum()),
    }

    # Subgroups
    labs_present = data[list(LAB_COLUMNS)].notna().any(axis=1).to_numpy()
    first_obs = (data.groupby("patient_id").cumcount() == 0).to_numpy()
    bands = pd.cut(data["age_years"], [0, 44, 64, 200], labels=["18-44", "45-64", "65+"]).astype(str).to_numpy()
    subgroup_masks = {
        **{f"age {b}": bands == b for b in ("18-44", "45-64", "65+")},
        **{f"sex {s}": (data["sex"] == s).to_numpy() for s in ("F", "M")},
        "labs measured in the observation": labs_present,
        "no labs in the observation": ~labs_present,
        "first observation of the stay": first_obs,
        "later observations": ~first_obs,
    }
    subgroups = {
        name: {"rows": int(mask.sum()), "positive_rows": int(y[mask].sum()),
               "auroc": _r(_auc(y[mask], p[mask])), "mean_output": _r(p[mask].mean()) if mask.any() else None,
               "observed_rate": _r(y[mask].mean()) if mask.any() else None}
        for name, mask in subgroup_masks.items()
    }

    # Behaviour probes: does an abnormal finding raise the output?
    probes = {}
    for name, column, finding in BEHAVIOUR_PROBES:
        mask = finding(data[column]).to_numpy()
        if mask.any():
            delta = p - _counterfactual(predictor, df, column)
            probes[name] = {
                "rows": int(mask.sum()),
                "share_where_value_lowered_output": _r((delta[mask] < 0).mean()),
                "median_change_in_output": _r(float(np.median(delta[mask]))),
                "mean_change_in_output": _r(float(np.mean(delta[mask]))),
            }

    return {
        "claim_scope": "synthetic engineering evidence only; not clinical performance",
        "model": {**predictor.artifact_status.provenance(), "model_name": predictor.metadata["model_name"]},
        "cohort_generator": {"generator": f"sepsis_vitals.ml.synthetic_data.generate_dataset ({generator_profile} profile)",
                             "profile_options": PROFILES[generator_profile],
                             "n_patients": n_patients, "sepsis_prevalence": prevalence, "seed": seed,
                             "training_seed": (predictor.metadata.get("data_provenance") or {}).get("seed"),
                             "note": "Generated independently of the training data (different seed)."
                                     + (" The legacy generator has changed since the committed model was trained"
                                        " (M4 timestamp fix), so this cohort is close to, not identical with,"
                                        " the training distribution." if generator_profile == "legacy" else "")},
        "dataset": dataset_fingerprint([df]),
        "label": "current_state: 1 for observations at or after the generated onset time. No prediction horizon.",
        "cohort": cohort,
        "discrimination": boot,
        "calibration": {**calib, "mean_output": _r(p.mean()), "observed_rate": _r(y.mean()),
                        "reliability_by_decile": reliability(y, p)},
        "decision_curve": decision_curve(y, p),
        "threshold_policy": policy,
        "alert_burden": burden,
        "alert_sources": alert_sources,
        "subgroups": subgroups,
        "failure_probes": probes,
        "behaviour_gate": behaviour_gate(probes),
    }
