"""Generator options and the profile-evaluation protocol (engineering evidence)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn missing")


def test_horizon_labels_need_an_explicit_horizon():
    from sepsis_vitals.ml.synthetic_data import generate_dataset

    with pytest.raises(ValueError, match="horizon_hours"):
        generate_dataset(n_patients=5, label_mode="onset_within_horizon")


def test_horizon_mode_scores_only_pre_onset_rows():
    from sepsis_vitals.ml.synthetic_data import generate_dataset

    df = generate_dataset(n_patients=300, seed=3, label_mode="onset_within_horizon",
                          horizon_hours=12, include_onset_time=True)
    assert not (df["timestamp"] >= df["sepsis_onset_time"]).any()
    lead_h = (df["sepsis_onset_time"] - df["timestamp"]).dt.total_seconds() / 3600
    clean = df[lead_h.notna()]
    # before label noise, positives are exactly the rows within the horizon
    assert ((lead_h[clean.index] <= 12) | (clean["sepsis_label"] == 0)).mean() > 0.95


def test_decoupling_removes_the_age_signal():
    from sklearn.metrics import roc_auc_score

    from sepsis_vitals.ml.synthetic_data import generate_dataset

    def age_auc(**kw):
        df = generate_dataset(n_patients=2500, seed=11, **kw)
        pts = df.groupby("patient_id").agg(age=("age_years", "first"), y=("sepsis_label", "max"))
        return roc_auc_score(pts["y"], pts["age"])

    assert age_auc() > 0.65
    assert abs(age_auc(age_effect=0.0, comorbidity_effect=0.0) - 0.5) < 0.06


def test_calibration_of_a_calibrated_predictor_is_near_identity():
    from sepsis_vitals.ml.profile_evaluation import calibration

    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    cal = calibration(y, p)
    assert abs(cal["slope"] - 1) < 0.1 and abs(cal["intercept"]) < 0.1


def test_profile_evaluation_reports_the_protocol_fields():
    from sepsis_vitals.ml.profile_evaluation import PROFILES, evaluate_profile

    r = evaluate_profile("decoupled", n_patients=400, seed=2, n_bootstrap=10, **PROFILES["decoupled"])
    for key in ("auroc", "auroc_95ci", "auprc", "brier", "calibration", "operating_point",
                "auroc_without_labs", "auroc_demographics_only", "subgroups"):
        assert key in r
    assert r["split"].startswith("patient-level, temporal")
