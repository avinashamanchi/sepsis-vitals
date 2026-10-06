"""Tests for the synthetic pipeline audit diagnostics."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def test_inference_style_blanks_history_features():
    """The audit must mirror how the live predictor builds single-observation features."""
    from sepsis_vitals.ml.pipeline_audit import inference_style

    features = pd.DataFrame({
        "heart_rate": [100.0], "heart_rate_delta": [5.0], "heart_rate_roll_mean": [95.0],
        "heart_rate_roll_std": [3.0], "obs_gap_min": [240.0], "age_years": [70.0],
    })
    X = inference_style(features, list(features.columns))
    assert np.isnan(X.loc[0, "heart_rate_delta"])
    assert np.isnan(X.loc[0, "heart_rate_roll_std"])
    assert np.isnan(X.loc[0, "obs_gap_min"])
    assert X.loc[0, "heart_rate_roll_mean"] == 100.0
    assert X.loc[0, "age_years"] == 70.0


def test_feature_groups_partition_labs_out_of_no_labs():
    from sepsis_vitals.ml.pipeline_audit import feature_groups

    cols = ["heart_rate", "lactate", "lactate_missing", "n_labs_missing", "age_years", "qsofa"]
    groups = feature_groups(cols)
    assert groups["no_labs"] == ["heart_rate", "age_years", "qsofa"]
    assert groups["demographics_and_comorbidities"] == ["age_years"]
    assert "lactate" not in groups["vitals_and_scores_only"]


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_run_audit_on_tiny_cohort_without_model():
    from sepsis_vitals.ml.pipeline_audit import run_audit

    report = run_audit(n_patients=300, seed=3, model_dir=None)
    assert report["claim_scope"] == "synthetic development evidence only"
    assert 0.0 <= report["generator"]["fraction_negative_time_gaps"] <= 1.0
    assert set(report["feature_groups"]) == {
        "demographics_and_comorbidities", "vitals_and_scores_only", "no_labs", "full",
    }
    assert "committed_model" not in report
