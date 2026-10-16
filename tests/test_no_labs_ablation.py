"""Tests for the reproducible no-labs ablation report."""

import pytest


def test_build_comparison_report_uses_held_out_test_auroc_and_records_delta():
    """Catches reports that substitute validation AUROC or reverse the degradation delta."""
    from sepsis_vitals.ml.ablation import build_comparison_report

    full_result = {
        "test_metrics": {"test_auroc": 0.81, "test_auprc": 0.42},
        "feature_cols": ["heart_rate", "lactate"],
        "best_model": "GradientBoosting",
    }
    no_labs_result = {
        "test_metrics": {"test_auroc": 0.76, "test_auprc": 0.35},
        "feature_cols": ["heart_rate"],
        "best_model": "GradientBoosting",
    }
    provenance = {
        "source": "synthetic",
        "n_patients": 100,
        "seed": 42,
        "clinical_validation": "NONE — synthetic data only.",
    }

    report = build_comparison_report(full_result, no_labs_result, provenance)

    assert report["comparison"]["full"]["held_out_test_auroc"] == 0.81
    assert report["comparison"]["no_labs"]["held_out_test_auroc"] == 0.76
    assert report["auroc_delta_no_labs_minus_full"] == pytest.approx(-0.05)
    assert report["comparison"]["no_labs"]["n_features"] == 1
    assert report["data_provenance"] == provenance
    assert report["claim_scope"] == "synthetic development evidence only"


def test_build_comparison_report_requires_test_auroc():
    """Catches incomplete experiment output before it can become a public claim."""
    from sepsis_vitals.ml.ablation import build_comparison_report

    incomplete = {
        "test_metrics": {},
        "feature_cols": ["heart_rate"],
        "best_model": "GradientBoosting",
    }

    with pytest.raises(ValueError, match="held-out test AUROC"):
        build_comparison_report(incomplete, incomplete, {"source": "synthetic"})
