"""
tests/test_uncertainty.py — paired patient-level bootstrap (M6) and the
ablation pipeline's held-out predictions (N38).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]


def _fixture(n_patients: int = 60, rows: int = 4, seed: int = 3):
    """Patients with several correlated rows; arm 'good' separates, arm 'noise' does not."""
    rng = np.random.default_rng(seed)
    groups, y, good, noise = [], [], [], []
    for pid in range(n_patients):
        label = int(pid % 3 == 0)
        for _ in range(rows):
            groups.append(f"p{pid}")
            y.append(label)
            good.append(0.7 * label + 0.3 * rng.random())
            noise.append(rng.random())
    return groups, y, good, noise


def test_intervals_are_reproducible_and_contain_the_estimates():
    from sepsis_vitals.ml.uncertainty import paired_cluster_bootstrap

    groups, y, good, noise = _fixture()
    a = paired_cluster_bootstrap(groups, y, {"noise": noise, "good": good}, n_boot=200, seed=11)
    b = paired_cluster_bootstrap(groups, y, {"noise": noise, "good": good}, n_boot=200, seed=11)
    c = paired_cluster_bootstrap(groups, y, {"noise": noise, "good": good}, n_boot=200, seed=12)
    assert a == b and a != c
    for arm in ("noise", "good"):
        for metric in ("auroc", "auprc"):
            est, (lo, hi) = a["arms"][arm][metric]["estimate"], a["arms"][arm][metric]["ci"]
            assert lo <= est <= hi
    diff = a["differences_vs_reference"]["good"]["auroc"]
    assert diff["estimate"] > 0.3 and diff["ci"][0] > 0          # clearly better, CI excludes 0
    assert a["n_patients"] == 60 and a["n_rows"] == 240 and a["n_positive_rows"] == 80
    assert a["reference_arm"] == "noise" and a["valid_replicates"] == 200 and a["reliable"]


def test_identical_arms_have_a_zero_width_difference():
    from sepsis_vitals.ml.uncertainty import paired_cluster_bootstrap

    groups, y, good, _ = _fixture()
    r = paired_cluster_bootstrap(groups, y, {"a": good, "b": list(good)}, n_boot=100, seed=1)
    assert r["differences_vs_reference"]["b"]["auroc"]["ci"] == [0.0, 0.0]


def test_patients_not_rows_are_resampled_and_degenerate_replicates_are_counted():
    """Two patients, one per class: a patient-level draw is single-class about
    half the time (a row-level draw almost never would be)."""
    from sepsis_vitals.ml.uncertainty import paired_cluster_bootstrap

    groups = ["a"] * 5 + ["b"] * 5
    y = [1] * 5 + [0] * 5
    r = paired_cluster_bootstrap(groups, y, {"m": [0.9] * 5 + [0.1] * 5}, n_boot=400, seed=5)
    assert 120 < r["degenerate_replicates"] < 280
    assert r["valid_replicates"] + r["degenerate_replicates"] == 400
    assert r["reliable"] is False


@pytest.mark.parametrize("kwargs,message", [
    ({"predictions": {"m": [0.1, 0.2]}}, "predictions for"),
    ({"predictions": {"m": [0.1, float("nan"), 0.3, 0.4]}}, "non-finite"),
    ({"y_true": [1, 1, 1, 1]}, "single class"),
    ({"reference": "other"}, "unknown reference"),
    ({"metrics": ("brier",)}, "unknown metrics"),
])
def test_invalid_inputs_are_rejected(kwargs, message):
    from sepsis_vitals.ml.uncertainty import paired_cluster_bootstrap

    args = {"groups": ["a", "a", "b", "b"], "y_true": [0, 1, 0, 1], "predictions": {"m": [0.1, 0.9, 0.2, 0.8]}}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        paired_cluster_bootstrap(args.pop("groups"), args.pop("y_true"), args.pop("predictions"), **args)


def _arm(groups, y, p, auroc):
    return {"best_model": "M", "feature_cols": ["x"], "test_metrics": {"test_auroc": auroc, "test_auprc": 0.5},
            "test_predictions": {"patient_id": list(groups), "y_true": list(y), "y_prob": list(p)}}


def test_ablation_report_gets_paired_intervals_and_rejects_unpaired_arms():
    from sklearn.metrics import roc_auc_score

    from sepsis_vitals.ml.ablation import add_uncertainty, build_comparison_report

    groups, y, good, noise = _fixture()
    full = _arm(groups, y, good, roc_auc_score(y, good))
    no_labs = _arm(groups, y, noise, roc_auc_score(y, noise))
    report = add_uncertainty(build_comparison_report(full, no_labs, {"n_patients": 60, "seed": 1}),
                             full, no_labs, n_boot=100, seed=2)
    unc = report["uncertainty"]
    assert unc["reference_arm"] == "full" and unc["differences_vs_reference"]["no_labs"]["auroc"]["ci"][1] < 0
    assert any("training variability" in item for item in report["limitations"])

    spec = importlib.util.spec_from_file_location("ablation_script", ROOT / "scripts" / "run_no_labs_ablation.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    markdown = script._markdown_report(report)
    assert "Held-out AUROC (95% CI)" in markdown and "## Uncertainty" in markdown and "95% CI" in markdown

    shuffled = _arm(list(reversed(groups)), y, noise, roc_auc_score(y, noise))
    with pytest.raises(ValueError, match="same held-out rows"):
        add_uncertainty(build_comparison_report(full, shuffled, {}), full, shuffled, n_boot=10)
    wrong = _arm(groups, y, noise, 0.99)
    with pytest.raises(ValueError, match="does not match"):
        add_uncertainty(build_comparison_report(full, wrong, {}), full, wrong, n_boot=10)


def test_profile_evaluation_reports_paired_intervals_for_probes():
    from sepsis_vitals.ml.profile_evaluation import PROFILES, evaluate_profile

    r = evaluate_profile("decoupled", n_patients=300, seed=4, n_bootstrap=50, **PROFILES["decoupled"])
    for key in ("auroc_95ci", "auroc_without_labs_95ci", "auroc_demographics_only_95ci"):
        lo, hi = r[key]
        assert 0.0 <= lo <= hi <= 1.0, key
    assert set(r["uncertainty"]["differences_vs_reference"]) == {"no_labs", "demographics_only"}


def test_pipeline_metrics_match_its_held_out_predictions_for_scaled_models(tmp_path, monkeypatch):
    """N38 regression: the evaluation report scaled the test features twice when
    the selected model used a scaler (logistic regression)."""
    import sys

    from sklearn.metrics import roc_auc_score

    sys.path.insert(0, str(ROOT))
    import retrain
    from sepsis_vitals.ml import trainer

    def pick_logistic(results):
        return next(r for r in results if r.scaler is not None)

    monkeypatch.setattr(trainer, "select_best_model", pick_logistic)
    train_df, val_df, test_df, provenance = retrain.load_synthetic_data(n_patients=120, prevalence=0.3, seed=8)
    result = retrain.run_pipeline(train_df, val_df, test_df, provenance, output_dir=str(tmp_path / "m"),
                                  cv_folds=2, skip_shap=True)
    preds = result["test_predictions"]
    assert len(preds["y_true"]) == len(test_df)
    recomputed = roc_auc_score(preds["y_true"], preds["y_prob"])
    assert result["test_metrics"]["test_auroc"] == pytest.approx(recomputed, abs=1e-9)
