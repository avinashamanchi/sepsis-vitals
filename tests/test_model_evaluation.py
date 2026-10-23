"""
tests/test_model_evaluation.py — the committed-model evaluation report covers
every required section and labels itself as synthetic evidence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None or not (MODELS / "manifest.json").exists(),
    reason="scikit-learn or committed model artifacts unavailable",
)


def test_evaluation_has_every_section_and_is_scoped_to_synthetic_data():
    from sepsis_vitals.ml.model_evaluation import evaluate_committed_model

    r = evaluate_committed_model(str(MODELS), n_patients=200, seed=3, n_boot=20)
    assert "synthetic" in r["claim_scope"] and "not clinical" in r["claim_scope"]
    for key in ("cohort", "label", "dataset", "discrimination", "calibration", "decision_curve",
                "alert_burden", "alert_sources", "subgroups", "failure_probes", "threshold_policy"):
        assert key in r, key
    auroc = r["discrimination"]["arms"]["model"]["auroc"]["estimate"]
    assert 0.0 <= auroc <= 1.0
    assert r["threshold_policy"]["status"].startswith("engineering-placeholder")
    assert {b["rule"] for b in r["alert_burden"]} >= {"rule-based alert alone",
                                                      "deployed alert rule (rules OR level OR cut-off)"}
    assert len(r["dataset"]["sha256"]) == 64


def test_decision_curve_matches_hand_computation():
    from sepsis_vitals.ml.model_evaluation import decision_curve

    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.3, 0.1])
    row = decision_curve(y, p, thresholds=(0.5,))[0]
    # flagged: rows 0 and 1 -> TP 1, FP 1; weight 1
    assert row["net_benefit_model"] == pytest.approx(1 / 4 - 1 / 4)
    assert row["net_benefit_all"] == pytest.approx(0.5 - 0.5)
