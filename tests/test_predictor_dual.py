"""Tests for dual operating point thresholds."""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestDualThresholds:
    """Test dual operating point computation and application."""

    def test_compute_dual_thresholds(self):
        from sepsis_vitals.ml.trainer import compute_dual_thresholds

        np.random.seed(42)
        y_true = np.array([0]*80 + [1]*20)
        y_prob = np.concatenate([
            np.random.beta(2, 5, 80),  # negatives: low scores
            np.random.beta(5, 2, 20),  # positives: high scores
        ])

        thresholds = compute_dual_thresholds(y_true, y_prob)
        assert "continuous" in thresholds
        assert "on_demand" in thresholds
        assert thresholds["continuous"]["target_specificity"] == 0.99
        assert thresholds["on_demand"]["target_specificity"] == 0.95
        assert thresholds["continuous"]["threshold"] >= thresholds["on_demand"]["threshold"]

    def test_classify_risk_continuous_mode(self):
        from sepsis_vitals.ml.predictor import classify_risk_dual

        thresholds = {
            "continuous": {"threshold": 0.7, "target_specificity": 0.99},
            "on_demand": {"threshold": 0.4, "target_specificity": 0.95},
        }

        # Below on-demand threshold -> low
        assert classify_risk_dual(0.3, thresholds, mode="continuous") == "low"
        # Above on-demand but below continuous -> moderate
        assert classify_risk_dual(0.5, thresholds, mode="continuous") == "moderate"
        # Above continuous threshold -> high
        assert classify_risk_dual(0.75, thresholds, mode="continuous") == "high"

    def test_classify_risk_on_demand_mode(self):
        from sepsis_vitals.ml.predictor import classify_risk_dual

        thresholds = {
            "continuous": {"threshold": 0.7, "target_specificity": 0.99},
            "on_demand": {"threshold": 0.4, "target_specificity": 0.95},
        }

        # Below on-demand -> low
        assert classify_risk_dual(0.3, thresholds, mode="on_demand") == "low"
        # Above on-demand -> moderate or higher
        result = classify_risk_dual(0.5, thresholds, mode="on_demand")
        assert result in ("moderate", "high")

    def test_threshold_stored_in_metadata(self):
        from sepsis_vitals.ml.trainer import compute_dual_thresholds

        y_true = np.array([0]*50 + [1]*10)
        y_prob = np.concatenate([
            np.random.beta(2, 5, 50),
            np.random.beta(5, 2, 10),
        ])
        thresholds = compute_dual_thresholds(y_true, y_prob)

        # Must be JSON-serializable
        json_str = json.dumps(thresholds)
        parsed = json.loads(json_str)
        assert parsed["continuous"]["threshold"] == thresholds["continuous"]["threshold"]
