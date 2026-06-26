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


class TestPredictorDualMode:
    """Test SepsisPredictor with dual operating points."""

    def test_predictor_loads_dual_thresholds(self, tmp_path):
        """SepsisPredictor should load dual_thresholds from metadata."""
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier

        # Create a minimal trained model
        model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
        X = np.random.randn(50, 3)
        y = np.array([0]*40 + [1]*10)
        model.fit(X, y)

        # Save model artifacts
        joblib.dump(model, tmp_path / "sepsis_model.joblib")

        metadata = {
            "model_name": "GradientBoosting",
            "version": "2.0.0",
            "feature_names": ["temperature", "heart_rate", "resp_rate"],
            "needs_scaling": False,
            "is_calibrated": False,
            "metrics": {"val_auroc": 0.85},
            "feature_importance": {"temperature": 0.5, "heart_rate": 0.3, "resp_rate": 0.2},
            "dual_thresholds": {
                "continuous": {"threshold": 0.7, "target_specificity": 0.99,
                               "achieved_specificity": 0.99, "sensitivity": 0.4},
                "on_demand": {"threshold": 0.4, "target_specificity": 0.95,
                              "achieved_specificity": 0.95, "sensitivity": 0.7},
            },
        }
        with open(tmp_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(tmp_path / "imputation_medians.json", "w") as f:
            json.dump({"temperature": 37.0, "heart_rate": 80.0, "resp_rate": 18.0}, f)

        from sepsis_vitals.ml.predictor import SepsisPredictor
        predictor = SepsisPredictor(str(tmp_path))
        predictor.load()

        assert predictor.dual_thresholds is not None
        assert predictor.dual_thresholds["continuous"]["threshold"] == 0.7

    def test_predictor_model_info_includes_thresholds(self, tmp_path):
        """model_info() should include dual threshold data."""
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
        X = np.random.randn(50, 3)
        y = np.array([0]*40 + [1]*10)
        model.fit(X, y)

        joblib.dump(model, tmp_path / "sepsis_model.joblib")
        metadata = {
            "model_name": "GradientBoosting",
            "version": "2.0.0",
            "feature_names": ["f1", "f2", "f3"],
            "needs_scaling": False,
            "is_calibrated": False,
            "metrics": {"val_auroc": 0.85},
            "feature_importance": {},
            "dual_thresholds": {
                "continuous": {"threshold": 0.7, "target_specificity": 0.99,
                               "achieved_specificity": 0.99, "sensitivity": 0.4},
                "on_demand": {"threshold": 0.4, "target_specificity": 0.95,
                              "achieved_specificity": 0.95, "sensitivity": 0.7},
            },
        }
        with open(tmp_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(tmp_path / "imputation_medians.json", "w") as f:
            json.dump({}, f)

        from sepsis_vitals.ml.predictor import SepsisPredictor
        predictor = SepsisPredictor(str(tmp_path))
        predictor.load()

        info = predictor.model_info()
        assert "dual_thresholds" in info
