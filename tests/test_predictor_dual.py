"""Tests for dual operating point thresholds."""

import json
import numpy as np


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

        from sepsis_vitals.ml.artifacts import write_manifest
        write_manifest(tmp_path, "unvalidated")  # loader requires a verified manifest
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
            "feature_names": ["temperature", "heart_rate", "resp_rate"],
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

        from sepsis_vitals.ml.artifacts import write_manifest
        write_manifest(tmp_path, "unvalidated")  # loader requires a verified manifest
        from sepsis_vitals.ml.predictor import SepsisPredictor
        predictor = SepsisPredictor(str(tmp_path))
        predictor.load()

        info = predictor.model_info()
        assert "dual_thresholds" in info


class FixedProbability:
    """Stand-in model returning one probability (module level so it pickles)."""

    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob]] * len(X))


class TestRuleAlertsCannotBeSuppressed:
    """Regression: with dual thresholds the model level replaced the rule level."""

    def _predictor(self, tmp_path, prob):
        import joblib

        from sepsis_vitals.ml.artifacts import write_manifest
        from sepsis_vitals.ml.predictor import SepsisPredictor
        from sepsis_vitals.ml.trainer import prepare_features

        names = ["temperature", "heart_rate", "resp_rate"]
        joblib.dump(FixedProbability(prob), tmp_path / "sepsis_model.joblib")
        metadata = {
            "model_name": "Fixed", "version": "0.0.1", "feature_names": names,
            "needs_scaling": False,
            "dual_thresholds": {
                "continuous": {"threshold": 0.7}, "on_demand": {"threshold": 0.4},
            },
        }
        (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))
        write_manifest(tmp_path, "unvalidated")
        p = SepsisPredictor(str(tmp_path), state_dir=str(tmp_path / "state"))
        p.load()
        assert prepare_features  # imported for parity with the training pipeline
        return p

    def test_low_model_probability_keeps_rule_based_critical(self, tmp_path):
        p = self._predictor(tmp_path, prob=0.01)
        vitals = {"heart_rate": 135, "resp_rate": 30, "sbp": 82, "temperature": 39.6, "gcs": 12}
        result = p.predict(vitals, patient_id="rule-check").to_dict()
        assert result["rule_risk_level"] == "critical"
        assert result["model_risk_level"] == "low"
        assert result["risk_level"] == "critical"
        assert result["alert"] is True

    def test_model_can_raise_but_not_lower_the_level(self, tmp_path):
        p = self._predictor(tmp_path, prob=0.95)
        vitals = {"heart_rate": 80, "resp_rate": 16, "sbp": 124, "temperature": 37.0, "gcs": 15}
        result = p.predict(vitals, patient_id="raise-check").to_dict()
        assert result["rule_risk_level"] == "low"
        assert result["risk_level"] == "critical"
        assert result["provenance"]["validation_status"] == "unvalidated"
