"""Tests for ensemble module (gated at ≥500 patients)."""

import numpy as np
import pytest
from unittest.mock import MagicMock


class TestEnsembleGating:
    """Test that ensemble activates only at ≥500 patients."""

    def test_should_use_ensemble_below_threshold(self):
        from sepsis_vitals.ml.ensemble import should_use_ensemble

        assert should_use_ensemble(100) is False
        assert should_use_ensemble(499) is False

    def test_should_use_ensemble_at_threshold(self):
        from sepsis_vitals.ml.ensemble import should_use_ensemble

        assert should_use_ensemble(500) is True
        assert should_use_ensemble(1000) is True


class TestEnsemblePredict:
    """Test ensemble prediction with mocked base models."""

    def test_ensemble_predict_averages_base_models(self):
        from sepsis_vitals.ml.ensemble import SepsisEnsemble

        # Create mock base models
        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.3, 0.7]])
        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.5, 0.5]])
        model3 = MagicMock()
        model3.predict_proba.return_value = np.array([[0.4, 0.6]])

        ensemble = SepsisEnsemble(
            base_models=[model1, model2, model3],
            base_model_names=["m1", "m2", "m3"],
            feature_names=["f1", "f2"],
        )

        X = np.array([[1.0, 2.0]])
        prob = ensemble.predict(X)
        # Average of 0.7, 0.5, 0.6 = 0.6
        assert abs(prob - 0.6) < 0.01

    def test_ensemble_predict_with_uncertainty(self):
        from sepsis_vitals.ml.ensemble import SepsisEnsemble

        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.2, 0.8]])
        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.6, 0.4]])

        ensemble = SepsisEnsemble(
            base_models=[model1, model2],
            base_model_names=["m1", "m2"],
            feature_names=["f1"],
        )

        X = np.array([[1.0]])
        prob, ci_lower, ci_upper = ensemble.predict_with_uncertainty(X)
        assert ci_lower < prob < ci_upper
        assert ci_lower >= 0.0
        assert ci_upper <= 1.0

    def test_ensemble_feature_importance(self):
        from sepsis_vitals.ml.ensemble import SepsisEnsemble

        model1 = MagicMock()
        model1.feature_importances_ = np.array([0.6, 0.4])
        model2 = MagicMock()
        model2.feature_importances_ = np.array([0.3, 0.7])

        ensemble = SepsisEnsemble(
            base_models=[model1, model2],
            base_model_names=["m1", "m2"],
            feature_names=["hr", "temp"],
        )

        importance = ensemble.feature_importance()
        assert "hr" in importance
        assert "temp" in importance
        # Average of importances
        assert abs(importance["hr"] - 0.45) < 0.01
        assert abs(importance["temp"] - 0.55) < 0.01


class TestEnsembleSerialization:
    """Test ensemble save/load."""

    def test_ensemble_metadata(self):
        from sepsis_vitals.ml.ensemble import SepsisEnsemble

        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.3, 0.7]])

        ensemble = SepsisEnsemble(
            base_models=[model1],
            base_model_names=["GradientBoosting"],
            feature_names=["f1", "f2"],
        )

        meta = ensemble.metadata()
        assert meta["model_type"] == "ensemble"
        assert meta["n_base_models"] == 1
        assert meta["base_model_names"] == ["GradientBoosting"]
        assert meta["min_patients_for_ensemble"] == 500
