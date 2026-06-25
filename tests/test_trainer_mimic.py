"""Tests for MIMIC-demo training path and LOPOCV evaluation."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestLOPOCV:
    """Test leave-one-patient-out cross-validation."""

    def test_lopocv_returns_per_patient_scores(self):
        from sepsis_vitals.ml.trainer import lopocv_evaluate

        # 5 patients, 3 observations each, binary labels
        np.random.seed(42)
        n_patients = 5
        rows = []
        for pid in range(n_patients):
            for _ in range(3):
                rows.append({
                    "patient_id": pid,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "sepsis_label": 1 if pid < 2 else 0,
                })
        df = pd.DataFrame(rows)
        feature_cols = ["feature1", "feature2"]

        results = lopocv_evaluate(df, feature_cols, model_type="logistic")
        assert "per_patient_auroc" not in results  # Not meaningful per-patient
        assert "mean_auroc" in results or "auroc" in results
        assert "patient_predictions" in results
        assert len(results["patient_predictions"]) == n_patients

    def test_lopocv_with_gradient_boosting(self):
        from sepsis_vitals.ml.trainer import lopocv_evaluate

        np.random.seed(42)
        rows = []
        for pid in range(10):
            for _ in range(5):
                sep = 1 if pid < 3 else 0
                rows.append({
                    "patient_id": pid,
                    "feature1": np.random.randn() + sep * 2,
                    "feature2": np.random.randn() + sep,
                    "sepsis_label": sep,
                })
        df = pd.DataFrame(rows)
        feature_cols = ["feature1", "feature2"]

        results = lopocv_evaluate(df, feature_cols, model_type="gradient_boosting")
        assert "auroc" in results
        assert 0.0 <= results["auroc"] <= 1.0

    def test_lopocv_handles_single_class_fold(self):
        """LOPOCV should handle folds where held-out patient has only one class."""
        from sepsis_vitals.ml.trainer import lopocv_evaluate

        np.random.seed(42)
        rows = []
        for pid in range(6):
            label = 1 if pid < 2 else 0
            for _ in range(3):
                rows.append({
                    "patient_id": pid,
                    "feature1": np.random.randn() + label * 2,
                    "feature2": np.random.randn(),
                    "sepsis_label": label,
                })
        df = pd.DataFrame(rows)

        # Should not crash even though each fold has single-class held-out
        results = lopocv_evaluate(df, ["feature1", "feature2"], model_type="logistic")
        assert "patient_predictions" in results
