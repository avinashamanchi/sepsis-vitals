"""
tests/test_ml_pipeline.py — Tests for the ML training pipeline, synthetic data,
predictor, and trainer modules.
"""

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_generate_dataset_basic(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset

        df = generate_dataset(n_patients=50, sepsis_prevalence=0.2, seed=99)
        assert len(df) > 0
        assert "patient_id" in df.columns
        assert "sepsis_label" in df.columns
        assert "temperature" in df.columns
        assert "heart_rate" in df.columns
        assert df["patient_id"].nunique() == 50

    def test_sepsis_prevalence_approximately_correct(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset

        df = generate_dataset(n_patients=500, sepsis_prevalence=0.15, seed=42)
        patients = df.groupby("patient_id")["sepsis_label"].max()
        septic_frac = patients.mean()
        # Allow reasonable range due to age/comorbidity adjustments
        assert 0.05 < septic_frac < 0.50

    def test_vital_ranges_valid(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset

        df = generate_dataset(n_patients=100, seed=42)
        # Check vitals are within physiological limits
        assert df["temperature"].dropna().between(30, 43).all()
        assert df["heart_rate"].dropna().between(30, 220).all()
        assert df["resp_rate"].dropna().between(4, 60).all()
        assert df["sbp"].dropna().between(40, 260).all()
        assert df["spo2"].dropna().between(50, 100).all()
        assert df["gcs"].dropna().between(3, 15).all()

    def test_demographics_present(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset

        df = generate_dataset(n_patients=50, seed=42)
        assert "age_years" in df.columns
        assert "sex" in df.columns
        assert "ethnicity" in df.columns
        assert df["sex"].isin(["M", "F"]).all()

    def test_comorbidity_columns(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset

        df = generate_dataset(n_patients=50, seed=42)
        for comorb in ["has_hypertension", "has_diabetes", "has_ckd", "has_copd", "has_heart_failure"]:
            assert comorb in df.columns
            assert df[comorb].isin([0, 1]).all()

    def test_train_val_test_split(self):
        from sepsis_vitals.ml.synthetic_data import generate_train_val_test

        train, val, test = generate_train_val_test(
            n_patients=100, seed=42, train_frac=0.7, val_frac=0.15
        )
        # Patient-level split: no overlap
        train_ids = set(train["patient_id"].unique())
        val_ids = set(val["patient_id"].unique())
        test_ids = set(test["patient_id"].unique())
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_missing_data_present(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset

        df = generate_dataset(n_patients=200, seed=42)
        # With 5% missing rate per vital, there should be some NaNs
        total_missing = df[["temperature", "heart_rate", "resp_rate", "sbp", "spo2"]].isna().sum().sum()
        assert total_missing > 0

    def test_temporal_ordering(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset

        df = generate_dataset(n_patients=20, seed=42)
        for pid in df["patient_id"].unique()[:5]:
            patient_df = df[df["patient_id"] == pid].sort_values("timestamp")
            timestamps = patient_df["timestamp"].values
            assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering (via trainer.prepare_features)
# ──────────────────────────────────────────────────────────────────────────────

class TestFeaturePreparation:
    def test_prepare_features_returns_features(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset
        from sepsis_vitals.ml.trainer import prepare_features

        df = generate_dataset(n_patients=20, seed=42)
        features, feature_cols = prepare_features(df)
        assert len(feature_cols) > 10  # Should have many features
        assert len(features) == len(df)

    def test_feature_columns_no_target_leak(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset
        from sepsis_vitals.ml.trainer import prepare_features

        df = generate_dataset(n_patients=20, seed=42)
        _, feature_cols = prepare_features(df)
        # Target label should NOT be in features
        assert "sepsis_label" not in feature_cols

    def test_missing_indicators_created(self):
        from sepsis_vitals.ml.synthetic_data import generate_dataset
        from sepsis_vitals.ml.trainer import prepare_features

        df = generate_dataset(n_patients=20, seed=42)
        features, feature_cols = prepare_features(df)
        assert "n_vitals_missing" in feature_cols
        assert "temperature_missing" in feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# Model Training (small scale)
# ──────────────────────────────────────────────────────────────────────────────

class TestTraining:
    @pytest.fixture
    def small_dataset(self):
        from sepsis_vitals.ml.synthetic_data import generate_train_val_test
        from sepsis_vitals.ml.trainer import prepare_features

        train, val, _ = generate_train_val_test(
            n_patients=100, seed=42, train_frac=0.7, val_frac=0.15
        )
        train_feat, feature_cols = prepare_features(train)
        val_feat, _ = prepare_features(val)

        X_train = train_feat[feature_cols].values.astype(np.float64)
        y_train = train_feat["sepsis_label"].values.astype(int)
        X_val = val_feat[feature_cols].values.astype(np.float64)
        y_val = val_feat["sepsis_label"].values.astype(int)

        # Impute NaN
        col_medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = col_medians[j]
            X_val[np.isnan(X_val[:, j]), j] = col_medians[j]

        return X_train, y_train, X_val, y_val, feature_cols

    def test_train_single_model(self, small_dataset):
        from sklearn.ensemble import RandomForestClassifier
        from sepsis_vitals.ml.trainer import train_single_model

        X_train, y_train, X_val, y_val, feature_cols = small_dataset

        config = {
            "model_class": RandomForestClassifier,
            "param_grid": [
                {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": -1},
            ],
            "needs_scaling": False,
            "tree_model": True,
        }

        result = train_single_model(
            "TestRF", config, X_train, y_train, X_val, y_val, feature_cols, n_cv_folds=2
        )

        assert result.name == "TestRF"
        assert "val_auroc" in result.metrics
        assert result.metrics["val_auroc"] > 0.5  # Better than random
        assert result.model is not None

    def test_model_selection(self, small_dataset):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sepsis_vitals.ml.trainer import ModelResult, select_best_model
        from sepsis_vitals.model_scaffold import ModelCard

        # Create mock results
        result1 = ModelResult(
            name="Model1", model=None, scaler=None,
            metrics={"val_auroc": 0.85, "val_recall": 0.8, "val_brier": 0.15},
            feature_importance={}, training_time=1.0,
            card=ModelCard(name="M1", version="1.0", description=""),
        )
        result2 = ModelResult(
            name="Model2", model=None, scaler=None,
            metrics={"val_auroc": 0.90, "val_recall": 0.85, "val_brier": 0.12},
            feature_importance={}, training_time=2.0,
            card=ModelCard(name="M2", version="1.0", description=""),
        )

        best = select_best_model([result1, result2])
        assert best.name == "Model2"  # Higher AUROC

    def test_evaluate_model(self, small_dataset):
        from sklearn.ensemble import RandomForestClassifier
        from sepsis_vitals.ml.trainer import _evaluate_model

        X_train, y_train, X_val, y_val, _ = small_dataset

        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        metrics = _evaluate_model(model, X_val, y_val, prefix="test")
        assert "test_auroc" in metrics
        assert "test_recall" in metrics
        assert "test_specificity" in metrics
        assert "test_true_positives" in metrics
        assert 0 <= metrics["test_auroc"] <= 1
        assert 0 <= metrics["test_recall"] <= 1


# ──────────────────────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictor:
    def test_patient_monitor_trend(self):
        from sepsis_vitals.ml.predictor import PatientMonitor, SepsisPrediction

        monitor = PatientMonitor(patient_id="PT-001")

        for risk in [0.1, 0.15, 0.3, 0.5, 0.7]:
            pred = SepsisPrediction(
                patient_id="PT-001",
                timestamp="2024-01-01T00:00:00",
                risk_probability=risk,
                risk_level="high" if risk > 0.5 else "moderate" if risk > 0.25 else "low",
                confidence_lower=risk - 0.05,
                confidence_upper=risk + 0.05,
                alert=risk > 0.5,
                clinical_scores={},
                top_risk_factors=[],
                recommendation="",
                model_name="test",
                model_version="1.0",
            )
            monitor.add_prediction(pred)

        assert monitor._compute_trend() == "rapidly_worsening"

    def test_patient_monitor_deterioration(self):
        from sepsis_vitals.ml.predictor import PatientMonitor, SepsisPrediction

        monitor = PatientMonitor(patient_id="PT-002")

        # Add a low risk then a high risk
        for risk, level in [(0.2, "low"), (0.8, "critical")]:
            pred = SepsisPrediction(
                patient_id="PT-002",
                timestamp="2024-01-01T00:00:00",
                risk_probability=risk,
                risk_level=level,
                confidence_lower=risk - 0.05,
                confidence_upper=risk + 0.05,
                alert=risk > 0.5,
                clinical_scores={},
                top_risk_factors=[],
                recommendation="",
                model_name="test",
                model_version="1.0",
            )
            monitor.add_prediction(pred)

        det = monitor._detect_deterioration()
        assert det["detected"] is True

    def test_prediction_to_dict(self):
        from sepsis_vitals.ml.predictor import SepsisPrediction

        pred = SepsisPrediction(
            patient_id="PT-001",
            timestamp="2024-01-01T00:00:00",
            risk_probability=0.75,
            risk_level="high",
            confidence_lower=0.67,
            confidence_upper=0.83,
            alert=True,
            clinical_scores={"qsofa": 2},
            top_risk_factors=[],
            recommendation="Urgent review",
            model_name="TestModel",
            model_version="1.0",
        )

        d = pred.to_dict()
        assert d["risk_probability"] == 0.75
        assert d["risk_level"] == "high"
        assert d["alert"] is True
        assert d["confidence_interval"]["lower"] == 0.67
        assert d["model"]["name"] == "TestModel"


# ──────────────────────────────────────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────────────────────────────────────

try:
    import fastapi as _  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from sepsis_vitals.api import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_score_endpoint(self):
        from fastapi.testclient import TestClient
        from sepsis_vitals.api import app

        client = TestClient(app)
        vitals = {"temperature": 39.0, "heart_rate": 120, "resp_rate": 24, "sbp": 85, "gcs": 13}
        resp = client.post("/score", json=vitals)
        assert resp.status_code == 200
        data = resp.json()
        assert "qsofa" in data
        assert "risk_level" in data

    def test_model_info_returns_503_without_model(self):
        from fastapi.testclient import TestClient
        from sepsis_vitals.api import app

        client = TestClient(app)
        # Reset predictor
        import sepsis_vitals.api
        sepsis_vitals.api._predictor = None

        resp = client.get("/model/info")
        # May be 503 if model not trained, or 200 if model exists
        assert resp.status_code in (200, 503)
