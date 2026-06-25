# Layer 2: ML Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the sepsis model on real MIMIC-IV Demo data using Sepsis-3 labels from Layer 1, with a single heavily-regularized GradientBoostingClassifier for the 100-patient demo dataset, LogisticRegression as interpretable baseline, LOPOCV evaluation, dual operating points (≥99% specificity for continuous monitoring, ≥95% for on-demand), and an ensemble module ready to activate at ≥500 patients.

**Architecture:** Modify `train.py` to support `--data-source mimic-demo` alongside existing synthetic mode. Add `ensemble.py` (built and tested but gated). Modify `predictor.py` to support dual operating points and on-demand SHAP. Add `--data-source` CLI flag, LOPOCV evaluator, and dual-threshold metadata to model artifacts.

**Tech Stack:** Python 3.9+, scikit-learn (GradientBoostingClassifier, LogisticRegression, CalibratedClassifierCV), pandas, numpy, joblib, pytest. LightGBM/XGBoost used when available but not required (missing libomp on this system).

**Environment constraint:** LightGBM and XGBoost fail to import on this machine (missing `libomp.dylib`). The plan uses sklearn's `GradientBoostingClassifier` as the primary model. The ensemble module wraps LightGBM/XGBoost with import guards so it works when they're installed.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/sepsis_vitals/ml/ensemble.py` | Create | Ensemble wrapper: multi-model stacking, gated at ≥500 patients |
| `src/sepsis_vitals/ml/trainer.py` | Modify | Add LOPOCV, dual operating points, MIMIC-demo training path |
| `src/sepsis_vitals/train.py` | Modify | Add `--data-source mimic-demo` and `--ensemble` CLI flags |
| `src/sepsis_vitals/ml/predictor.py` | Modify | Dual thresholds (`mode="continuous"` vs `"on_demand"`), on-demand SHAP |
| `tests/test_ensemble.py` | Create | Tests for ensemble module (with mocked base models) |
| `tests/test_trainer_mimic.py` | Create | Tests for MIMIC-demo training path and LOPOCV |
| `tests/test_predictor_dual.py` | Create | Tests for dual operating points |

---

### Task 1: LOPOCV Evaluator

**Files:**
- Modify: `src/sepsis_vitals/ml/trainer.py`
- Create: `tests/test_trainer_mimic.py`

- [ ] **Step 1: Write failing tests for LOPOCV**

Create `tests/test_trainer_mimic.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_trainer_mimic.py::TestLOPOCV -v`
Expected: FAIL — `cannot import name 'lopocv_evaluate'`

- [ ] **Step 3: Implement LOPOCV evaluator**

Add to `src/sepsis_vitals/ml/trainer.py` (after `select_best_model`):

```python
def lopocv_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = "gradient_boosting",
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Leave-one-patient-out cross-validation.

    More appropriate than k-fold CV for small clinical datasets (<200 patients)
    because it prevents data leakage between observations from the same patient.

    Parameters
    ----------
    df : DataFrame
        Must have 'patient_id', 'sepsis_label', and all feature_cols.
    feature_cols : list of str
        Feature column names.
    model_type : str
        'gradient_boosting' or 'logistic'.
    model_params : dict, optional
        Override default model hyperparameters.

    Returns
    -------
    dict with keys: auroc, auprc, brier, patient_predictions, n_patients
    """
    if model_params is None:
        if model_type == "gradient_boosting":
            model_params = {
                "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
                "min_samples_leaf": 20, "subsample": 0.8, "random_state": 42,
            }
        else:
            model_params = {"C": 0.1, "max_iter": 2000, "random_state": 42, "solver": "lbfgs"}

    patient_ids = df["patient_id"].unique()
    all_y_true = []
    all_y_prob = []
    patient_predictions = {}

    for pid in patient_ids:
        test_mask = df["patient_id"] == pid
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feature_cols].values.astype(np.float64)
        y_train = df.loc[train_mask, "sepsis_label"].values.astype(int)
        X_test = df.loc[test_mask, feature_cols].values.astype(np.float64)
        y_test = df.loc[test_mask, "sepsis_label"].values.astype(int)

        # Impute NaN with training medians
        col_medians = np.nanmedian(X_train, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = col_medians[j]
            X_test[np.isnan(X_test[:, j]), j] = col_medians[j]

        # Skip if training set has only one class
        if len(np.unique(y_train)) < 2:
            continue

        if model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**model_params)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = LogisticRegression(**model_params)

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        patient_predictions[str(pid)] = {
            "y_true": y_test.tolist(),
            "y_prob": y_prob.tolist(),
        }

    # Compute aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    results = {
        "n_patients": len(patient_ids),
        "n_evaluated": len(patient_predictions),
        "patient_predictions": patient_predictions,
    }

    if len(np.unique(all_y_true)) >= 2 and len(all_y_true) > 0:
        results["auroc"] = float(roc_auc_score(all_y_true, all_y_prob))
        results["auprc"] = float(average_precision_score(all_y_true, all_y_prob))
        results["brier"] = float(brier_score_loss(all_y_true, all_y_prob))

        # Sensitivity at operating points
        fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
        for target_spec in [0.95, 0.99]:
            target_fpr = 1 - target_spec
            idx = np.searchsorted(fpr, target_fpr)
            sens = float(tpr[idx]) if idx < len(tpr) else float(tpr[-1])
            thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
            results[f"sensitivity_at_{int(target_spec*100)}spec"] = sens
            results[f"threshold_at_{int(target_spec*100)}spec"] = thresh
    else:
        results["auroc"] = float("nan")

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trainer_mimic.py::TestLOPOCV -v`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/trainer.py tests/test_trainer_mimic.py
git commit -m "feat: add LOPOCV evaluator for small clinical datasets"
```

---

### Task 2: Dual Operating Points

**Files:**
- Modify: `src/sepsis_vitals/ml/trainer.py`
- Modify: `src/sepsis_vitals/ml/predictor.py`
- Create: `tests/test_predictor_dual.py`

- [ ] **Step 1: Write failing tests for dual thresholds**

Create `tests/test_predictor_dual.py`:

```python
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

        # Below on-demand threshold → low
        assert classify_risk_dual(0.3, thresholds, mode="continuous") == "low"
        # Above on-demand but below continuous → moderate
        assert classify_risk_dual(0.5, thresholds, mode="continuous") == "moderate"
        # Above continuous threshold → high
        assert classify_risk_dual(0.75, thresholds, mode="continuous") == "high"

    def test_classify_risk_on_demand_mode(self):
        from sepsis_vitals.ml.predictor import classify_risk_dual

        thresholds = {
            "continuous": {"threshold": 0.7, "target_specificity": 0.99},
            "on_demand": {"threshold": 0.4, "target_specificity": 0.95},
        }

        # Below on-demand → low
        assert classify_risk_dual(0.3, thresholds, mode="on_demand") == "low"
        # Above on-demand → moderate or higher
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_predictor_dual.py -v`
Expected: FAIL — `cannot import name 'compute_dual_thresholds'`

- [ ] **Step 3: Implement dual threshold computation in trainer.py**

Add to `src/sepsis_vitals/ml/trainer.py` (after `lopocv_evaluate`):

```python
def compute_dual_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute dual operating point thresholds.

    - Continuous monitoring: ≥99% specificity (~1% FPR). When this alerts,
      pay attention. ~1 false alarm per patient per 4 days.
    - On-demand assessment: ≥95% specificity. Higher sensitivity acceptable
      since clinician is already engaged.

    Returns dict with 'continuous' and 'on_demand' keys, each containing
    threshold, target_specificity, achieved_specificity, sensitivity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    result = {}
    for name, target_spec in [("continuous", 0.99), ("on_demand", 0.95)]:
        target_fpr = 1.0 - target_spec
        # Find the threshold that gives us <= target_fpr
        idx = np.searchsorted(fpr, target_fpr)
        if idx >= len(thresholds):
            idx = len(thresholds) - 1

        threshold = float(thresholds[idx])
        achieved_fpr = float(fpr[idx])
        sensitivity = float(tpr[idx])

        result[name] = {
            "threshold": threshold,
            "target_specificity": target_spec,
            "achieved_specificity": round(1.0 - achieved_fpr, 4),
            "sensitivity": round(sensitivity, 4),
        }

    return result
```

- [ ] **Step 4: Implement dual risk classification in predictor.py**

Add to `src/sepsis_vitals/ml/predictor.py` (module-level function, before `SepsisPredictor`):

```python
def classify_risk_dual(
    risk_prob: float,
    thresholds: Dict[str, Dict[str, float]],
    mode: str = "continuous",
) -> str:
    """Classify risk level using dual operating point thresholds.

    Parameters
    ----------
    risk_prob : float
        Predicted probability of sepsis.
    thresholds : dict
        Must have 'continuous' and 'on_demand' keys with 'threshold' values.
    mode : str
        'continuous' (≥99% spec) or 'on_demand' (≥95% spec).
    """
    continuous_thresh = thresholds["continuous"]["threshold"]
    on_demand_thresh = thresholds["on_demand"]["threshold"]

    if risk_prob >= continuous_thresh + 0.15:
        return "critical"
    elif risk_prob >= continuous_thresh:
        return "high"
    elif risk_prob >= on_demand_thresh:
        return "moderate"
    else:
        return "low"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_predictor_dual.py -v`
Expected: All 4 PASS

- [ ] **Step 6: Commit**

```bash
git add src/sepsis_vitals/ml/trainer.py src/sepsis_vitals/ml/predictor.py tests/test_predictor_dual.py
git commit -m "feat: add dual operating points — 99% spec continuous, 95% on-demand"
```

---

### Task 3: Ensemble Module (Gated)

**Files:**
- Create: `src/sepsis_vitals/ml/ensemble.py`
- Create: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing tests for ensemble**

Create `tests/test_ensemble.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ensemble.py -v`
Expected: FAIL — `No module named 'sepsis_vitals.ml.ensemble'`

- [ ] **Step 3: Implement ensemble module**

Create `src/sepsis_vitals/ml/ensemble.py`:

```python
"""
sepsis_vitals.ml.ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-model ensemble for sepsis prediction.

Built and tested but only activated when the training dataset has ≥500 patients.
On small datasets (like the 100-patient MIMIC-IV Demo), a single regularized
model is used instead to avoid overfitting from correlated tree ensembles.

When activated, the ensemble:
1. Trains multiple base models (GBM variants + LogReg)
2. Averages their predicted probabilities (simple averaging)
3. Uses inter-model disagreement for confidence intervals
4. Aggregates feature importance across base models
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MIN_PATIENTS_FOR_ENSEMBLE = 500


def should_use_ensemble(n_patients: int) -> bool:
    """Check whether the dataset is large enough for ensemble training."""
    return n_patients >= MIN_PATIENTS_FOR_ENSEMBLE


class SepsisEnsemble:
    """Multi-model ensemble predictor.

    Uses simple probability averaging across base models. More robust
    than a stacker on moderate-sized datasets (500-5000 patients).

    Parameters
    ----------
    base_models : list
        Trained sklearn-compatible models with predict_proba().
    base_model_names : list of str
        Names for each base model.
    feature_names : list of str
        Feature column names (shared across all base models).
    scalers : list, optional
        Per-model scalers (None for tree models that don't need scaling).
    """

    def __init__(
        self,
        base_models: List[Any],
        base_model_names: List[str],
        feature_names: List[str],
        scalers: Optional[List[Any]] = None,
    ) -> None:
        self.base_models = base_models
        self.base_model_names = base_model_names
        self.feature_names = feature_names
        self.scalers = scalers or [None] * len(base_models)

    def predict(self, X: np.ndarray) -> float:
        """Predict sepsis probability (single sample).

        Returns averaged probability across base models.
        """
        probs = self._get_all_probabilities(X)
        return float(np.mean(probs))

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[float, float, float]:
        """Predict with confidence interval from inter-model disagreement.

        Returns (probability, ci_lower, ci_upper).
        """
        probs = self._get_all_probabilities(X)
        mean_prob = float(np.mean(probs))

        if len(probs) > 1:
            std = float(np.std(probs))
            ci_lower = max(0.0, mean_prob - 1.96 * std)
            ci_upper = min(1.0, mean_prob + 1.96 * std)
        else:
            ci_lower = max(0.0, mean_prob - 0.05)
            ci_upper = min(1.0, mean_prob + 0.05)

        return mean_prob, ci_lower, ci_upper

    def feature_importance(self) -> Dict[str, float]:
        """Average feature importance across base models.

        Uses model.feature_importances_ for tree models, abs(coef_) for linear.
        """
        all_importances = []

        for model in self.base_models:
            if hasattr(model, "feature_importances_"):
                all_importances.append(model.feature_importances_)
            elif hasattr(model, "coef_"):
                all_importances.append(np.abs(model.coef_[0]))

        if not all_importances:
            return {name: 0.0 for name in self.feature_names}

        # Normalize each model's importances to sum to 1, then average
        normalized = []
        for imp in all_importances:
            total = imp.sum()
            normalized.append(imp / total if total > 0 else imp)

        avg = np.mean(normalized, axis=0)
        return dict(zip(self.feature_names, avg.tolist()))

    def metadata(self) -> Dict[str, Any]:
        """Return ensemble metadata for model_metadata.json."""
        return {
            "model_type": "ensemble",
            "n_base_models": len(self.base_models),
            "base_model_names": self.base_model_names,
            "feature_names": self.feature_names,
            "min_patients_for_ensemble": MIN_PATIENTS_FOR_ENSEMBLE,
        }

    def _get_all_probabilities(self, X: np.ndarray) -> List[float]:
        """Get sepsis probability from each base model."""
        probs = []
        for model, scaler in zip(self.base_models, self.scalers):
            X_input = scaler.transform(X) if scaler is not None else X
            prob = float(model.predict_proba(X_input)[:, 1][0])
            probs.append(prob)
        return probs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ensemble.py -v`
Expected: All 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/ensemble.py tests/test_ensemble.py
git commit -m "feat: add ensemble module — gated at ≥500 patients, averaging strategy"
```

---

### Task 4: MIMIC-Demo Training Path in train.py

**Files:**
- Modify: `src/sepsis_vitals/train.py`
- Modify: `tests/test_trainer_mimic.py`

- [ ] **Step 1: Write failing test for MIMIC-demo CLI path**

Append to `tests/test_trainer_mimic.py`:

```python
class TestMIMICDemoTraining:
    """Test the --data-source mimic-demo training path."""

    def test_train_cli_accepts_data_source_flag(self):
        """Verify argparse accepts --data-source."""
        from sepsis_vitals.train import main
        import argparse

        # Should not raise on parse
        # (We can't run full training in a unit test, just verify arg parsing)
        with pytest.raises(SystemExit):
            # --help triggers SystemExit(0)
            main(["--help"])

    def test_train_mimic_demo_small(self):
        """Integration test: train on MIMIC-IV Demo with max_patients=5."""
        from pathlib import Path

        mimic_path = Path("physionet.org/files/mimic-iv-demo/2.2")
        if not (mimic_path / "hosp" / "patients.csv.gz").exists():
            pytest.skip("MIMIC-IV Demo data not available")

        from sepsis_vitals.train import main
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = main([
                "--data-source", "mimic-demo",
                "--max-patients", "10",
                "--output", tmpdir,
                "--skip-shap",
            ])

            # Should produce model artifacts
            assert Path(tmpdir, "sepsis_model.joblib").exists()
            assert Path(tmpdir, "model_metadata.json").exists()
            assert Path(tmpdir, "imputation_medians.json").exists()

            # Metadata should have dual thresholds
            import json
            with open(Path(tmpdir, "model_metadata.json")) as f:
                meta = json.load(f)
            assert "dual_thresholds" in meta or "thresholds" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trainer_mimic.py::TestMIMICDemoTraining -v`
Expected: FAIL

- [ ] **Step 3: Update train.py with MIMIC-demo path**

Modify `src/sepsis_vitals/train.py`:

1. Add new CLI arguments after existing ones:

```python
    parser.add_argument(
        "--data-source", type=str, default="synthetic",
        choices=["synthetic", "mimic-demo"],
        help="Data source: 'synthetic' (default) or 'mimic-demo'"
    )
    parser.add_argument(
        "--max-patients", type=int, default=None,
        help="Limit number of patients (for faster iteration)"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use ensemble training (requires ≥500 patients)"
    )
```

2. Replace Step 1 data loading with a branch:

```python
    if opts.data_source == "mimic-demo":
        print("\n" + "─" * 70)
        print("  STEP 1: Loading MIMIC-IV Demo data with Sepsis-3 labels")
        print("─" * 70)

        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        full_df = loader.build_training_dataset(max_patients=opts.max_patients)

        n_patients = full_df["patient_id"].nunique()
        print(f"\n  Loaded {len(full_df)} observations from {n_patients} patients")
        print(f"  Sepsis prevalence: {full_df['sepsis_label'].mean():.1%}")

        if "label_source" in full_df.columns:
            print(f"  Label sources: {full_df['label_source'].value_counts().to_dict()}")

        # For small datasets, don't split — use LOPOCV instead
        if n_patients < 200:
            print(f"\n  Small dataset ({n_patients} patients) → using LOPOCV evaluation")
            use_lopocv = True
            train_df = full_df  # All data used for LOPOCV and final model
            val_df = full_df    # Same data for calibration (LOPOCV handles eval)
            test_df = full_df   # LOPOCV provides unbiased estimates
        else:
            use_lopocv = False
            # Standard 70/15/15 split by patient
            patient_ids = full_df["patient_id"].unique()
            np.random.seed(opts.seed)
            np.random.shuffle(patient_ids)
            n_train = int(0.7 * len(patient_ids))
            n_val = int(0.15 * len(patient_ids))

            train_pids = set(patient_ids[:n_train])
            val_pids = set(patient_ids[n_train:n_train + n_val])
            test_pids = set(patient_ids[n_train + n_val:])

            train_df = full_df[full_df["patient_id"].isin(train_pids)]
            val_df = full_df[full_df["patient_id"].isin(val_pids)]
            test_df = full_df[full_df["patient_id"].isin(test_pids)]
    else:
        # Existing synthetic data path (unchanged)
        ...
```

3. After model training, compute dual thresholds:

```python
    # ── Dual operating points ───────────────────────────────────────────
    from sepsis_vitals.ml.trainer import compute_dual_thresholds

    y_prob_val = best.model.predict_proba(
        best.scaler.transform(X_val) if best.scaler else X_val
    )[:, 1]
    dual_thresholds = compute_dual_thresholds(y_val, y_prob_val)

    print(f"\n  Dual operating points:")
    print(f"    Continuous (99% spec): threshold={dual_thresholds['continuous']['threshold']:.3f}, "
          f"sensitivity={dual_thresholds['continuous']['sensitivity']:.3f}")
    print(f"    On-demand  (95% spec): threshold={dual_thresholds['on_demand']['threshold']:.3f}, "
          f"sensitivity={dual_thresholds['on_demand']['sensitivity']:.3f}")
```

4. Add dual_thresholds to metadata before saving:

```python
    # Add to metadata dict before json.dump
    metadata["dual_thresholds"] = dual_thresholds
```

5. If LOPOCV mode, run LOPOCV after training and include results:

```python
    if opts.data_source == "mimic-demo" and use_lopocv:
        print("\n" + "─" * 70)
        print("  LOPOCV EVALUATION")
        print("─" * 70)

        from sepsis_vitals.ml.trainer import lopocv_evaluate

        lopocv_results = lopocv_evaluate(
            full_df_features, feature_cols,
            model_type="gradient_boosting" if "GradientBoosting" in best.name else "logistic",
        )

        print(f"    LOPOCV AUROC: {lopocv_results.get('auroc', 'N/A')}")
        print(f"    Patients evaluated: {lopocv_results['n_evaluated']}/{lopocv_results['n_patients']}")
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_trainer_mimic.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/train.py tests/test_trainer_mimic.py
git commit -m "feat: add --data-source mimic-demo training path with LOPOCV and dual thresholds"
```

---

### Task 5: Wire Predictor to Use Dual Thresholds

**Files:**
- Modify: `src/sepsis_vitals/ml/predictor.py`
- Modify: `tests/test_predictor_dual.py`

- [ ] **Step 1: Write failing test for predictor dual-mode support**

Append to `tests/test_predictor_dual.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_predictor_dual.py::TestPredictorDualMode -v`
Expected: FAIL — `SepsisPredictor has no attribute 'dual_thresholds'`

- [ ] **Step 3: Update predictor.py to load and use dual thresholds**

Modify `src/sepsis_vitals/ml/predictor.py`:

1. In `SepsisPredictor.__init__`, add: `self.dual_thresholds = None`

2. In `SepsisPredictor.load()`, after loading metadata, add:
```python
        self.dual_thresholds = self.metadata.get("dual_thresholds")
```

3. Update `_classify_risk` to use dual thresholds when available:
```python
    def _classify_risk(self, prob: float, scores: Any) -> str:
        """Classify risk using dual operating points if available."""
        if self.dual_thresholds:
            return classify_risk_dual(prob, self.dual_thresholds, mode="continuous")

        # Fallback: existing fixed thresholds
        if prob >= 0.75 or scores.risk_level == "critical":
            return "critical"
        elif prob >= 0.50 or scores.risk_level == "high":
            return "high"
        elif prob >= 0.25 or scores.risk_level == "moderate":
            return "moderate"
        else:
            return "low"
```

4. Add `model_info()` method if not present, or update it to include dual_thresholds:
```python
    def model_info(self) -> Dict[str, Any]:
        """Return model information for the /model/info API endpoint."""
        if not self._loaded:
            self.load()
        info = {
            "model_name": self.metadata.get("model_name", "unknown"),
            "version": self.metadata.get("version", "unknown"),
            "feature_count": len(self.feature_names),
            "is_calibrated": self.metadata.get("is_calibrated", False),
            "metrics": self.metadata.get("metrics", {}),
        }
        if self.dual_thresholds:
            info["dual_thresholds"] = self.dual_thresholds
        return info
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_predictor_dual.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing predictor tests to check regressions**

Run: `python3 -m pytest tests/ -k "predictor" -v`

- [ ] **Step 6: Commit**

```bash
git add src/sepsis_vitals/ml/predictor.py tests/test_predictor_dual.py
git commit -m "feat: wire dual operating points into SepsisPredictor"
```

---

### Task 6: Retrain on MIMIC-IV Demo Data

**Files:**
- No new files — this task runs the training pipeline

- [ ] **Step 1: Run training on MIMIC-IV Demo data**

```bash
python3 -m sepsis_vitals.train \
    --data-source mimic-demo \
    --output models/mimic-demo \
    --skip-shap
```

Expected output: model artifacts in `models/mimic-demo/`, LOPOCV results, dual thresholds.

- [ ] **Step 2: Verify model artifacts**

```bash
python3 -c "
import json
from pathlib import Path

meta_path = Path('models/mimic-demo/model_metadata.json')
meta = json.loads(meta_path.read_text())
print(f'Model: {meta[\"model_name\"]}')
print(f'AUROC: {meta[\"metrics\"].get(\"val_auroc\", \"N/A\")}')
print(f'Thresholds: {json.dumps(meta.get(\"dual_thresholds\", {}), indent=2)}')
print(f'Features: {len(meta[\"feature_names\"])}')
"
```

- [ ] **Step 3: Commit model artifacts**

```bash
git add models/mimic-demo/
git commit -m "feat: retrain sepsis model on MIMIC-IV Demo with Sepsis-3 labels"
```

---

### Task 7: Full Test Suite and Final Cleanup

**Files:**
- All Layer 2 files

- [ ] **Step 1: Run the complete test suite**

Run: `python3 -m pytest tests/test_ensemble.py tests/test_trainer_mimic.py tests/test_predictor_dual.py -v`
Expected: All PASS

- [ ] **Step 2: Run existing tests for regressions**

Run: `python3 -m pytest tests/ --ignore=tests/test_mimic_loader_demo.py --ignore=tests/test_layer1_integration.py -v`
Expected: All PASS, no regressions

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Layer 2 ML model — LOPOCV, dual thresholds, ensemble ready"
```
