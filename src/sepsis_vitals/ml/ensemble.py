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
