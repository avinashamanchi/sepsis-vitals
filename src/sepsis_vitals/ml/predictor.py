"""
sepsis_vitals.ml.predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~
Investigational sepsis prediction engine.

Loads a trained model and provides real-time sepsis risk predictions
with SHAP explanations, confidence intervals, and clinical score integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sepsis_vitals.scores import compute_scores


@dataclass
class SepsisPrediction:
    """Complete sepsis risk prediction with explanations."""

    patient_id: str
    timestamp: str
    risk_probability: float
    risk_level: str
    confidence_lower: float
    confidence_upper: float
    alert: bool
    clinical_scores: Dict[str, Any]
    top_risk_factors: List[Dict[str, Any]]
    recommendation: str
    model_name: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "risk_probability": round(self.risk_probability, 4),
            "risk_level": self.risk_level,
            "confidence_interval": {
                "lower": round(self.confidence_lower, 4),
                "upper": round(self.confidence_upper, 4),
            },
            "alert": self.alert,
            "clinical_scores": self.clinical_scores,
            "top_risk_factors": self.top_risk_factors,
            "recommendation": self.recommendation,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
            },
        }


@dataclass
class PatientMonitor:
    """Tracks a patient's predictions over time for deterioration detection."""

    patient_id: str
    predictions: List[SepsisPrediction] = field(default_factory=list)
    baseline_risk: Optional[float] = None

    def add_prediction(self, pred: SepsisPrediction) -> Dict[str, Any]:
        """Add a prediction and check for deterioration."""
        self.predictions.append(pred)

        if self.baseline_risk is None and len(self.predictions) >= 2:
            self.baseline_risk = self.predictions[0].risk_probability

        trend = self._compute_trend()
        deterioration = self._detect_deterioration()

        return {
            "trend": trend,
            "deterioration_detected": deterioration["detected"],
            "deterioration_details": deterioration,
            "n_observations": len(self.predictions),
        }

    def _compute_trend(self) -> str:
        if len(self.predictions) < 2:
            return "insufficient_data"

        recent = [p.risk_probability for p in self.predictions[-3:]]
        if len(recent) >= 2:
            delta = recent[-1] - recent[0]
            if delta > 0.1:
                return "rapidly_worsening"
            elif delta > 0.03:
                return "worsening"
            elif delta < -0.1:
                return "rapidly_improving"
            elif delta < -0.03:
                return "improving"
        return "stable"

    def _detect_deterioration(self) -> Dict[str, Any]:
        if len(self.predictions) < 2:
            return {"detected": False, "reason": "insufficient_data"}

        current = self.predictions[-1]
        previous = self.predictions[-2]

        reasons = []

        # Rapid risk increase
        delta = current.risk_probability - previous.risk_probability
        if delta > 0.15:
            reasons.append(f"Risk jumped by {delta:.0%} in one observation")

        # Crossed into high/critical
        risk_order = ["low", "moderate", "high", "critical"]
        curr_idx = risk_order.index(current.risk_level) if current.risk_level in risk_order else 0
        prev_idx = risk_order.index(previous.risk_level) if previous.risk_level in risk_order else 0
        if curr_idx > prev_idx and curr_idx >= 2:
            reasons.append(f"Risk escalated from {previous.risk_level} to {current.risk_level}")

        # Sustained high risk
        if len(self.predictions) >= 3:
            last_3 = [p.risk_probability for p in self.predictions[-3:]]
            if all(r > 0.6 for r in last_3):
                reasons.append("Sustained high risk (>60%) for 3+ observations")

        return {
            "detected": len(reasons) > 0,
            "reasons": reasons,
            "current_risk": current.risk_probability,
            "previous_risk": previous.risk_probability,
            "delta": delta,
        }


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
        'continuous' (>=99% spec) or 'on_demand' (>=95% spec).
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


class SepsisPredictor:
    """Investigational sepsis prediction engine.

    Loads a trained model and provides predictions with:
    - ML risk probability
    - Clinical score integration (qSOFA, SIRS, NEWS2, Shock Index)
    - SHAP-based risk factor explanations
    - Confidence intervals
    - Deterioration detection (persisted to SQLite, survives restarts)
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        # These objects are deserialized from joblib/JSON and intentionally
        # remain implementation-agnostic across supported sklearn estimators.
        self.model: Any = None
        self.scaler: Any = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self._state_store: Any = None
        self._imputation_medians: Dict[str, float] = {}
        self._loaded = False
        self.dual_thresholds: Optional[Dict[str, Dict[str, float]]] = None
        self.conformal_predictor: Any = None

    def load(self) -> None:
        """Load model, scaler, metadata, and imputation medians from disk."""
        model_path = self.model_dir / "sepsis_model.joblib"
        metadata_path = self.model_dir / "model_metadata.json"
        medians_path = self.model_dir / "imputation_medians.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run 'python -m sepsis_vitals.train' first."
            )

        self.model = joblib.load(model_path)

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]
        self.dual_thresholds = self.metadata.get("dual_thresholds")

        # Load imputation medians for NaN handling
        self._imputation_medians = {}
        if medians_path.exists():
            with open(medians_path) as f:
                self._imputation_medians = json.load(f)

        if self.metadata.get("needs_scaling"):
            scaler_path = self.model_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

        # Load conformal predictor if available (backwards-compatible)
        cp_path = self.model_dir / "conformal_predictor.joblib"
        if cp_path.exists() and self.metadata.get("has_conformal_predictor"):
            self.conformal_predictor = joblib.load(cp_path)

        # Initialize persistent state store (SQLite-backed)
        from sepsis_vitals.ml.state_store import PatientStateStore
        db_path = str(self.model_dir / "patient_state.db")
        self._state_store = PatientStateStore(db_path=db_path)

        self._loaded = True

    def predict(
        self,
        vitals: Dict[str, Any],
        patient_id: str = "unknown",
        age_years: Optional[int] = None,
        comorbidities: Optional[Dict[str, int]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        observed_at: Optional[Any] = None,
    ) -> SepsisPrediction:
        """Generate a sepsis risk prediction from vital signs.

        Parameters
        ----------
        vitals : dict
            Vital signs: temperature, heart_rate, resp_rate, sbp, dbp,
            spo2, gcs, map
        patient_id : str
            Patient identifier
        age_years : int, optional
            Patient age
        comorbidities : dict, optional
            Dict of {has_hypertension, has_diabetes, has_ckd, has_copd, has_heart_failure}: 0/1
        history : list of dict, optional
            Earlier observations of this patient, each with a ``timestamp``
            and the same vital/lab keys. Deltas, rolling statistics and the
            observation gap are computed from them exactly as in training;
            without history the request is scored as a first observation.
        observed_at : datetime-like, optional
            Time of the current observation (defaults to now).

        Returns
        -------
        SepsisPrediction
        """
        if not self._loaded:
            self.load()

        # Compute clinical scores
        scores = compute_scores(vitals)

        # Build the feature vector with the training pipeline (no train/serve skew)
        feature_vector = self._build_feature_vector(
            vitals, age_years, comorbidities, history or [], observed_at
        )

        # Impute NaN values using training medians
        for i, feat_name in enumerate(self.feature_names):
            if np.isnan(feature_vector[0, i]):
                feature_vector[0, i] = self._imputation_medians.get(feat_name, 0.0)

        # Scale if needed
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        # Predict
        risk_prob = float(self.model.predict_proba(feature_vector)[:, 1][0])

        # Confidence interval using ensemble variance
        ci_lower, ci_upper = self._compute_confidence_interval(
            feature_vector, risk_prob
        )

        # Risk level classification
        risk_level = self._classify_risk(risk_prob, scores)

        # Alert determination
        alert = risk_level in ("high", "critical") or risk_prob > 0.6

        # Top risk factors
        top_factors = self._explain_prediction(feature_vector, vitals, scores)

        # Clinical recommendation
        recommendation = self._generate_recommendation(risk_level, risk_prob, scores, top_factors)

        timestamp = pd.Timestamp.now().isoformat()

        prediction = SepsisPrediction(
            patient_id=patient_id,
            timestamp=timestamp,
            risk_probability=risk_prob,
            risk_level=risk_level,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            alert=alert,
            clinical_scores=scores.as_dict(),
            top_risk_factors=top_factors,
            recommendation=recommendation,
            model_name=self.metadata["model_name"],
            model_version=self.metadata["version"],
        )

        # Track in persistent state store (survives restarts)
        if self._state_store is not None:
            self._state_store.add_prediction(
                patient_id=patient_id,
                timestamp=timestamp,
                risk_probability=risk_prob,
                risk_level=risk_level,
            )

        return prediction

    def get_patient_trend(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get the monitoring trend for a patient (from persistent storage)."""
        if self._state_store is None:
            return None
        return self._state_store.get_trend(patient_id)

    _INPUT_COLUMNS = (
        "temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "gcs", "map",
        "lactate", "wbc", "procalcitonin", "on_supplemental_o2", "spo2_scale2",
    )
    _COMORBIDITIES = (
        "has_hypertension", "has_diabetes", "has_ckd", "has_copd", "has_heart_failure",
    )
    # Rolling window 3 plus a margin; only the last three rows affect features.
    _HISTORY_ROWS = 4

    def _build_feature_vector(
        self,
        vitals: Dict[str, Any],
        age_years: Optional[int],
        comorbidities: Optional[Dict[str, int]],
        history: List[Dict[str, Any]],
        observed_at: Optional[Any] = None,
    ) -> np.ndarray:
        """Build the model's feature vector with the *training* pipeline.

        The current observation is appended to the patient's recent history
        and passed through :func:`sepsis_vitals.ml.trainer.prepare_features`,
        so deltas, rolling statistics, missingness and the observation gap are
        computed exactly as during training. Earlier versions scored every
        request as a first observation, which lowered AUROC and roughly halved
        predicted risk (see reports/synthetic_pipeline_audit.md).
        """
        from sepsis_vitals.ml.trainer import prepare_features

        now = pd.Timestamp(observed_at) if observed_at is not None else pd.Timestamp.now()
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
        static: Dict[str, Any] = {
            "patient_id": "_",
            "age_years": age_years if age_years is not None else 55,
        }
        for name in self._COMORBIDITIES:
            static[name] = int((comorbidities or {}).get(name, 0) or 0)

        def _row(obs: Dict[str, Any], ts: pd.Timestamp) -> Dict[str, Any]:
            row = {col: obs.get(col, np.nan) for col in self._INPUT_COLUMNS}
            return {**row, **static, "timestamp": ts}

        rows = []
        for obs in history:
            raw_ts = obs.get("timestamp")
            if raw_ts is None:
                continue  # an observation without a time cannot be ordered
            ts = pd.Timestamp(raw_ts)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            if ts < now:  # only observations strictly before the current one
                rows.append(_row(obs, ts))
        rows = sorted(rows, key=lambda r: r["timestamp"])[-self._HISTORY_ROWS:]
        rows.append(_row(vitals, now))

        frame = pd.DataFrame(rows)
        for col in ("on_supplemental_o2", "spo2_scale2"):
            frame[col] = frame[col].fillna(False).astype(bool)
        features, _ = prepare_features(frame)
        last = features.iloc[-1]
        return np.array(
            [[float(last.get(name, np.nan)) for name in self.feature_names]],
            dtype=np.float64,
        )

    def _compute_confidence_interval(
        self, feature_vector: np.ndarray, risk_prob: float
    ) -> tuple:
        """Compute confidence interval for the predicted risk probability.

        Priority:
        1. Conformal predictor (statistically valid coverage guarantee at
           the alpha level used during calibration) — used when available.
        2. Ensemble stage variance for GradientBoosting models.
        3. ECE-based heuristic — final fallback for all other models.
        """
        # --- 1. Conformal prediction interval ---
        if self.conformal_predictor is not None and self.conformal_predictor._calibrated:
            try:
                lower, upper, _ = self.conformal_predictor.predict_interval(
                    self.model, feature_vector
                )
                return float(lower[0]), float(upper[0])
            except Exception:
                pass  # fall through to heuristics

        # --- 2. Ensemble stage variance (GradientBoosting only) ---
        try:
            if hasattr(self.model, "staged_predict_proba"):
                # Use last 20% of stages to estimate prediction variance
                staged = list(self.model.staged_predict_proba(feature_vector))
                n_stages = len(staged)
                tail_start = max(0, int(n_stages * 0.8))
                tail_probs = [float(s[:, 1][0]) for s in staged[tail_start:]]
                if len(tail_probs) > 1:
                    std = float(np.std(tail_probs))
                    # 95% CI using ~2 standard deviations
                    ci_lower = max(0.0, risk_prob - 1.96 * std)
                    ci_upper = min(1.0, risk_prob + 1.96 * std)
                    return ci_lower, ci_upper
        except Exception:
            pass

        # --- 3. Fallback: use ECE from model metadata as uncertainty width ---
        ece = 0.02  # default
        if self.metadata:
            cal = self.metadata.get("calibration", {})
            if isinstance(cal, dict):
                ece = cal.get("ece", 0.02)
            else:
                metrics = self.metadata.get("metrics", {})
                ece = metrics.get("val_brier", 0.08) ** 0.5

        # Scale uncertainty: wider at mid-range, narrower at extremes
        base_width = max(ece, 0.01) * 3
        distance_from_edge = min(risk_prob, 1.0 - risk_prob)
        width = base_width * (0.5 + distance_from_edge)
        ci_lower = max(0.0, risk_prob - width)
        ci_upper = min(1.0, risk_prob + width)
        return ci_lower, ci_upper

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

    def _explain_prediction(
        self,
        feature_vector: np.ndarray,
        vitals: Dict[str, Any],
        scores: Any,
    ) -> List[Dict[str, Any]]:
        """Generate risk factor explanations."""
        factors = []

        # Use feature importance from metadata
        importance = self.metadata.get("feature_importance", {})

        # Map feature importance to readable explanations
        vital_labels = {
            "heart_rate": "Heart Rate",
            "resp_rate": "Respiratory Rate",
            "sbp": "Systolic Blood Pressure",
            "temperature": "Temperature",
            "spo2": "Oxygen Saturation",
            "gcs": "Glasgow Coma Scale",
            "map": "Mean Arterial Pressure",
            "dbp": "Diastolic Blood Pressure",
            "shock_index_computed": "Shock Index",
            "qsofa": "qSOFA Score",
            "news2_computed": "NEWS2 Score",
            "sirs_computed": "SIRS Criteria",
            "lactate": "Serum Lactate",
            "wbc": "White Blood Cell Count",
            "procalcitonin": "Procalcitonin",
        }

        # Normal ranges for context
        normal_ranges = {
            "heart_rate": (60, 100),
            "resp_rate": (12, 20),
            "sbp": (90, 140),
            "temperature": (36.1, 37.2),
            "spo2": (95, 100),
            "gcs": (15, 15),
            "map": (70, 105),
            "lactate": (0.5, 2.0),
            "wbc": (4.5, 11.0),
            "procalcitonin": (0.0, 0.1),
        }

        for feat_name, imp_value in list(importance.items())[:10]:
            base_name = feat_name.replace("_delta", "").replace("_roll_mean", "").replace("_roll_std", "")
            label = vital_labels.get(feat_name, vital_labels.get(base_name, feat_name))
            value = vitals.get(feat_name, vitals.get(base_name))

            factor = {
                "feature": feat_name,
                "label": label,
                "importance": round(float(imp_value), 4),
            }

            if value is not None and base_name in normal_ranges:
                lo, hi = normal_ranges[base_name]
                factor["current_value"] = value
                factor["normal_range"] = f"{lo}-{hi}"
                if value > hi:
                    factor["status"] = "elevated"
                elif value < lo:
                    factor["status"] = "low"
                else:
                    factor["status"] = "normal"

            factors.append(factor)

        return factors[:5]

    def _generate_recommendation(
        self,
        risk_level: str,
        risk_prob: float,
        scores: Any,
        factors: List[Dict[str, Any]],
    ) -> str:
        """Generate a research interpretation without treatment instructions."""
        context = f"Development-model output: {risk_prob:.0%}; qSOFA: {scores.qsofa}/3."
        if risk_level == "critical":
            return (
                "High-priority research signal. Flag this case for blinded study review; "
                "do not use this output to guide diagnosis or treatment. "
                f"{context}"
            )
        elif risk_level == "high":
            return (
                "Elevated research signal. Include this case in the study review queue "
                "and compare it with the prespecified reference standard. "
                f"{context}"
            )
        elif risk_level == "moderate":
            return (
                "Intermediate research signal. Record the result for cohort-level "
                "calibration and alert-burden analysis. "
                f"{context}"
            )
        else:
            return (
                "Lower research signal. A low output does not rule out sepsis and must "
                "not alter the existing standard of care. "
                f"{context}"
            )
