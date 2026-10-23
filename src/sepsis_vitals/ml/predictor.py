"""
sepsis_vitals.ml.predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~
Investigational sepsis prediction engine.

Loads a verified model artifact and scores observations for research
evaluation: model output, rule-based scores, per-patient attribution, data-
quality warnings and the threshold policy in force. Decision support only;
not a diagnosis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sepsis_vitals.ml.decision_policy import model_level as model_band
from sepsis_vitals.ml.decision_policy import threshold_policy
from sepsis_vitals.scores import compute_scores

ATTRIBUTION_METHOD = "median substitution (per patient)"
ATTRIBUTION_NOTE = (
    "Each value is the change in this model output when that input is replaced by "
    "its training median. It describes the model, not the patient's physiology, "
    "and is not causal."
)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _value(value: Any) -> Any:
    """Pipeline value: None becomes NaN so numeric columns stay numeric."""
    return np.nan if value is None else value


def _utc_naive(value: Any) -> pd.Timestamp:
    """Timestamp as naive UTC; naive input is taken to be UTC already."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


@dataclass
class SepsisPrediction:
    """Complete sepsis risk prediction with explanations."""

    patient_id: str
    timestamp: str
    risk_probability: float
    risk_level: str
    # None unless a validated interval exists; see SepsisPredictor._uncertainty.
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]
    alert: bool
    clinical_scores: Dict[str, Any]
    top_risk_factors: List[Dict[str, Any]]
    recommendation: str
    model_name: str
    model_version: str
    rule_risk_level: str = "unknown"
    model_risk_level: str = "unknown"
    provenance: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    threshold_policy: Dict[str, Any] = field(default_factory=dict)
    observed_at: Optional[str] = None

    def versions(self) -> Dict[str, Any]:
        """Everything needed to reproduce this result, in one place."""
        return {
            "model_version": self.model_version,
            "artifact_sha256": self.provenance.get("artifact_sha256"),
            "feature_schema_sha256": self.provenance.get("feature_schema_sha256"),
            "preprocessing_version": self.provenance.get("pipeline_version"),
            "threshold_policy_id": self.threshold_policy.get("id"),
            "threshold_policy_sha256": self.threshold_policy.get("sha256"),
        }

    def to_dict(self) -> Dict[str, Any]:
        from sepsis_vitals.ml.decision_policy import DECISION_SUPPORT_NOTICE

        interval = None
        if self.confidence_lower is not None and self.confidence_upper is not None:
            interval = {
                "lower": round(self.confidence_lower, 4),
                "upper": round(self.confidence_upper, 4),
            }
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "observed_at": self.observed_at,
            "risk_probability": round(self.risk_probability, 4),
            "risk_level": self.risk_level,
            "confidence_interval": interval,
            "uncertainty": self.uncertainty,
            "alert": self.alert,
            "clinical_scores": self.clinical_scores,
            "top_risk_factors": self.top_risk_factors,
            "recommendation": self.recommendation,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
            },
            # risk_level is the higher of the two; both are reported so that
            # disagreement between the rules and the model is visible.
            "rule_risk_level": self.rule_risk_level,
            "model_risk_level": self.model_risk_level,
            "provenance": self.provenance,
            "explanation": self.explanation,
            "data_quality": self.data_quality,
            "threshold_policy": self.threshold_policy,
            "versions": self.versions(),
            "decision_support_notice": DECISION_SUPPORT_NOTICE,
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


_RISK_ORDER = ["low", "moderate", "high", "critical"]


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
    - the development-model output (synthetic data; not a patient probability)
    - clinical score integration (qSOFA, SIRS, NEWS2, Shock Index)
    - per-patient attribution by median substitution (not causal)
    - uncertainty only where the artifact supports it (otherwise none)
    - deterioration tracking (persisted to SQLite, survives restarts)
    """

    def __init__(self, model_dir: Optional[str] = None, state_dir: Optional[str] = None):
        # SEPSIS_MODEL_DIR may point at a read-only mount; mutable patient
        # state goes to SEPSIS_STATE_DIR (defaults to the model directory for
        # backward compatibility with local development).
        self.model_dir = Path(model_dir or os.getenv("SEPSIS_MODEL_DIR") or "models")
        self.state_dir = Path(state_dir or os.getenv("SEPSIS_STATE_DIR") or str(self.model_dir))
        self.artifact_status: Any = None
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
        """Verify and load model, scaler, metadata, and imputation medians.

        Artifacts are checked against ``manifest.json`` (checksums, feature
        schema, scikit-learn version) *before* anything is unpickled; see
        :mod:`sepsis_vitals.ml.artifacts`. Raises ``FileNotFoundError`` when no
        model is present and ``ModelArtifactError`` when it is unusable.
        """
        from sepsis_vitals.ml.artifacts import (
            allow_unverified_from_env,
            check_feature_compatibility,
            verify_artifacts,
        )

        model_path = self.model_dir / "sepsis_model.joblib"
        metadata_path = self.model_dir / "model_metadata.json"
        medians_path = self.model_dir / "imputation_medians.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run 'python -m sepsis_vitals.train' first."
            )

        status = verify_artifacts(self.model_dir, allow_unverified=allow_unverified_from_env())
        # Only reached once the file matches its recorded checksum.
        self.model = joblib.load(model_path)  # nosec B301

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]
        # Raw inputs this artifact actually reads (a vitals-only model reads no labs).
        self.inputs_used = {
            name for name in self._VITALS + self._LABS
            if any(f == name or f.startswith(name + "_") for f in self.feature_names)
        }
        self.dual_thresholds = self.metadata.get("dual_thresholds")
        check_feature_compatibility(self.feature_names)
        self.artifact_status = status

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
        from sepsis_vitals.ml.state_store import default_store_path
        db_path = str(default_store_path(str(self.state_dir)))
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
        track: bool = True,
    ) -> SepsisPrediction:
        """Score one observation, with its recent history, for research evaluation.

        Parameters
        ----------
        vitals : dict
            The current observation: temperature, heart_rate, resp_rate, sbp,
            dbp, spo2, gcs, map, lactate, wbc, procalcitonin (units as in
            :class:`sepsis_vitals.schemas.VitalsInput`).
        patient_id : str
            Patient identifier
        age_years : int, optional
            Patient age. When missing the pipeline uses 55 and the result
            carries a data-quality warning.
        comorbidities : dict, optional
            Dict of {has_hypertension, has_diabetes, has_ckd, has_copd, has_heart_failure}: 0/1
        history : list of dict, optional
            Earlier observations of this patient, each with a ``timestamp``
            and the same vital/lab keys. Deltas, rolling statistics and the
            observation gap are computed from them exactly as in training;
            without history the request is scored as a first observation.
        observed_at : datetime-like, optional
            Time of the current observation (defaults to now). Naive
            timestamps are taken as UTC.
        track : bool
            Record the result in the patient-trend store. Callers that must
            write an audit record first pass False and call :meth:`track`
            once the record exists.

        Returns
        -------
        SepsisPrediction
        """
        if not self._loaded:
            self.load()

        # Rule-based scores on the current observation
        scores = compute_scores(vitals)

        # The training pipeline over this observation and its recent history
        # (no train/serve skew). The attribution variants go through the same
        # single pass: one feature build, one imputation, one scaling.
        rows, now, ignored = self._observation_rows(
            vitals, age_years, comorbidities, history or [], observed_at
        )
        variants = self._attribution_variants(rows, age_years, comorbidities)
        inputs = self._model_inputs([rows] + [v["frame"] for v in variants])
        probabilities = self.model.predict_proba(inputs)[:, 1]
        risk_prob = float(probabilities[0])

        policy = self.threshold_policy()
        rule_level = scores.risk_level
        model_level = model_band(risk_prob, policy)
        # The model may raise the rule-based level, never lower it.
        risk_level = max(rule_level, model_level, key=_RISK_ORDER.index)

        # A rule-based alert can never be suppressed.
        alert = (
            bool(scores.alert_flag)
            or risk_level in ("high", "critical")
            or risk_prob > policy["alert_cutoff"]
        )

        top_factors = self._attributions(variants, risk_prob, probabilities[1:])
        recommendation = self._generate_recommendation(risk_level, risk_prob, scores, top_factors)
        timestamp = pd.Timestamp.now(tz="UTC").isoformat()

        prediction = SepsisPrediction(
            patient_id=patient_id,
            timestamp=timestamp,
            risk_probability=risk_prob,
            risk_level=risk_level,
            confidence_lower=None,
            confidence_upper=None,
            alert=alert,
            clinical_scores=scores.as_dict(),
            top_risk_factors=top_factors,
            recommendation=recommendation,
            model_name=self.metadata["model_name"],
            model_version=self.metadata["version"],
            rule_risk_level=rule_level,
            model_risk_level=model_level,
            provenance=self.artifact_status.provenance() if self.artifact_status else {},
            uncertainty=self._uncertainty(risk_prob),
            explanation={"method": ATTRIBUTION_METHOD, "note": ATTRIBUTION_NOTE},
            data_quality=self._data_quality(vitals, age_years, comorbidities, rows, ignored),
            threshold_policy=policy,
            observed_at=now.isoformat() + "Z",
        )

        if track:
            self.track(prediction)
        return prediction

    def track(self, prediction: SepsisPrediction) -> None:
        """Add a result to the persistent patient-trend store (survives restarts)."""
        if self._state_store is not None:
            self._state_store.add_prediction(
                patient_id=prediction.patient_id,
                timestamp=prediction.timestamp,
                risk_probability=prediction.risk_probability,
                risk_level=prediction.risk_level,
            )

    def threshold_policy(self) -> Dict[str, Any]:
        """The cut-offs in force for this artifact, with id and digest."""
        return threshold_policy(self.dual_thresholds)

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
    _VITALS = ("temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "gcs", "map")
    _LABS = ("lactate", "wbc", "procalcitonin")
    # Used by the pipeline when the caller gives no age (reported as a warning).
    _DEFAULT_AGE = 55
    # Rolling window 3 plus a margin; only the last three rows affect features.
    _HISTORY_ROWS = 4

    def _observation_rows(
        self,
        vitals: Dict[str, Any],
        age_years: Optional[int],
        comorbidities: Optional[Dict[str, int]],
        history: List[Dict[str, Any]],
        observed_at: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], pd.Timestamp, int]:
        """Pipeline rows: recent history (oldest first) plus the current observation.

        Returns the rows, the time of the current observation (naive UTC) and
        the number of history entries ignored because they had no timestamp
        or were not strictly earlier than the current observation.
        """
        now = _utc_naive(observed_at) if observed_at is not None else _utc_naive(pd.Timestamp.now(tz="UTC"))
        static: Dict[str, Any] = {
            "age_years": age_years if age_years is not None else self._DEFAULT_AGE,
        }
        for name in self._COMORBIDITIES:
            static[name] = int((comorbidities or {}).get(name, 0) or 0)

        def _row(obs: Dict[str, Any], ts: pd.Timestamp) -> Dict[str, Any]:
            row = {col: _value(obs.get(col)) for col in self._INPUT_COLUMNS}
            return {**row, **static, "timestamp": ts}

        rows = []
        ignored = 0
        for obs in history:
            raw_ts = obs.get("timestamp")
            if raw_ts is None:
                ignored += 1  # an observation without a time cannot be ordered
                continue
            ts = _utc_naive(raw_ts)
            if ts < now:  # only observations strictly before the current one
                rows.append(_row(obs, ts))
            else:
                ignored += 1
        rows = sorted(rows, key=lambda r: r["timestamp"])[-self._HISTORY_ROWS:]
        rows.append(_row(vitals, now))
        return rows, now, ignored

    def _raw_features(self, frames: List[List[Dict[str, Any]]]) -> np.ndarray:
        """Feature rows (before imputation) for the last observation of each frame.

        Each frame is one sequence of pipeline rows; frames are scored as
        separate patients, so they cannot affect each other's deltas or
        rolling statistics.
        """
        from sepsis_vitals.ml.trainer import prepare_features

        ids = [f"f{i:05d}" for i in range(len(frames))]
        tagged = [{**row, "patient_id": pid} for pid, rows in zip(ids, frames) for row in rows]
        frame = pd.DataFrame(tagged)
        for col in ("on_supplemental_o2", "spo2_scale2"):
            frame[col] = frame[col].astype(object).where(frame[col].notna(), False).astype(bool)
        features, _ = prepare_features(frame)
        last = features.groupby("patient_id", sort=False).tail(1).set_index("patient_id").loc[ids]
        return last.reindex(columns=self.feature_names).to_numpy(dtype=np.float64)

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
        rows, _, _ = self._observation_rows(vitals, age_years, comorbidities, history, observed_at)
        return self._raw_features([rows])

    def _model_inputs(self, frames: List[List[Dict[str, Any]]]) -> np.ndarray:
        """Model input matrix: training medians for missing values, then the
        artifact's scaler (if any). Each row is preprocessed exactly once."""
        X = self._raw_features(frames)
        medians = np.array(
            [self._imputation_medians.get(name, 0.0) for name in self.feature_names],
            dtype=np.float64,
        )
        X = np.where(np.isnan(X), medians, X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def _uncertainty(self, risk_prob: float) -> Dict[str, Any]:
        """Uncertainty that this artifact can actually support, or none.

        Only a calibrated split-conformal predictor gives a statement with a
        stated coverage, and what it gives is a set of outcome labels, not an
        interval for the patient's probability. Without one there is no
        validated estimate, and none is shown. (Earlier versions reported the
        spread of the last boosting stages as a "95% CI"; it was not one.)
        """
        cp = self.conformal_predictor
        quantile = getattr(cp, "_quantile", None)
        if cp is not None and getattr(cp, "_calibrated", False) and quantile is not None:
            labels = []
            if risk_prob <= quantile:
                labels.append("negative")
            if risk_prob >= 1 - quantile:
                labels.append("positive")
            note = (
                "Outcome labels consistent with this output at the stated coverage on "
                "the calibration data (synthetic). Not an interval for this patient's risk."
            )
            if not labels:
                note += (" Empty set: at this coverage the output supports neither label, "
                         "so it should not be read as either high or low risk.")
            elif len(labels) == 2:
                note += " Both labels: the output cannot separate them at this coverage."
            return {
                "method": "split-conformal label set",
                "coverage_target": round(1 - float(cp.alpha), 4),
                "label_set": labels,
                "note": note,
            }
        return {
            "method": "none",
            "note": (
                "No validated uncertainty estimate is available for this model. "
                "The score is a single model output."
            ),
        }

    def _data_quality(
        self,
        vitals: Dict[str, Any],
        age_years: Optional[int],
        comorbidities: Optional[Dict[str, int]],
        rows: List[Dict[str, Any]],
        ignored: int,
    ) -> Dict[str, Any]:
        """Facts about the input that limit how far the output can be read.

        Descriptive only: no clinical judgement about whether the input is
        adequate is made here.
        """
        warnings: List[Dict[str, Any]] = []

        def warn(code: str, message: str, fields: Optional[List[str]] = None) -> None:
            warnings.append({"code": code, "message": message, "fields": fields or []})

        used = getattr(self, "inputs_used", set(self._VITALS + self._LABS))
        absent_vitals = [v for v in self._VITALS if v in used and _missing(vitals.get(v))]
        absent_labs = [v for v in self._LABS if v in used and _missing(vitals.get(v))]
        unused_inputs = [v for v in self._VITALS + self._LABS if v not in used and not _missing(vitals.get(v))]
        if absent_vitals:
            warn("vitals_not_measured",
                 "Not in this observation; the model substituted typical training values.",
                 absent_vitals)
        if absent_labs:
            warn("labs_not_measured",
                 "Not in this observation; the model substituted typical training values.",
                 absent_labs)
        if unused_inputs:
            warn("inputs_not_used_by_model",
                 "Recorded, but this model does not use them; the rule-based scores still do.",
                 unused_inputs)
        if age_years is None:
            warn("age_not_supplied",
                 f"Age not supplied; the pipeline used {self._DEFAULT_AGE} years.", ["age_years"])
        if comorbidities is None:
            warn("comorbidities_not_supplied",
                 "Comorbidities not supplied; all were treated as absent.", list(self._COMORBIDITIES))
        history_used = len(rows) - 1
        if history_used == 0:
            warn("no_history",
                 "No earlier observations; changes and rolling statistics could not be computed.")
        if ignored:
            warn("history_ignored",
                 f"{ignored} earlier observation(s) ignored: no timestamp, or not before the "
                 "current observation.")
        span = None
        if history_used:
            span = round((rows[-1]["timestamp"] - rows[0]["timestamp"]).total_seconds() / 60, 1)
        return {
            "warnings": warnings,
            "observations_used": len(rows),
            "history_span_minutes": span,
            "imputed_fields": absent_vitals + absent_labs,
        }

    def model_info(self) -> Dict[str, Any]:
        """Return model information for the /model/info API endpoint."""
        if not self._loaded:
            self.load()
        info = {
            "artifact_status": self.artifact_status.as_dict() if self.artifact_status else None,
            "model_name": self.metadata.get("model_name", "unknown"),
            "version": self.metadata.get("version", "unknown"),
            "feature_count": len(self.feature_names),
            "is_calibrated": self.metadata.get("is_calibrated", False),
            "metrics": self.metadata.get("metrics", {}),
        }
        if self.dual_thresholds:
            info["dual_thresholds"] = self.dual_thresholds
        return info

    _LABELS = {
        "heart_rate": "Heart rate",
        "resp_rate": "Respiratory rate",
        "sbp": "Systolic blood pressure",
        "dbp": "Diastolic blood pressure",
        "map": "Mean arterial pressure",
        "temperature": "Temperature",
        "spo2": "Oxygen saturation",
        "gcs": "Glasgow Coma Scale",
        "lactate": "Serum lactate",
        "wbc": "White blood cell count",
        "procalcitonin": "Procalcitonin",
        "age_years": "Age",
        "comorbidities": "Comorbidities",
    }

    def _attribution_variants(
        self,
        rows: List[Dict[str, Any]],
        age_years: Optional[int],
        comorbidities: Optional[Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        """One counterfactual sequence per input this patient actually supplied.

        In each, that input's measured values (current and earlier) are
        replaced by its training median; missing values stay missing, so
        missingness indicators do not change.
        """
        variants: List[Dict[str, Any]] = []
        used = getattr(self, "inputs_used", set(self._VITALS + self._LABS))
        for name in self._VITALS + self._LABS:
            if name not in used or all(_missing(r.get(name)) for r in rows):
                continue
            reference = self._imputation_medians.get(name)
            if reference is None or _missing(reference):
                continue
            frame = [
                {**r, name: r[name] if _missing(r.get(name)) else reference} for r in rows
            ]
            current = rows[-1].get(name)
            variants.append({
                "feature": name,
                "current_value": None if current is None or _missing(current) else float(current),
                "reference_value": float(reference),
                "frame": frame,
            })
        reference_age = self._imputation_medians.get("age_years")
        if age_years is not None and reference_age is not None:
            variants.append({
                "feature": "age_years",
                "current_value": float(age_years),
                "reference_value": float(reference_age),
                "frame": [{**r, "age_years": reference_age} for r in rows],
            })
        present = [c for c in self._COMORBIDITIES if int((comorbidities or {}).get(c, 0) or 0)]
        if present:
            variants.append({
                "feature": "comorbidities",
                "current_value": present,
                "reference_value": [],
                "frame": [{**r, **{c: 0 for c in self._COMORBIDITIES}} for r in rows],
            })
        return variants

    def _attributions(
        self,
        variants: List[Dict[str, Any]],
        risk_prob: float,
        variant_probs: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Per-patient attribution: how much each supplied input moved the output.

        ``contribution`` = this output minus the output with that input set
        to its training median (probability units). Positive means this
        patient's values raised the model output relative to a typical
        training value. It describes the model's behaviour, not physiology,
        and inputs interact, so contributions need not sum to the output.
        """
        factors = []
        for variant, prob in zip(variants, variant_probs):
            contribution = risk_prob - float(prob)
            factors.append({
                "feature": variant["feature"],
                "label": self._LABELS.get(variant["feature"], variant["feature"]),
                "contribution": round(contribution, 4),
                "direction": "raised" if contribution > 0 else "lowered" if contribution < 0 else "none",
                "current_value": variant["current_value"],
                "reference_value": variant["reference_value"],
            })
        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
        return factors

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
