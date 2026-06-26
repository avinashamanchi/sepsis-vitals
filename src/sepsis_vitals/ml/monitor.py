"""
sepsis_vitals.ml.monitor
~~~~~~~~~~~~~~~~~~~~~~~~
Autonomous prediction engine: event-driven monitoring, deterioration
tracking, and alert escalation.

Three main components:
- DeteriorationTracker: 2-hour rolling window clinical deterioration detection
- PatientRegistry: Patient lifecycle management for monitoring
- VitalsIngester: Unified data intake that triggers predictions on arrival
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Data types ─────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    """A single prediction in the rolling window."""
    timestamp: float
    risk_probability: float
    risk_level: str


@dataclass
class AlertEvent:
    """An alert event to broadcast via WebSocket."""
    alert_type: str  # "new_risk", "escalation", "deterioration", "recovery"
    patient_id: str
    risk_probability: float
    risk_level: str
    previous_risk_level: Optional[str] = None
    risk_delta: float = 0.0
    deterioration_rate_per_hour: float = 0.0
    window_hours: float = 0.0


# ─── Alert state machine ───────────────────────────────────────────────

ALERT_STATES = ["normal", "elevated", "escalated", "critical"]


def _next_alert_state(current: str) -> str:
    """Progress alert state by one level (no skipping)."""
    idx = ALERT_STATES.index(current)
    if idx < len(ALERT_STATES) - 1:
        return ALERT_STATES[idx + 1]
    return current


def _prev_alert_state(current: str) -> str:
    """Regress alert state by one level."""
    idx = ALERT_STATES.index(current)
    if idx > 0:
        return ALERT_STATES[idx - 1]
    return current


# ─── DeteriorationTracker ──────────────────────────────────────────────

class DeteriorationTracker:
    """Tracks deterioration over a 2-hour rolling window.

    Clinically meaningful deterioration requires:
    1. Risk increase >= 0.15 sustained over the window
    2. Current risk above the risk floor (on-demand threshold)

    Single-reading spikes are noted but NOT escalated. The alert state
    machine progresses one level at a time: normal → elevated → escalated
    → critical. No level skipping.

    Parameters
    ----------
    window_seconds : int
        Rolling window duration in seconds. Default: 7200 (2 hours).
    min_delta : float
        Minimum risk increase to trigger deterioration. Default: 0.15.
    risk_floor : float
        Current risk must be above this to trigger alerts. Default: 0.50.
    recovery_delta : float
        Minimum risk decrease to trigger recovery. Default: 0.15.
    """

    def __init__(
        self,
        window_seconds: int = 7200,
        min_delta: float = 0.15,
        risk_floor: float = 0.50,
        recovery_delta: float = 0.15,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_delta = min_delta
        self.risk_floor = risk_floor
        self.recovery_delta = recovery_delta

        # patient_id → list of PredictionRecord (sorted by timestamp)
        self._windows: Dict[str, List[PredictionRecord]] = {}
        # patient_id → current alert state
        self._alert_states: Dict[str, str] = {}

    def add_prediction(
        self,
        patient_id: str,
        timestamp: float,
        risk_probability: float,
        risk_level: str,
    ) -> None:
        """Add a prediction to the rolling window, evicting expired entries."""
        if patient_id not in self._windows:
            self._windows[patient_id] = []
            self._alert_states[patient_id] = "normal"

        self._windows[patient_id].append(
            PredictionRecord(
                timestamp=timestamp,
                risk_probability=risk_probability,
                risk_level=risk_level,
            )
        )

        # Evict entries outside the window
        cutoff = timestamp - self.window_seconds
        self._windows[patient_id] = [
            p for p in self._windows[patient_id] if p.timestamp >= cutoff
        ]

    def get_window(self, patient_id: str) -> List[PredictionRecord]:
        """Return the current rolling window for a patient."""
        return self._windows.get(patient_id, [])

    def get_alert_state(self, patient_id: str) -> str:
        """Return the current alert state for a patient."""
        return self._alert_states.get(patient_id, "normal")

    def evaluate(self, patient_id: str) -> Dict[str, Any]:
        """Evaluate the rolling window and return alert information.

        Returns dict with:
        - alert_type: "deterioration" | "recovery" | None
        - risk_delta: change from window start to end
        - deterioration_rate_per_hour: slope of risk over window
        - window_size: number of predictions in window
        - window_hours: time span of window in hours
        - current_risk: latest risk probability
        - alert_state: current state machine state
        """
        window = self._windows.get(patient_id, [])
        current_state = self._alert_states.get(patient_id, "normal")

        result = {
            "alert_type": None,
            "risk_delta": 0.0,
            "deterioration_rate_per_hour": 0.0,
            "window_size": len(window),
            "window_hours": 0.0,
            "current_risk": 0.0,
            "previous_risk_level": None,
            "alert_state": current_state,
        }

        if len(window) < 2:
            if window:
                result["current_risk"] = window[-1].risk_probability
            return result

        oldest = window[0]
        newest = window[-1]
        result["current_risk"] = newest.risk_probability
        result["previous_risk_level"] = oldest.risk_level

        # Time span
        time_span_hours = (newest.timestamp - oldest.timestamp) / 3600.0
        result["window_hours"] = time_span_hours

        # Risk delta
        risk_delta = newest.risk_probability - oldest.risk_probability
        result["risk_delta"] = risk_delta

        # Slope (risk change per hour)
        if time_span_hours > 0:
            result["deterioration_rate_per_hour"] = risk_delta / time_span_hours

        # Check for deterioration
        if (
            risk_delta >= self.min_delta
            and newest.risk_probability >= self.risk_floor
        ):
            result["alert_type"] = "deterioration"
            # Progress alert state by one level
            new_state = _next_alert_state(current_state)
            self._alert_states[patient_id] = new_state
            result["alert_state"] = new_state

        # Check for recovery
        elif risk_delta <= -self.recovery_delta:
            result["alert_type"] = "recovery"
            new_state = _prev_alert_state(current_state)
            self._alert_states[patient_id] = new_state
            result["alert_state"] = new_state

        return result

    def remove_patient(self, patient_id: str) -> None:
        """Remove a patient from tracking."""
        self._windows.pop(patient_id, None)
        self._alert_states.pop(patient_id, None)


# ─── PatientRegistry ───────────────────────────────────────────────────

@dataclass
class MonitoredPatient:
    """State for a patient under active monitoring."""
    patient_id: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    comorbidities: Dict[str, int] = field(default_factory=dict)
    vitals: Dict[str, float] = field(default_factory=dict)
    risk_probability: float = 0.0
    risk_level: str = "unknown"
    trend_direction: str = "unknown"  # improving, stable, worsening
    last_prediction_time: float = 0.0
    last_vitals_time: float = 0.0
    registered_at: float = field(default_factory=time.time)


class PatientRegistry:
    """Manages patients under active monitoring.

    Handles registration, deregistration, vitals caching, and debounce
    logic. This is the in-memory layer — backed by Redis/Postgres in
    production via the state stores.

    Parameters
    ----------
    debounce_seconds : int
        Minimum time between predictions for the same patient.
        Default: 300 (5 minutes).
    """

    def __init__(self, debounce_seconds: int = 300) -> None:
        self.debounce_seconds = debounce_seconds
        self._patients: Dict[str, MonitoredPatient] = {}

    def register(
        self,
        patient_id: str,
        demographics: Optional[Dict[str, Any]] = None,
        comorbidities: Optional[Dict[str, int]] = None,
    ) -> None:
        """Register a patient for monitoring."""
        self._patients[patient_id] = MonitoredPatient(
            patient_id=patient_id,
            demographics=demographics or {},
            comorbidities=comorbidities or {},
        )

    def unregister(self, patient_id: str) -> None:
        """Remove a patient from monitoring."""
        self._patients.pop(patient_id, None)

    def is_registered(self, patient_id: str) -> bool:
        """Check if a patient is registered for monitoring."""
        return patient_id in self._patients

    def list_patients(self) -> List[Dict[str, Any]]:
        """List all monitored patients with current state."""
        return [self.get_patient_info(pid) for pid in self._patients]

    def get_patient_info(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get current state for a monitored patient."""
        patient = self._patients.get(patient_id)
        if patient is None:
            return None
        return {
            "patient_id": patient.patient_id,
            "demographics": patient.demographics,
            "vitals": patient.vitals,
            "risk_probability": patient.risk_probability,
            "risk_level": patient.risk_level,
            "trend_direction": patient.trend_direction,
            "last_prediction_time": patient.last_prediction_time,
            "last_vitals_time": patient.last_vitals_time,
            "registered_at": patient.registered_at,
        }

    def update_risk(
        self, patient_id: str, risk_probability: float, risk_level: str
    ) -> None:
        """Update the risk for a monitored patient."""
        patient = self._patients.get(patient_id)
        if patient:
            patient.risk_probability = risk_probability
            patient.risk_level = risk_level

    def update_vitals(self, patient_id: str, vitals: Dict[str, float]) -> None:
        """Update cached vitals for a monitored patient."""
        patient = self._patients.get(patient_id)
        if patient:
            patient.vitals.update(vitals)
            patient.last_vitals_time = time.time()

    def record_prediction_time(self, patient_id: str, timestamp: float) -> None:
        """Record when the last prediction was made."""
        patient = self._patients.get(patient_id)
        if patient:
            patient.last_prediction_time = timestamp

    def should_debounce(self, patient_id: str, current_time: float) -> bool:
        """Check if a prediction should be debounced (too recent)."""
        patient = self._patients.get(patient_id)
        if patient is None or patient.last_prediction_time == 0.0:
            return False
        return (current_time - patient.last_prediction_time) < self.debounce_seconds


# ─── VitalsIngester ────────────────────────────────────────────────────

class VitalsIngester:
    """Unified vitals intake — single entry point for all data sources.

    On data arrival:
    1. Auto-register the patient if not already monitored
    2. Update cached vitals in the registry
    3. Check debounce (5-min minimum between predictions)
    4. If not debounced: run SepsisPredictor.predict()
    5. Feed prediction to DeteriorationTracker
    6. Broadcast result via WebSocket

    Parameters
    ----------
    predictor : SepsisPredictor
        The loaded ML predictor.
    registry : PatientRegistry
        Patient monitoring registry.
    tracker : DeteriorationTracker
        Deterioration detection engine.
    ws_manager : ConnectionManager
        WebSocket connection manager for broadcasting.
    """

    def __init__(
        self,
        predictor: Any,
        registry: PatientRegistry,
        tracker: DeteriorationTracker,
        ws_manager: Any,
    ) -> None:
        self.predictor = predictor
        self.registry = registry
        self.tracker = tracker
        self.ws_manager = ws_manager

    async def ingest_single(
        self,
        patient_id: str,
        vitals: Dict[str, Any],
        demographics: Optional[Dict[str, Any]] = None,
        comorbidities: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ingest vitals for a single patient.

        Auto-registers the patient for monitoring if not already registered.
        Returns the prediction result dict, or None if debounced.
        """
        # Auto-register if not monitored
        if not self.registry.is_registered(patient_id):
            self.registry.register(
                patient_id,
                demographics=demographics,
                comorbidities=comorbidities,
            )

        # Update cached vitals
        numeric_vitals = {
            k: v for k, v in vitals.items()
            if isinstance(v, (int, float)) and v is not None
        }
        self.registry.update_vitals(patient_id, numeric_vitals)

        # Check debounce
        now = time.time()
        if self.registry.should_debounce(patient_id, now):
            logger.debug("Debounced prediction for patient %s", patient_id)
            return None

        # Run prediction
        age_years = None
        if demographics:
            age_years = demographics.get("age") or demographics.get("age_years")

        prediction = self.predictor.predict(
            vitals=vitals,
            patient_id=patient_id,
            age_years=age_years,
            comorbidities=comorbidities,
        )

        # Record prediction time
        self.registry.record_prediction_time(patient_id, now)

        # Update registry risk
        self.registry.update_risk(
            patient_id, prediction.risk_probability, prediction.risk_level
        )

        # Feed to deterioration tracker
        self.tracker.add_prediction(
            patient_id, now, prediction.risk_probability, prediction.risk_level
        )
        deterioration = self.tracker.evaluate(patient_id)

        # Build result
        result = prediction.to_dict()

        # Broadcast via WebSocket
        message = {
            "type": "patient_update",
            "patient_id": patient_id,
            "risk_level": prediction.risk_level,
            "risk_probability": prediction.risk_probability,
            "timestamp": result.get("timestamp"),
            "trend": deterioration.get("alert_state", "normal"),
        }

        if deterioration["alert_type"] == "deterioration":
            message["type"] = "deterioration_alert"
            message["alert_type"] = "deterioration"
            message["risk_delta"] = deterioration["risk_delta"]
            message["deterioration_rate"] = deterioration["deterioration_rate_per_hour"]
            message["window_hours"] = deterioration["window_hours"]
            message["previous_risk_level"] = deterioration.get("previous_risk_level")

        elif deterioration["alert_type"] == "recovery":
            message["type"] = "recovery_alert"
            message["alert_type"] = "recovery"
            message["risk_delta"] = deterioration["risk_delta"]
            message["previous_risk_level"] = deterioration.get("previous_risk_level")

        elif prediction.risk_level in ("high", "critical"):
            message["type"] = "sepsis_alert"

        await self.ws_manager.broadcast(message)

        return result

    async def ingest_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, Any]]]:
        """Ingest vitals for multiple patients.

        Each record must have 'patient_id' and 'vitals' keys.
        Returns list of prediction results (None for debounced).
        """
        results = []
        for record in records:
            result = await self.ingest_single(
                patient_id=record["patient_id"],
                vitals=record["vitals"],
                demographics=record.get("demographics"),
                comorbidities=record.get("comorbidities"),
            )
            results.append(result)
        return results
