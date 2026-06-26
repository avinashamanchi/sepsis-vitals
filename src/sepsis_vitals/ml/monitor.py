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
