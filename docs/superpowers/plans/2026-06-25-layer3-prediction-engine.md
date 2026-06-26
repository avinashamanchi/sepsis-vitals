# Layer 3: Autonomous Prediction Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing prediction engine (`SepsisPredictor`) into an event-driven monitoring system where vitals ingestion triggers predictions automatically, deterioration is tracked over clinically meaningful 2-hour windows, and alerts propagate via WebSocket — all using the existing Redis/Postgres infrastructure.

**Architecture:** New `monitor.py` module with three classes (`PatientRegistry`, `VitalsIngester`, `DeteriorationTracker`). The ingester is the single entry point for all vitals data — from the `/predict` endpoint, FHIR router, and batch uploads. On data arrival, it debounces (5-min minimum), triggers `SepsisPredictor.predict()`, feeds results to the deterioration tracker, and broadcasts via WebSocket. The existing `state.py` (`RedisPatientStateStore`/`InMemoryPatientStateStore`) handles rolling windows; `state_store.py` (`PatientStateStore` SQLite) handles prediction persistence. No polling loop.

**Tech Stack:** Python 3.9+, FastAPI, asyncio, Redis (via `state.py`), SQLite (via `state_store.py`), WebSocket (via `realtime/websocket.py`), existing `SepsisPredictor`.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/sepsis_vitals/ml/monitor.py` | Create | `PatientRegistry`, `VitalsIngester`, `DeteriorationTracker` |
| `src/sepsis_vitals/api.py` | Modify | Add `/monitor/register`, `/monitor/{id}`, `/monitor/status` endpoints; wire `/predict` to ingester |
| `src/sepsis_vitals/realtime/websocket.py` | Modify | Add typed alert messages: `patient_update`, `escalation`, `deterioration`, `recovery` |
| `tests/test_monitor.py` | Create | Tests for all three monitor classes |
| `tests/test_monitor_api.py` | Create | Tests for monitor API endpoints |

---

### Task 1: DeteriorationTracker

**Files:**
- Create: `src/sepsis_vitals/ml/monitor.py` (partial — DeteriorationTracker only)
- Create: `tests/test_monitor.py` (partial — DeteriorationTracker tests)

This task builds the clinical deterioration detection engine independent of any I/O or infrastructure. Pure logic, easily testable.

- [ ] **Step 1: Write failing tests for DeteriorationTracker**

Create `tests/test_monitor.py`:

```python
"""Tests for the autonomous prediction engine monitor module."""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock


class TestDeteriorationTracker:
    """Test 2-hour rolling window deterioration detection."""

    def test_add_prediction_stores_in_window(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        now = time.time()

        tracker.add_prediction("P001", now, 0.3, "low")
        tracker.add_prediction("P001", now + 60, 0.35, "low")

        window = tracker.get_window("P001")
        assert len(window) == 2

    def test_window_evicts_old_entries(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker(window_seconds=7200)  # 2 hours
        now = time.time()

        # Add prediction 3 hours ago — should be evicted
        tracker.add_prediction("P001", now - 10800, 0.2, "low")
        # Add prediction 1 hour ago — should stay
        tracker.add_prediction("P001", now - 3600, 0.3, "low")
        # Add current prediction
        tracker.add_prediction("P001", now, 0.4, "moderate")

        window = tracker.get_window("P001")
        assert len(window) == 2  # old one evicted

    def test_no_escalation_below_threshold(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        now = time.time()

        tracker.add_prediction("P001", now - 7000, 0.2, "low")
        tracker.add_prediction("P001", now, 0.25, "low")

        result = tracker.evaluate("P001")
        assert result["alert_type"] is None

    def test_escalation_on_sustained_increase(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        now = time.time()

        # Sustained increase of 0.2 over 2 hours, current above threshold
        tracker.add_prediction("P001", now - 7000, 0.35, "low")
        tracker.add_prediction("P001", now - 3600, 0.45, "moderate")
        tracker.add_prediction("P001", now, 0.55, "moderate")

        result = tracker.evaluate("P001")
        assert result["alert_type"] == "deterioration"
        assert result["risk_delta"] >= 0.15

    def test_no_escalation_if_below_risk_floor(self):
        """Even with increase >= 0.15, don't alert if risk is below on-demand threshold."""
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker(risk_floor=0.50)
        now = time.time()

        # Risk increased 0.2 but current is only 0.3 — below floor
        tracker.add_prediction("P001", now - 7000, 0.10, "low")
        tracker.add_prediction("P001", now, 0.30, "low")

        result = tracker.evaluate("P001")
        assert result["alert_type"] is None

    def test_recovery_on_sustained_decrease(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        now = time.time()

        tracker.add_prediction("P001", now - 7000, 0.65, "high")
        tracker.add_prediction("P001", now - 3600, 0.55, "moderate")
        tracker.add_prediction("P001", now, 0.45, "moderate")

        result = tracker.evaluate("P001")
        assert result["alert_type"] == "recovery"

    def test_trend_slope_computation(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        now = time.time()

        # Linear increase: 0.3 → 0.5 over 2 hours = +0.1/hr
        tracker.add_prediction("P001", now - 7200, 0.3, "low")
        tracker.add_prediction("P001", now, 0.5, "moderate")

        result = tracker.evaluate("P001")
        assert abs(result["deterioration_rate_per_hour"] - 0.1) < 0.02

    def test_single_reading_no_alert(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        tracker.add_prediction("P001", time.time(), 0.8, "high")

        result = tracker.evaluate("P001")
        assert result["alert_type"] is None  # insufficient data

    def test_alert_state_machine_no_skip(self):
        """Alert state must progress: normal → elevated → escalated → critical."""
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        now = time.time()

        # First deterioration → elevated (not straight to critical)
        tracker.add_prediction("P001", now - 7000, 0.3, "low")
        tracker.add_prediction("P001", now, 0.85, "critical")

        result = tracker.evaluate("P001")
        state = tracker.get_alert_state("P001")
        assert state in ("elevated", "normal")  # can't skip to critical

    def test_unknown_patient_returns_empty(self):
        from sepsis_vitals.ml.monitor import DeteriorationTracker

        tracker = DeteriorationTracker()
        result = tracker.evaluate("UNKNOWN")
        assert result["alert_type"] is None
        assert result["window_size"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_monitor.py::TestDeteriorationTracker -v`
Expected: FAIL — `No module named 'sepsis_vitals.ml.monitor'`

- [ ] **Step 3: Implement DeteriorationTracker**

Create `src/sepsis_vitals/ml/monitor.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_monitor.py::TestDeteriorationTracker -v`
Expected: All 10 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/monitor.py tests/test_monitor.py
git commit -m "feat: add DeteriorationTracker with 2-hour rolling window and alert state machine"
```

---

### Task 2: PatientRegistry

**Files:**
- Modify: `src/sepsis_vitals/ml/monitor.py` (add PatientRegistry)
- Modify: `tests/test_monitor.py` (add PatientRegistry tests)

- [ ] **Step 1: Write failing tests for PatientRegistry**

Append to `tests/test_monitor.py`:

```python
class TestPatientRegistry:
    """Test patient monitoring lifecycle."""

    def test_register_patient(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001", demographics={"age": 65, "sex": "M"})

        assert registry.is_registered("P001")
        assert not registry.is_registered("P002")

    def test_unregister_patient(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001")
        registry.unregister("P001")

        assert not registry.is_registered("P001")

    def test_list_monitored_patients(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001")
        registry.register("P002")
        registry.register("P003")

        patients = registry.list_patients()
        assert len(patients) == 3
        assert "P001" in [p["patient_id"] for p in patients]

    def test_update_patient_risk(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001")
        registry.update_risk("P001", 0.45, "moderate")

        info = registry.get_patient_info("P001")
        assert info["risk_probability"] == 0.45
        assert info["risk_level"] == "moderate"

    def test_debounce_check(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry(debounce_seconds=300)
        registry.register("P001")

        now = time.time()
        registry.record_prediction_time("P001", now)

        # Immediately after — should be debounced
        assert registry.should_debounce("P001", now + 60) is True
        # After 5 minutes — should not be debounced
        assert registry.should_debounce("P001", now + 301) is False

    def test_patient_info_includes_last_vitals(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001")
        registry.update_vitals("P001", {"heart_rate": 95, "temperature": 38.2})

        info = registry.get_patient_info("P001")
        assert info["vitals"]["heart_rate"] == 95

    def test_unregistered_patient_returns_none(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        assert registry.get_patient_info("UNKNOWN") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_monitor.py::TestPatientRegistry -v`
Expected: FAIL — `cannot import name 'PatientRegistry'`

- [ ] **Step 3: Implement PatientRegistry**

Add to `src/sepsis_vitals/ml/monitor.py` (after DeteriorationTracker):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_monitor.py::TestPatientRegistry -v`
Expected: All 7 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/monitor.py tests/test_monitor.py
git commit -m "feat: add PatientRegistry for monitoring lifecycle management"
```

---

### Task 3: VitalsIngester

**Files:**
- Modify: `src/sepsis_vitals/ml/monitor.py` (add VitalsIngester)
- Modify: `tests/test_monitor.py` (add VitalsIngester tests)

- [ ] **Step 1: Write failing tests for VitalsIngester**

Append to `tests/test_monitor.py`:

```python
import asyncio


class TestVitalsIngester:
    """Test unified vitals intake and prediction triggering."""

    @pytest.fixture
    def mock_predictor(self):
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.risk_probability = 0.45
        prediction.risk_level = "moderate"
        prediction.to_dict.return_value = {
            "risk_probability": 0.45,
            "risk_level": "moderate",
            "patient_id": "P001",
            "timestamp": "2026-06-25T12:00:00",
        }
        predictor.predict.return_value = prediction
        return predictor

    @pytest.fixture
    def mock_ws_manager(self):
        mgr = MagicMock()
        mgr.broadcast = AsyncMock()
        return mgr

    def test_ingest_single_triggers_prediction(self, mock_predictor, mock_ws_manager):
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker

        registry = PatientRegistry()
        tracker = DeteriorationTracker()
        ingester = VitalsIngester(
            predictor=mock_predictor,
            registry=registry,
            tracker=tracker,
            ws_manager=mock_ws_manager,
        )

        result = asyncio.get_event_loop().run_until_complete(
            ingester.ingest_single("P001", {"heart_rate": 95, "temperature": 38.1})
        )

        # Should auto-register the patient
        assert registry.is_registered("P001")
        # Should call predict
        mock_predictor.predict.assert_called_once()
        # Should return the prediction
        assert result["risk_probability"] == 0.45

    def test_ingest_single_debounces(self, mock_predictor, mock_ws_manager):
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker

        registry = PatientRegistry(debounce_seconds=300)
        tracker = DeteriorationTracker()
        ingester = VitalsIngester(
            predictor=mock_predictor,
            registry=registry,
            tracker=tracker,
            ws_manager=mock_ws_manager,
        )

        # First call — should predict
        asyncio.get_event_loop().run_until_complete(
            ingester.ingest_single("P001", {"heart_rate": 95})
        )
        assert mock_predictor.predict.call_count == 1

        # Second call within debounce window — should skip prediction
        result = asyncio.get_event_loop().run_until_complete(
            ingester.ingest_single("P001", {"heart_rate": 96})
        )
        assert mock_predictor.predict.call_count == 1
        assert result is None  # debounced

    def test_ingest_batch_processes_all(self, mock_predictor, mock_ws_manager):
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker

        registry = PatientRegistry()
        tracker = DeteriorationTracker()
        ingester = VitalsIngester(
            predictor=mock_predictor,
            registry=registry,
            tracker=tracker,
            ws_manager=mock_ws_manager,
        )

        records = [
            {"patient_id": "P001", "vitals": {"heart_rate": 88}},
            {"patient_id": "P002", "vitals": {"heart_rate": 110}},
        ]

        results = asyncio.get_event_loop().run_until_complete(
            ingester.ingest_batch(records)
        )

        assert len(results) == 2
        assert registry.is_registered("P001")
        assert registry.is_registered("P002")

    def test_ingest_broadcasts_alert_on_high_risk(self, mock_ws_manager):
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker

        predictor = MagicMock()
        prediction = MagicMock()
        prediction.risk_probability = 0.85
        prediction.risk_level = "high"
        prediction.to_dict.return_value = {
            "risk_probability": 0.85,
            "risk_level": "high",
            "patient_id": "P001",
            "timestamp": "2026-06-25T12:00:00",
        }
        predictor.predict.return_value = prediction

        registry = PatientRegistry()
        tracker = DeteriorationTracker()
        ingester = VitalsIngester(
            predictor=predictor,
            registry=registry,
            tracker=tracker,
            ws_manager=mock_ws_manager,
        )

        asyncio.get_event_loop().run_until_complete(
            ingester.ingest_single("P001", {"heart_rate": 130})
        )

        # Should broadcast alert
        mock_ws_manager.broadcast.assert_called()
        call_args = mock_ws_manager.broadcast.call_args[0][0]
        assert call_args["type"] in ("patient_update", "sepsis_alert")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_monitor.py::TestVitalsIngester -v`
Expected: FAIL — `cannot import name 'VitalsIngester'`

- [ ] **Step 3: Implement VitalsIngester**

Add to `src/sepsis_vitals/ml/monitor.py` (after PatientRegistry):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_monitor.py::TestVitalsIngester -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/monitor.py tests/test_monitor.py
git commit -m "feat: add VitalsIngester — event-driven prediction on data arrival"
```

---

### Task 4: Enhanced WebSocket Messages

**Files:**
- Modify: `src/sepsis_vitals/realtime/websocket.py`
- Create: `tests/test_websocket_messages.py`

- [ ] **Step 1: Write failing tests for enhanced messages**

Create `tests/test_websocket_messages.py`:

```python
"""Tests for enhanced WebSocket alert messages."""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestWebSocketMessages:
    """Test typed alert message formatting."""

    def test_format_patient_update(self):
        from sepsis_vitals.realtime.websocket import format_alert_message

        msg = format_alert_message(
            alert_type="patient_update",
            patient_id="P001",
            risk_probability=0.35,
            risk_level="low",
        )
        assert msg["type"] == "patient_update"
        assert msg["patient_id"] == "P001"
        assert msg["risk_probability"] == 0.35

    def test_format_deterioration_alert(self):
        from sepsis_vitals.realtime.websocket import format_alert_message

        msg = format_alert_message(
            alert_type="deterioration",
            patient_id="P001",
            risk_probability=0.65,
            risk_level="high",
            previous_risk_level="moderate",
            risk_delta=0.2,
            deterioration_rate=0.1,
            window_hours=2.0,
        )
        assert msg["type"] == "deterioration_alert"
        assert msg["risk_delta"] == 0.2
        assert msg["deterioration_rate"] == 0.1
        assert msg["window_hours"] == 2.0
        assert msg["previous_risk_level"] == "moderate"

    def test_format_recovery_alert(self):
        from sepsis_vitals.realtime.websocket import format_alert_message

        msg = format_alert_message(
            alert_type="recovery",
            patient_id="P001",
            risk_probability=0.35,
            risk_level="low",
            previous_risk_level="high",
            risk_delta=-0.25,
        )
        assert msg["type"] == "recovery_alert"
        assert msg["risk_delta"] == -0.25

    def test_format_escalation_alert(self):
        from sepsis_vitals.realtime.websocket import format_alert_message

        msg = format_alert_message(
            alert_type="escalation",
            patient_id="P001",
            risk_probability=0.8,
            risk_level="critical",
            previous_risk_level="high",
        )
        assert msg["type"] == "escalation_alert"
        assert msg["previous_risk_level"] == "high"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_websocket_messages.py -v`
Expected: FAIL — `cannot import name 'format_alert_message'`

- [ ] **Step 3: Add format_alert_message to websocket.py**

Add to `src/sepsis_vitals/realtime/websocket.py` (after the `manager` singleton):

```python
def format_alert_message(
    alert_type: str,
    patient_id: str,
    risk_probability: float,
    risk_level: str,
    previous_risk_level: str | None = None,
    risk_delta: float = 0.0,
    deterioration_rate: float = 0.0,
    window_hours: float = 0.0,
) -> dict:
    """Format a typed alert message for WebSocket broadcast.

    Alert types:
    - patient_update: routine vitals/risk refresh
    - deterioration: sustained risk increase over 2-hour window
    - recovery: sustained risk decrease over 2-hour window
    - escalation: risk level crossed into high/critical
    - new_risk: first prediction for a patient
    """
    type_map = {
        "patient_update": "patient_update",
        "deterioration": "deterioration_alert",
        "recovery": "recovery_alert",
        "escalation": "escalation_alert",
        "new_risk": "new_risk_alert",
    }

    msg = {
        "type": type_map.get(alert_type, alert_type),
        "patient_id": patient_id,
        "risk_probability": risk_probability,
        "risk_level": risk_level,
    }

    if previous_risk_level is not None:
        msg["previous_risk_level"] = previous_risk_level

    if alert_type in ("deterioration", "recovery"):
        msg["risk_delta"] = risk_delta

    if alert_type == "deterioration":
        msg["deterioration_rate"] = deterioration_rate
        msg["window_hours"] = window_hours

    return msg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_websocket_messages.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/realtime/websocket.py tests/test_websocket_messages.py
git commit -m "feat: add typed WebSocket alert messages — deterioration, recovery, escalation"
```

---

### Task 5: Monitor API Endpoints

**Files:**
- Modify: `src/sepsis_vitals/api.py`
- Create: `tests/test_monitor_api.py`

- [ ] **Step 1: Write failing tests for monitor endpoints**

Create `tests/test_monitor_api.py`:

```python
"""Tests for monitoring API endpoints."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestMonitorEndpoints:
    """Test /monitor/* API endpoints using the monitor module directly."""

    def test_register_patient_creates_entry(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001", demographics={"age": 65})

        assert registry.is_registered("P001")
        info = registry.get_patient_info("P001")
        assert info["demographics"]["age"] == 65

    def test_unregister_patient_removes_entry(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001")
        registry.unregister("P001")
        assert not registry.is_registered("P001")

    def test_status_lists_all_monitored(self):
        from sepsis_vitals.ml.monitor import PatientRegistry

        registry = PatientRegistry()
        registry.register("P001")
        registry.register("P002")
        registry.update_risk("P001", 0.6, "high")

        patients = registry.list_patients()
        assert len(patients) == 2
        p1 = next(p for p in patients if p["patient_id"] == "P001")
        assert p1["risk_probability"] == 0.6

    def test_predict_endpoint_feeds_ingester(self):
        """When /predict is called, it should also feed the monitor ingester."""
        # This test verifies the wiring concept — the actual endpoint test
        # requires a running FastAPI app with auth, so we test the logic directly.
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker

        predictor = MagicMock()
        prediction = MagicMock()
        prediction.risk_probability = 0.3
        prediction.risk_level = "low"
        prediction.to_dict.return_value = {"risk_probability": 0.3, "risk_level": "low"}
        predictor.predict.return_value = prediction

        registry = PatientRegistry()
        tracker = DeteriorationTracker()
        ws = MagicMock()
        ws.broadcast = AsyncMock()
        ingester = VitalsIngester(predictor, registry, tracker, ws)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            ingester.ingest_single("P001", {"heart_rate": 80, "temperature": 37.0})
        )

        # Patient should be registered
        assert registry.is_registered("P001")
        # Prediction should have been made
        predictor.predict.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they pass** (these test the monitor module, not HTTP)

Run: `python3 -m pytest tests/test_monitor_api.py -v`

- [ ] **Step 3: Add monitor endpoints to api.py**

Add the following to `src/sepsis_vitals/api.py`:

1. Near the top, after the predictor singleton, add monitor singleton setup:

```python
# ---------------------------------------------------------------------------
# Monitor singleton (lazy initialization)
# ---------------------------------------------------------------------------

_monitor_registry = None
_monitor_tracker = None
_monitor_ingester = None


def _get_monitor_components():
    """Lazy-initialize the monitoring components."""
    global _monitor_registry, _monitor_tracker, _monitor_ingester

    if _monitor_registry is None:
        from sepsis_vitals.ml.monitor import (
            PatientRegistry,
            DeteriorationTracker,
            VitalsIngester,
        )

        _monitor_registry = PatientRegistry()
        _monitor_tracker = DeteriorationTracker()

        predictor = _get_predictor()
        if predictor is not None:
            _monitor_ingester = VitalsIngester(
                predictor=predictor,
                registry=_monitor_registry,
                tracker=_monitor_tracker,
                ws_manager=ws_manager,
            )

    return _monitor_registry, _monitor_tracker, _monitor_ingester
```

2. Add endpoints after the existing `/patient/{patient_id}/trend` endpoint:

```python
@app.post("/monitor/register", dependencies=[Depends(check_rate_limit)])
async def monitor_register(body: dict, user: Dict = Depends(verify_auth)):
    """Register a patient for continuous monitoring."""
    patient_id = sanitise_string(body.get("patient_id", ""))
    if not patient_id:
        raise HTTPException(status_code=422, detail="patient_id required")

    registry, tracker, ingester = _get_monitor_components()
    registry.register(
        patient_id,
        demographics=body.get("demographics"),
        comorbidities=body.get("comorbidities"),
    )

    return {"status": "registered", "patient_id": patient_id}


@app.delete("/monitor/{patient_id}", dependencies=[Depends(check_rate_limit)])
async def monitor_unregister(patient_id: str, user: Dict = Depends(verify_auth)):
    """Remove a patient from continuous monitoring."""
    registry, tracker, ingester = _get_monitor_components()
    registry.unregister(sanitise_string(patient_id))
    tracker.remove_patient(sanitise_string(patient_id))

    return {"status": "unregistered", "patient_id": patient_id}


@app.get("/monitor/status", dependencies=[Depends(check_rate_limit)])
async def monitor_status(user: Dict = Depends(verify_auth)):
    """List all monitored patients with current risk and trend."""
    registry, tracker, ingester = _get_monitor_components()
    patients = registry.list_patients()

    # Enrich with deterioration data
    for p in patients:
        pid = p["patient_id"]
        det = tracker.evaluate(pid)
        p["alert_state"] = det.get("alert_state", "normal")
        p["deterioration_rate"] = det.get("deterioration_rate_per_hour", 0.0)
        p["window_hours"] = det.get("window_hours", 0.0)

    return {"patients": patients, "count": len(patients)}
```

3. In the existing `/predict` endpoint, after the prediction is made and before the return, add a call to feed the monitor:

```python
    # Feed into monitor if active
    _, _, ingester = _get_monitor_components()
    if ingester is not None and registry.is_registered(body.patient_id):
        registry.update_risk(body.patient_id, result["risk_probability"], result["risk_level"])
        tracker.add_prediction(
            body.patient_id, time.time(), result["risk_probability"], result["risk_level"]
        )
```

- [ ] **Step 4: Run all monitor tests**

Run: `python3 -m pytest tests/test_monitor.py tests/test_monitor_api.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/api.py tests/test_monitor_api.py
git commit -m "feat: add /monitor/register, /monitor/status, /monitor/{id} endpoints"
```

---

### Task 6: Wire FHIR Router to Ingester

**Files:**
- Modify: `src/sepsis_vitals/fhir/router.py`
- Modify: `tests/test_monitor.py` (add FHIR ingestion test)

- [ ] **Step 1: Write failing test for FHIR → ingester wiring**

Append to `tests/test_monitor.py`:

```python
class TestFHIRIngestion:
    """Test that FHIR observations flow into the monitor."""

    def test_ingest_fhir_observation(self):
        """VitalsIngester.ingest_single should handle FHIR-extracted vitals."""
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker

        predictor = MagicMock()
        prediction = MagicMock()
        prediction.risk_probability = 0.55
        prediction.risk_level = "moderate"
        prediction.to_dict.return_value = {
            "risk_probability": 0.55,
            "risk_level": "moderate",
            "patient_id": "fhir-123",
            "timestamp": "2026-06-25T12:00:00",
        }
        predictor.predict.return_value = prediction

        ws = MagicMock()
        ws.broadcast = AsyncMock()
        registry = PatientRegistry()
        tracker = DeteriorationTracker()
        ingester = VitalsIngester(predictor, registry, tracker, ws)

        # Simulate FHIR-extracted vitals
        fhir_vitals = {
            "heart_rate": 110,
            "temperature": 38.5,
            "resp_rate": 24,
            "sbp": 95,
            "spo2": 93,
        }

        result = asyncio.get_event_loop().run_until_complete(
            ingester.ingest_single("fhir-123", fhir_vitals)
        )

        assert result is not None
        assert result["risk_probability"] == 0.55
        assert registry.is_registered("fhir-123")
```

- [ ] **Step 2: Run test to verify it passes** (already implemented via VitalsIngester)

Run: `python3 -m pytest tests/test_monitor.py::TestFHIRIngestion -v`
Expected: PASS

- [ ] **Step 3: Wire FHIR router to ingester**

In `src/sepsis_vitals/fhir/router.py`, in the `POST /fhir/Observation` endpoint (the `create_observation` function at ~line 126), add the following **after** `db.commit()` (line 166) and **before** the `fhir_obs = to_fhir_observation(...)` call (line 168):

```python
    # Feed into monitor if available
    try:
        from sepsis_vitals.api import _get_monitor_components
        _, _, ingester = _get_monitor_components()
        if ingester is not None:
            import asyncio
            asyncio.ensure_future(
                ingester.ingest_single(
                    str(patient.id),
                    {obs.internal_name: obs.value},
                )
            )
    except ImportError:
        pass  # Monitor not available
```

The variables used are from the existing endpoint: `patient` (resolved DB row), `obs.internal_name` (e.g., `"heart_rate"`), and `obs.value` (float). These are already computed earlier in the function at lines 149 and 162.

- [ ] **Step 4: Commit**

```bash
git add src/sepsis_vitals/fhir/router.py tests/test_monitor.py
git commit -m "feat: wire FHIR Observation endpoint to monitor ingester"
```

---

### Task 7: Integration Test and Full Suite

**Files:**
- All Layer 3 files

- [ ] **Step 1: Write integration test**

Append to `tests/test_monitor.py`:

```python
class TestMonitorIntegration:
    """End-to-end integration test for the monitoring pipeline."""

    def test_full_pipeline_flow(self):
        """Vitals in → prediction → deterioration tracking → alert."""
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker

        predictor = MagicMock()
        ws = MagicMock()
        ws.broadcast = AsyncMock()
        registry = PatientRegistry(debounce_seconds=0)  # disable debounce for test
        tracker = DeteriorationTracker(risk_floor=0.40)

        call_count = 0
        def mock_predict(**kwargs):
            nonlocal call_count
            call_count += 1
            pred = MagicMock()
            # Simulate escalating risk
            risks = [0.3, 0.45, 0.55, 0.65]
            levels = ["low", "moderate", "moderate", "high"]
            idx = min(call_count - 1, len(risks) - 1)
            pred.risk_probability = risks[idx]
            pred.risk_level = levels[idx]
            pred.to_dict.return_value = {
                "risk_probability": risks[idx],
                "risk_level": levels[idx],
                "patient_id": kwargs.get("patient_id", "P001"),
                "timestamp": f"2026-06-25T12:{call_count:02d}:00",
            }
            return pred
        predictor.predict.side_effect = mock_predict

        ingester = VitalsIngester(predictor, registry, tracker, ws)

        import asyncio

        # Ingest 4 sets of vitals
        for i in range(4):
            asyncio.get_event_loop().run_until_complete(
                ingester.ingest_single("P001", {"heart_rate": 80 + i * 10})
            )

        # Verify state
        assert registry.is_registered("P001")
        info = registry.get_patient_info("P001")
        assert info["risk_level"] == "high"

        # Verify WebSocket was called 4 times
        assert ws.broadcast.call_count == 4
```

- [ ] **Step 2: Run the full Layer 3 test suite**

Run: `python3 -m pytest tests/test_monitor.py tests/test_monitor_api.py tests/test_websocket_messages.py -v`
Expected: All PASS

- [ ] **Step 3: Run full project test suite for regressions**

Run: `python3 -m pytest tests/ --ignore=tests/test_mimic_loader_demo.py --ignore=tests/test_layer1_integration.py -v`
Expected: All PASS, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_monitor.py
git commit -m "feat: add Layer 3 integration tests — full monitoring pipeline flow"
```
