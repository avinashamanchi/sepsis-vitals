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
