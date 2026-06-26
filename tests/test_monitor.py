"""Tests for the autonomous prediction engine monitor module."""

import asyncio
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
