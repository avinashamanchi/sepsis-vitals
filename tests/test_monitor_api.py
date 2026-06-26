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
