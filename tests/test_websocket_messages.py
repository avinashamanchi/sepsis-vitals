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
