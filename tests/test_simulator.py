"""Tests for the simulator module — case replay and ward simulation."""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
import numpy as np


class TestCaseReplay:
    """Test MIMIC-IV case replay at configurable speed."""

    @pytest.fixture
    def sample_timeline(self):
        """A patient's vitals timeline — 6 observations over 6 hours."""
        base = pd.Timestamp("2150-01-01 08:00:00")
        return pd.DataFrame({
            "charttime": [base + pd.Timedelta(hours=i) for i in range(6)],
            "vital_name": ["heart_rate"] * 6,
            "valuenum": [80, 85, 92, 100, 108, 115],
        })

    @pytest.fixture
    def sample_case_meta(self):
        return {
            "subject_id": 1001,
            "stay_id": 3001,
            "age_years": 65,
            "sex": "M",
            "sepsis_label": 1,
            "icu_los_hours": 48.0,
        }

    @pytest.fixture
    def mock_ingester(self):
        ingester = MagicMock()
        ingester.ingest_single = AsyncMock(return_value={
            "risk_probability": 0.5,
            "risk_level": "moderate",
        })
        return ingester

    def test_create_replay(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        assert replay.session_id is not None
        assert replay.speed == 720
        assert replay.patient_id == "mimic-1001"
        assert replay.total_observations == 6

    def test_replay_emits_observations(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        # Run one step
        asyncio.get_event_loop().run_until_complete(replay.step())

        # Should have called ingest_single with the first observation
        mock_ingester.ingest_single.assert_called_once()
        call_kwargs = mock_ingester.ingest_single.call_args
        assert call_kwargs[0][0] == "mimic-1001"  # patient_id
        assert "heart_rate" in call_kwargs[0][1]  # vitals dict

    def test_replay_tracks_position(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        assert replay.position == 0

        asyncio.get_event_loop().run_until_complete(replay.step())
        assert replay.position == 1

        asyncio.get_event_loop().run_until_complete(replay.step())
        assert replay.position == 2

    def test_replay_completes(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        # Step through all observations
        for _ in range(6):
            asyncio.get_event_loop().run_until_complete(replay.step())

        assert replay.is_complete
        assert replay.position == 6

    def test_replay_computes_delay(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        # First observation has no delay (immediate)
        delay = replay.next_delay()
        assert delay == 0.0

        # After first step, delay should be: 1 hour / 720 = 5 seconds
        asyncio.get_event_loop().run_until_complete(replay.step())
        delay = replay.next_delay()
        assert abs(delay - 5.0) < 0.1

    def test_replay_status(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        status = replay.status()
        assert status["session_id"] == replay.session_id
        assert status["type"] == "replay"
        assert status["subject_id"] == 1001
        assert status["total_observations"] == 6
        assert status["position"] == 0
        assert status["progress"] == 0.0
        assert status["is_complete"] is False

    def test_replay_pivots_timeline_to_vitals_dict(self, mock_ingester):
        """Timeline has multiple vital types — pivot to single dict per timepoint."""
        from sepsis_vitals.ml.simulator import CaseReplay

        base = pd.Timestamp("2150-01-01 08:00:00")
        timeline = pd.DataFrame({
            "charttime": [base, base, base, base + pd.Timedelta(hours=1), base + pd.Timedelta(hours=1)],
            "vital_name": ["heart_rate", "temperature", "sbp", "heart_rate", "temperature"],
            "valuenum": [80, 37.2, 120, 90, 37.8],
        })

        meta = {"subject_id": 1001, "stay_id": 3001, "age_years": 65, "sex": "M",
                "sepsis_label": 1, "icu_los_hours": 24.0}

        replay = CaseReplay(case_meta=meta, timeline=timeline, ingester=mock_ingester, speed=720)

        # Should have 2 timepoints (pivoted)
        assert replay.total_observations == 2

        asyncio.get_event_loop().run_until_complete(replay.step())
        call_args = mock_ingester.ingest_single.call_args[0]
        vitals = call_args[1]
        assert "heart_rate" in vitals
        assert "temperature" in vitals
        assert vitals["heart_rate"] == 80
        assert vitals["temperature"] == 37.2


class TestWardSimulator:
    """Test synthetic ward simulation."""

    @pytest.fixture
    def mock_ingester(self):
        ingester = MagicMock()
        ingester.ingest_single = AsyncMock(return_value={
            "risk_probability": 0.3,
            "risk_level": "low",
        })
        return ingester

    def test_create_ward(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=8,
            speed=360,
            sepsis_count=2,
            seed=42,
        )

        assert ward.session_id is not None
        assert ward.n_patients == 8
        assert ward.speed == 360
        assert len(ward.patient_ids) == 8

    def test_ward_generates_trajectories(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=6,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        # Should have generated trajectories for all patients
        assert len(ward._trajectories) == 6

    def test_ward_step_emits_observations(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        asyncio.get_event_loop().run_until_complete(ward.step())

        # Should have called ingest_single for each patient with an observation at this timepoint
        assert mock_ingester.ingest_single.call_count >= 1

    def test_ward_includes_guaranteed_deterioration(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=8,
            speed=360,
            sepsis_count=2,
            seed=42,
        )

        # At least one patient should be the scripted deterioration case
        has_severe = any(
            t.get("sepsis_severity") == "severe"
            for t in ward._patient_configs
        )
        assert has_severe

    def test_ward_status(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=6,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        status = ward.status()
        assert status["session_id"] == ward.session_id
        assert status["type"] == "ward"
        assert status["n_patients"] == 6
        assert status["sepsis_count"] == 1
        assert status["is_complete"] is False

    def test_ward_completes(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=4,
            speed=720,
            sepsis_count=1,
            seed=42,
            obs_per_patient=3,  # small for testing
        )

        # Step through all observations
        for _ in range(20):  # more than enough steps
            if ward.is_complete:
                break
            asyncio.get_event_loop().run_until_complete(ward.step())

        assert ward.is_complete


class TestSimulationManager:
    """Test simulation session lifecycle management."""

    @pytest.fixture
    def mock_ingester(self):
        ingester = MagicMock()
        ingester.ingest_single = AsyncMock(return_value={
            "risk_probability": 0.3,
            "risk_level": "low",
        })
        return ingester

    def test_create_manager(self):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        assert manager.list_sessions() == []

    def test_start_ward_session(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        session_id = manager.start_ward(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        assert session_id is not None
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["type"] == "ward"

    def test_stop_session(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        session_id = manager.start_ward(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
        )

        stopped = manager.stop_session(session_id)
        assert stopped is True

    def test_stop_nonexistent_session(self):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        assert manager.stop_session("nonexistent") is False

    def test_get_session_status(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        session_id = manager.start_ward(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
        )

        status = manager.get_session(session_id)
        assert status is not None
        assert status["session_id"] == session_id


class TestSimulatorIntegration:
    """End-to-end integration test for the simulator pipeline."""

    def test_ward_flows_through_prediction_pipeline(self):
        """Ward sim → VitalsIngester → predict → DeteriorationTracker → WebSocket."""
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker
        from sepsis_vitals.ml.simulator import WardSimulator

        predictor = MagicMock()
        call_count = 0

        def mock_predict(**kwargs):
            nonlocal call_count
            call_count += 1
            pred = MagicMock()
            pred.risk_probability = 0.3 + (call_count % 5) * 0.1
            level = "low" if pred.risk_probability < 0.5 else "moderate"
            pred.risk_level = level
            pred.to_dict.return_value = {
                "risk_probability": pred.risk_probability,
                "risk_level": level,
                "patient_id": kwargs.get("patient_id", "test"),
                "timestamp": "2026-01-01T08:00:00",
            }
            return pred

        predictor.predict.side_effect = mock_predict

        ws = MagicMock()
        ws.broadcast = AsyncMock()
        registry = PatientRegistry(debounce_seconds=0)
        tracker = DeteriorationTracker(risk_floor=0.40)
        ingester = VitalsIngester(predictor, registry, tracker, ws)

        ward = WardSimulator(
            ingester=ingester,
            n_patients=4,
            speed=720,
            sepsis_count=1,
            seed=42,
            obs_per_patient=3,
        )

        # Run all steps
        for _ in range(ward.total_steps + 5):
            if ward.is_complete:
                break
            asyncio.get_event_loop().run_until_complete(ward.step())

        # Verify: all patients registered
        for pid in ward.patient_ids:
            assert registry.is_registered(pid)

        # Verify: predictions were made
        assert predictor.predict.call_count == ward.position

        # Verify: WebSocket broadcasts happened
        assert ws.broadcast.call_count == ward.position

    def test_case_replay_flows_through_pipeline(self):
        """CaseReplay → VitalsIngester → predict → tracking."""
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker
        from sepsis_vitals.ml.simulator import CaseReplay

        predictor = MagicMock()
        pred = MagicMock()
        pred.risk_probability = 0.6
        pred.risk_level = "moderate"
        pred.to_dict.return_value = {
            "risk_probability": 0.6, "risk_level": "moderate",
            "patient_id": "mimic-1001", "timestamp": "2026-01-01T08:00:00",
        }
        predictor.predict.return_value = pred

        ws = MagicMock()
        ws.broadcast = AsyncMock()
        registry = PatientRegistry(debounce_seconds=0)
        tracker = DeteriorationTracker()
        ingester = VitalsIngester(predictor, registry, tracker, ws)

        base = pd.Timestamp("2150-01-01 08:00")
        timeline = pd.DataFrame({
            "charttime": [base + pd.Timedelta(hours=i) for i in range(4)],
            "vital_name": ["heart_rate"] * 4,
            "valuenum": [80, 90, 100, 110],
        })
        meta = {
            "subject_id": 1001, "stay_id": 3001,
            "age_years": 65, "sex": "M",
            "sepsis_label": 1, "icu_los_hours": 24.0,
        }

        replay = CaseReplay(meta, timeline, ingester, speed=720)

        for _ in range(4):
            asyncio.get_event_loop().run_until_complete(replay.step())

        assert replay.is_complete
        assert registry.is_registered("mimic-1001")
        assert predictor.predict.call_count == 4
        assert ws.broadcast.call_count == 4
