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
