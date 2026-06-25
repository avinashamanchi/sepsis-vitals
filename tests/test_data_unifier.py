"""Tests for time-window binning and multi-source data unification.

Covers epoch binning with median/max aggregation, forward-fill,
per-patient isolation, and multi-source dataset unification.
"""
from __future__ import annotations

import pandas as pd
import pytest

from sepsis_vitals.ml.data_unifier import bin_to_epochs, unify_datasets


# ---------------------------------------------------------------------------
# TestTimeWindowBinning
# ---------------------------------------------------------------------------
class TestTimeWindowBinning:
    """Test epoch binning logic for clinical time-series data."""

    def test_bins_observations_to_1h_epochs(self) -> None:
        """HR at 10:01 + SBP at 10:45 -> one epoch at 10:00 with both values.
        HR at 11:30 -> second epoch at 11:00."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P1"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:01",
                        "2024-01-01 10:45",
                        "2024-01-01 11:30",
                    ]
                ),
                "heart_rate": [80.0, None, 95.0],
                "sbp": [None, 120.0, None],
            }
        )
        result = bin_to_epochs(df, epoch_minutes=60, forward_fill=False)

        # Should have 2 epochs: 10:00 and 11:00
        assert len(result) == 2
        epoch_10 = result[result["epoch"] == pd.Timestamp("2024-01-01 10:00")]
        epoch_11 = result[result["epoch"] == pd.Timestamp("2024-01-01 11:00")]

        assert len(epoch_10) == 1
        assert len(epoch_11) == 1

        # Epoch 10:00 should have both HR and SBP
        assert epoch_10["heart_rate"].iloc[0] == 80.0
        assert epoch_10["sbp"].iloc[0] == 120.0

        # Epoch 11:00 should have HR only
        assert epoch_11["heart_rate"].iloc[0] == 95.0
        assert pd.isna(epoch_11["sbp"].iloc[0])

    def test_median_aggregation_within_epoch(self) -> None:
        """3 HR readings [80, 90, 85] in same epoch -> median = 85.0."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P1"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:05",
                        "2024-01-01 10:20",
                        "2024-01-01 10:40",
                    ]
                ),
                "heart_rate": [80.0, 90.0, 85.0],
            }
        )
        result = bin_to_epochs(df, epoch_minutes=60, forward_fill=False)
        assert len(result) == 1
        assert result["heart_rate"].iloc[0] == 85.0

    def test_forward_fill_missing_vitals(self) -> None:
        """Epoch 1 has HR+SBP; Epoch 2 has HR only -> SBP forward-filled from epoch 1."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P1", "P1"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00",
                        "2024-01-01 10:30",
                        "2024-01-01 11:15",
                    ]
                ),
                "heart_rate": [80.0, 82.0, 90.0],
                "sbp": [120.0, 118.0, None],
            }
        )
        result = bin_to_epochs(df, epoch_minutes=60, forward_fill=True)
        assert len(result) == 2

        epoch_11 = result[result["epoch"] == pd.Timestamp("2024-01-01 11:00")]
        # SBP should be forward-filled from epoch 10:00
        assert not pd.isna(epoch_11["sbp"].iloc[0])

    def test_per_patient_binning(self) -> None:
        """Two patients at same time -> separate rows, not merged."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:05",
                        "2024-01-01 10:15",
                    ]
                ),
                "heart_rate": [80.0, 100.0],
            }
        )
        result = bin_to_epochs(df, epoch_minutes=60, forward_fill=False)
        assert len(result) == 2

        p1 = result[result["patient_id"] == "P1"]
        p2 = result[result["patient_id"] == "P2"]
        assert p1["heart_rate"].iloc[0] == 80.0
        assert p2["heart_rate"].iloc[0] == 100.0

    def test_gcs_uses_max_not_median(self) -> None:
        """GCS [12, 15] in same epoch -> max = 15 (best neurological state)."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P1"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:05",
                        "2024-01-01 10:45",
                    ]
                ),
                "gcs": [12.0, 15.0],
            }
        )
        result = bin_to_epochs(df, epoch_minutes=60, forward_fill=False)
        assert len(result) == 1
        assert result["gcs"].iloc[0] == 15.0

    def test_custom_epoch_minutes(self) -> None:
        """30-minute epochs should split hourly data into two bins."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P1"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:10",
                        "2024-01-01 10:40",
                    ]
                ),
                "heart_rate": [80.0, 90.0],
            }
        )
        result = bin_to_epochs(df, epoch_minutes=30, forward_fill=False)
        assert len(result) == 2

    def test_forward_fill_does_not_cross_patients(self) -> None:
        """Forward-fill should not leak data across patients."""
        df = pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00",
                        "2024-01-01 11:00",
                    ]
                ),
                "heart_rate": [80.0, None],
                "sbp": [120.0, None],
            }
        )
        result = bin_to_epochs(df, epoch_minutes=60, forward_fill=True)
        p2 = result[result["patient_id"] == "P2"]
        # P2's SBP should remain NaN -- no forward-fill from P1
        assert pd.isna(p2["sbp"].iloc[0])


# ---------------------------------------------------------------------------
# TestDataUnification
# ---------------------------------------------------------------------------
class TestDataUnification:
    """Test multi-source data unification."""

    def test_unify_aligns_columns(self) -> None:
        """df1 has HR only, df2 has HR+Temp -> unified has both columns."""
        df1 = pd.DataFrame(
            {
                "patient_id": ["P1"],
                "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
                "heart_rate": [80.0],
            }
        )
        df2 = pd.DataFrame(
            {
                "patient_id": ["P2"],
                "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
                "heart_rate": [90.0],
                "temperature": [37.0],
            }
        )
        result = unify_datasets([df1, df2])
        assert "heart_rate" in result.columns
        assert "temperature" in result.columns
        assert len(result) == 2

    def test_unify_deduplicates(self) -> None:
        """Same (patient_id, timestamp) from two sources -> one row in output."""
        df1 = pd.DataFrame(
            {
                "patient_id": ["P1"],
                "timestamp": pd.to_datetime(["2024-01-01 10:05"]),
                "heart_rate": [80.0],
            }
        )
        df2 = pd.DataFrame(
            {
                "patient_id": ["P1"],
                "timestamp": pd.to_datetime(["2024-01-01 10:15"]),
                "heart_rate": [82.0],
                "temperature": [37.0],
            }
        )
        result = unify_datasets([df1, df2])
        # Both timestamps fall in epoch 10:00 for patient P1
        p1_rows = result[result["patient_id"] == "P1"]
        assert len(p1_rows) == 1

    def test_unify_sorts_by_patient_and_epoch(self) -> None:
        """Output should be sorted by (patient_id, epoch)."""
        df1 = pd.DataFrame(
            {
                "patient_id": ["P2", "P1"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 11:00", "2024-01-01 10:00"]
                ),
                "heart_rate": [90.0, 80.0],
            }
        )
        result = unify_datasets([df1])
        assert list(result["patient_id"]) == ["P1", "P2"]

    def test_unify_keeps_row_with_most_values(self) -> None:
        """When deduplicating, keep the row with more non-null vitals."""
        df1 = pd.DataFrame(
            {
                "patient_id": ["P1"],
                "timestamp": pd.to_datetime(["2024-01-01 10:05"]),
                "heart_rate": [80.0],
                "temperature": [None],
            }
        )
        df2 = pd.DataFrame(
            {
                "patient_id": ["P1"],
                "timestamp": pd.to_datetime(["2024-01-01 10:20"]),
                "heart_rate": [82.0],
                "temperature": [37.5],
            }
        )
        result = unify_datasets([df1, df2])
        p1 = result[result["patient_id"] == "P1"]
        assert len(p1) == 1
        # The row with both HR and temp should win
        assert not pd.isna(p1["temperature"].iloc[0])

    def test_unify_logs_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """unify_datasets should log observation and patient counts."""
        import logging

        df1 = pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00", "2024-01-01 10:00"]
                ),
                "heart_rate": [80.0, 90.0],
            }
        )
        with caplog.at_level(logging.INFO):
            unify_datasets([df1])
        assert any("Unified dataset" in msg for msg in caplog.messages)
