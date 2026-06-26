"""Tests for CaseLibrary — MIMIC-IV case indexing and lookup."""

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


class TestCaseLibrary:
    """Test case indexing and lookup."""

    @pytest.fixture
    def sample_stays(self):
        """Fake derive_sepsis_labels output."""
        return pd.DataFrame({
            "subject_id": [1001, 1002, 1003, 1004, 1005],
            "hadm_id": [2001, 2002, 2003, 2004, 2005],
            "stay_id": [3001, 3002, 3003, 3004, 3005],
            "intime": pd.to_datetime([
                "2150-01-01 08:00", "2150-01-02 10:00", "2150-01-03 06:00",
                "2150-01-04 14:00", "2150-01-05 09:00",
            ]),
            "outtime": pd.to_datetime([
                "2150-01-03 08:00", "2150-01-04 10:00", "2150-01-04 06:00",
                "2150-01-06 14:00", "2150-01-06 09:00",
            ]),
            "sepsis_label": [1, 0, 1, 0, 1],
            "label_source": ["sepsis3", "none", "sepsis3", "none", "icd_fallback"],
        })

    @pytest.fixture
    def sample_demographics(self):
        """Fake demographics output."""
        return pd.DataFrame({
            "hadm_id": [2001, 2002, 2003, 2004, 2005],
            "age_years": [65, 45, 72, 55, 80],
            "sex_m": [1, 0, 1, 0, 1],
        })

    @pytest.fixture
    def sample_vitals(self):
        """Fake vitals output — varying observation counts per stay."""
        rows = []
        obs_counts = {3001: 20, 3002: 15, 3003: 8, 3004: 30, 3005: 12}
        for stay_id, n_obs in obs_counts.items():
            for i in range(n_obs):
                rows.append({
                    "stay_id": stay_id,
                    "charttime": pd.Timestamp("2150-01-01") + pd.Timedelta(hours=i),
                    "vital_name": "heart_rate",
                    "valuenum": 80 + np.random.randn() * 5,
                })
        return pd.DataFrame(rows)

    def test_build_index_creates_sqlite(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(
            stays=sample_stays,
            demographics=sample_demographics,
            vitals=sample_vitals,
        )

        # SQLite file should exist
        assert (tmp_path / "cases.db").exists()

        # Should have 5 cases
        cases = lib.list_cases()
        assert len(cases) == 5

    def test_get_sepsis_cases(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        sepsis_cases = lib.get_sepsis_cases()
        assert len(sepsis_cases) == 3  # subjects 1001, 1003, 1005
        assert all(c["sepsis_label"] == 1 for c in sepsis_cases)

    def test_get_case_by_subject_id(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_case(subject_id=1001)
        assert case is not None
        assert case["subject_id"] == 1001
        assert case["age_years"] == 65
        assert case["sex"] == "M"
        assert case["sepsis_label"] == 1
        assert case["n_observations"] == 20

    def test_get_case_unknown_returns_none(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        assert lib.get_case(subject_id=9999) is None

    def test_get_random_case_sepsis(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_random_case(sepsis=True)
        assert case is not None
        assert case["sepsis_label"] == 1

    def test_get_random_case_nonsepsis(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_random_case(sepsis=False)
        assert case is not None
        assert case["sepsis_label"] == 0

    def test_icu_los_hours(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_case(subject_id=1001)
        # 2150-01-01 08:00 -> 2150-01-03 08:00 = 48 hours
        assert abs(case["icu_los_hours"] - 48.0) < 0.1
