"""Tests for the Sepsis-3 labeling module.

Covers SOFA component scoring, composite SOFA, suspected infection
detection, sepsis onset derivation, and observation labeling.
"""
from __future__ import annotations

import pandas as pd
import pytest

from sepsis_vitals.ml.sepsis3_labeler import (
    compute_sofa_scores,
    derive_sepsis_onset,
    find_suspected_infections,
    label_observations,
    sofa_cardiovascular,
    sofa_cns,
    sofa_coagulation,
    sofa_liver,
    sofa_renal,
    sofa_respiratory,
)


# ---------------------------------------------------------------------------
# TestSOFAComponents — 28 tests across 6 components + None handling
# ---------------------------------------------------------------------------
class TestSOFAComponents:
    """Test individual SOFA component scoring functions."""

    # -- Respiratory (PaO2/FiO2) --
    def test_respiratory_score_0(self) -> None:
        assert sofa_respiratory(450) == 0

    def test_respiratory_score_1(self) -> None:
        assert sofa_respiratory(350) == 1

    def test_respiratory_score_2(self) -> None:
        assert sofa_respiratory(250) == 2

    def test_respiratory_score_3(self) -> None:
        assert sofa_respiratory(150) == 3

    def test_respiratory_score_4(self) -> None:
        assert sofa_respiratory(80) == 4

    def test_respiratory_none(self) -> None:
        assert sofa_respiratory(None) == 0

    # Boundary: exactly 400 is score 0
    def test_respiratory_boundary_400(self) -> None:
        assert sofa_respiratory(400) == 0

    # -- Coagulation (Platelets) --
    def test_coagulation_score_0(self) -> None:
        assert sofa_coagulation(200) == 0

    def test_coagulation_score_1(self) -> None:
        assert sofa_coagulation(120) == 1

    def test_coagulation_score_2(self) -> None:
        assert sofa_coagulation(75) == 2

    def test_coagulation_score_3(self) -> None:
        assert sofa_coagulation(30) == 3

    def test_coagulation_score_4(self) -> None:
        assert sofa_coagulation(15) == 4

    def test_coagulation_none(self) -> None:
        assert sofa_coagulation(None) == 0

    # -- Liver (Bilirubin) --
    def test_liver_score_0(self) -> None:
        assert sofa_liver(0.8) == 0

    def test_liver_score_1(self) -> None:
        assert sofa_liver(1.5) == 1

    def test_liver_score_2(self) -> None:
        assert sofa_liver(3.0) == 2

    def test_liver_score_3(self) -> None:
        assert sofa_liver(8.0) == 3

    def test_liver_score_4(self) -> None:
        assert sofa_liver(13.0) == 4

    def test_liver_none(self) -> None:
        assert sofa_liver(None) == 0

    # -- Cardiovascular --
    def test_cardiovascular_score_0(self) -> None:
        assert sofa_cardiovascular(map_mmhg=75) == 0

    def test_cardiovascular_score_1(self) -> None:
        assert sofa_cardiovascular(map_mmhg=65) == 1

    def test_cardiovascular_score_2_dopamine(self) -> None:
        assert sofa_cardiovascular(map_mmhg=65, dopamine_rate=4.0) == 2

    def test_cardiovascular_score_2_dobutamine(self) -> None:
        assert sofa_cardiovascular(map_mmhg=65, dobutamine_rate=3.0) == 2

    def test_cardiovascular_score_3(self) -> None:
        assert sofa_cardiovascular(map_mmhg=65, dopamine_rate=8.0) == 3

    def test_cardiovascular_score_3_norepi(self) -> None:
        assert sofa_cardiovascular(map_mmhg=65, norepinephrine_rate=0.05) == 3

    def test_cardiovascular_score_4_dopamine(self) -> None:
        assert sofa_cardiovascular(map_mmhg=65, dopamine_rate=16.0) == 4

    def test_cardiovascular_score_4_norepi(self) -> None:
        assert sofa_cardiovascular(map_mmhg=65, norepinephrine_rate=0.2) == 4

    def test_cardiovascular_none(self) -> None:
        assert sofa_cardiovascular() == 0

    # -- CNS (GCS) --
    def test_cns_score_0(self) -> None:
        assert sofa_cns(15) == 0

    def test_cns_score_1(self) -> None:
        assert sofa_cns(13) == 1

    def test_cns_score_2(self) -> None:
        assert sofa_cns(11) == 2

    def test_cns_score_3(self) -> None:
        assert sofa_cns(7) == 3

    def test_cns_score_4(self) -> None:
        assert sofa_cns(5) == 4

    def test_cns_none(self) -> None:
        assert sofa_cns(None) == 0

    # -- Renal (Creatinine) --
    def test_renal_score_0(self) -> None:
        assert sofa_renal(0.9) == 0

    def test_renal_score_1(self) -> None:
        assert sofa_renal(1.5) == 1

    def test_renal_score_2(self) -> None:
        assert sofa_renal(2.5) == 2

    def test_renal_score_3(self) -> None:
        assert sofa_renal(4.0) == 3

    def test_renal_score_4(self) -> None:
        assert sofa_renal(5.5) == 4

    def test_renal_none(self) -> None:
        assert sofa_renal(None) == 0


# ---------------------------------------------------------------------------
# TestCompositeSOFA
# ---------------------------------------------------------------------------
class TestCompositeSOFA:
    """Test composite SOFA score computation on DataFrames."""

    def test_compute_sofa_single_row(self) -> None:
        """All normal values should yield SOFA total = 0."""
        df = pd.DataFrame(
            {
                "gcs": [15],
                "map": [80],
                "platelets": [250],
                "bilirubin_total": [0.5],
                "creatinine": [0.8],
                "pao2_fio2": [450.0],
            }
        )
        result = compute_sofa_scores(df)
        assert result["sofa_total"].iloc[0] == 0

    def test_compute_sofa_organ_dysfunction(self) -> None:
        """Multiple organ dysfunction should sum correctly.

        GCS 12 = 2, MAP 65 = 1, platelets 80 = 2, bilirubin 3.0 = 2,
        creatinine 2.5 = 2, pao2_fio2 250 = 2 => total = 11.
        """
        df = pd.DataFrame(
            {
                "gcs": [12],
                "map": [65],
                "platelets": [80],
                "bilirubin_total": [3.0],
                "creatinine": [2.5],
                "pao2_fio2": [250.0],
            }
        )
        result = compute_sofa_scores(df)
        assert result["sofa_total"].iloc[0] == 11

    def test_compute_sofa_missing_labs(self) -> None:
        """Only GCS and MAP provided; missing labs default to score 0.

        GCS 10 = 2, MAP 75 = 0 => total = 2.
        """
        df = pd.DataFrame({"gcs": [10], "map": [75]})
        result = compute_sofa_scores(df)
        assert result["sofa_total"].iloc[0] == 2


# ---------------------------------------------------------------------------
# TestSuspectedInfection
# ---------------------------------------------------------------------------
class TestSuspectedInfection:
    """Test suspected infection detection from antibiotics + cultures."""

    def _make_antibiotics(
        self, subject_id: int, hadm_id: int, starttime: str, drug: str
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "subject_id": [subject_id],
                "hadm_id": [hadm_id],
                "starttime": pd.to_datetime([starttime]),
                "drug": [drug],
            }
        )

    def _make_cultures(
        self, subject_id: int, hadm_id: int, charttime: str, spec_type_desc: str
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "subject_id": [subject_id],
                "hadm_id": [hadm_id],
                "charttime": pd.to_datetime([charttime]),
                "spec_type_desc": [spec_type_desc],
            }
        )

    def test_abx_and_culture_within_72h(self) -> None:
        """Culture and abx at the same time should detect infection."""
        abx = self._make_antibiotics(1, 100, "2024-01-01 10:00", "vancomycin")
        cul = self._make_cultures(1, 100, "2024-01-01 10:00", "BLOOD CULTURE")
        result = find_suspected_infections(abx, cul)
        assert len(result) == 1
        assert result["subject_id"].iloc[0] == 1
        assert result["hadm_id"].iloc[0] == 100

    def test_abx_before_culture_within_72h(self) -> None:
        """Abx 24h before culture should still detect infection."""
        abx = self._make_antibiotics(1, 100, "2024-01-01 10:00", "ceftriaxone")
        cul = self._make_cultures(1, 100, "2024-01-02 10:00", "BLOOD CULTURE")
        result = find_suspected_infections(abx, cul)
        assert len(result) == 1
        # t_suspected should be the earlier of the two
        assert result["t_suspected_infection"].iloc[0] == pd.Timestamp("2024-01-01 10:00")

    def test_abx_and_culture_beyond_72h(self) -> None:
        """Abx and culture >72h apart should not detect infection."""
        abx = self._make_antibiotics(1, 100, "2024-01-01 10:00", "vancomycin")
        cul = self._make_cultures(1, 100, "2024-01-05 10:00", "BLOOD CULTURE")
        result = find_suspected_infections(abx, cul)
        assert len(result) == 0

    def test_no_culture_no_infection(self) -> None:
        """No cultures should yield no suspected infections."""
        abx = self._make_antibiotics(1, 100, "2024-01-01 10:00", "vancomycin")
        cul = pd.DataFrame(
            columns=["subject_id", "hadm_id", "charttime", "spec_type_desc"]
        )
        result = find_suspected_infections(abx, cul)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestSepsisOnset
# ---------------------------------------------------------------------------
class TestSepsisOnset:
    """Test sepsis onset derivation from SOFA + infection timing."""

    def test_sofa_increase_after_infection(self) -> None:
        """SOFA increase >= 2 within 48h of infection -> onset detected."""
        sofa_series = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hadm_id": [100, 100, 100],
                "charttime": pd.to_datetime(
                    ["2024-01-01 06:00", "2024-01-01 12:00", "2024-01-01 18:00"]
                ),
                "sofa_total": [1, 1, 4],
            }
        )
        infections = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [100],
                "t_suspected_infection": pd.to_datetime(["2024-01-01 10:00"]),
            }
        )
        result = derive_sepsis_onset(sofa_series, infections)
        assert len(result) == 1
        assert result["hadm_id"].iloc[0] == 100

    def test_no_sofa_increase(self) -> None:
        """No SOFA increase >= 2 -> no onset."""
        sofa_series = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hadm_id": [100, 100, 100],
                "charttime": pd.to_datetime(
                    ["2024-01-01 06:00", "2024-01-01 12:00", "2024-01-01 18:00"]
                ),
                "sofa_total": [1, 1, 2],
            }
        )
        infections = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [100],
                "t_suspected_infection": pd.to_datetime(["2024-01-01 10:00"]),
            }
        )
        result = derive_sepsis_onset(sofa_series, infections)
        assert len(result) == 0

    def test_sofa_increase_outside_48h_window(self) -> None:
        """SOFA increase >48h from infection -> no onset."""
        sofa_series = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hadm_id": [100, 100, 100],
                "charttime": pd.to_datetime(
                    ["2024-01-01 06:00", "2024-01-01 12:00", "2024-01-05 00:00"]
                ),
                "sofa_total": [1, 1, 5],
            }
        )
        infections = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [100],
                "t_suspected_infection": pd.to_datetime(["2024-01-01 10:00"]),
            }
        )
        result = derive_sepsis_onset(sofa_series, infections)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestObservationLabeling
# ---------------------------------------------------------------------------
class TestObservationLabeling:
    """Test per-observation binary labeling."""

    def test_label_observations_with_onset(self) -> None:
        """Observations at/after onset should be labeled 1."""
        observations = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hadm_id": [100, 100, 100],
                "charttime": pd.to_datetime(
                    ["2024-01-01 06:00", "2024-01-01 12:00", "2024-01-01 18:00"]
                ),
                "hr": [80, 90, 110],
            }
        )
        onsets = pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [100],
                "t_sepsis_onset": pd.to_datetime(["2024-01-01 12:00"]),
            }
        )
        result = label_observations(observations, onsets)
        assert list(result["sepsis_label"]) == [0, 1, 1]
        assert all(result["label_source"] == "sepsis3")

    def test_label_observations_no_sepsis(self) -> None:
        """No onset -> all labels 0."""
        observations = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [100, 100],
                "charttime": pd.to_datetime(
                    ["2024-01-01 06:00", "2024-01-01 12:00"]
                ),
                "hr": [80, 85],
            }
        )
        onsets = pd.DataFrame(
            columns=["subject_id", "hadm_id", "t_sepsis_onset"]
        )
        result = label_observations(observations, onsets)
        assert list(result["sepsis_label"]) == [0, 0]
        assert all(result["label_source"] == "sepsis3")

    def test_label_with_icd_fallback(self) -> None:
        """ICD fallback hadms without Sepsis-3 onset get label=1 + source='icd_fallback'."""
        observations = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "hadm_id": [100, 100, 200, 200],
                "charttime": pd.to_datetime(
                    [
                        "2024-01-01 06:00",
                        "2024-01-01 12:00",
                        "2024-01-01 06:00",
                        "2024-01-01 12:00",
                    ]
                ),
                "hr": [80, 85, 90, 95],
            }
        )
        onsets = pd.DataFrame(
            columns=["subject_id", "hadm_id", "t_sepsis_onset"]
        )
        icd_fallback_hadms = {100, 200}
        result = label_observations(observations, onsets, icd_fallback_hadms=icd_fallback_hadms)
        # All observations in ICD fallback hadms should be labeled 1
        assert list(result["sepsis_label"]) == [1, 1, 1, 1]
        assert all(result["label_source"] == "icd_fallback")
