"""
tests/test_mimic_loader_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for MIMICLoader using the real MIMIC-IV Demo 2.2 dataset.
Tests are skipped if the demo data is not available locally.
"""

from pathlib import Path

import pandas as pd
import pytest

MIMIC_DEMO_PATH = Path("physionet.org/files/mimic-iv-demo/2.2")
HAS_DEMO_DATA = (MIMIC_DEMO_PATH / "hosp" / "patients.csv.gz").exists()


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="MIMIC-IV Demo data not available")
class TestMIMICLoaderDemo:
    """Integration tests that exercise MIMICLoader against real MIMIC-IV Demo data."""

    def test_from_demo_creates_loader(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        assert loader.root == MIMIC_DEMO_PATH

    def test_load_demographics(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        demo = loader.load_demographics()

        assert "subject_id" in demo.columns
        assert "hadm_id" in demo.columns
        assert "age_years" in demo.columns
        assert "sex_m" in demo.columns
        assert len(demo) > 0

        # Ages should be in a reasonable range (0-120)
        assert demo["age_years"].between(0, 120).all()

    def test_load_vitals(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()

        # Load ICU stays to get first 5 stay_ids
        stays = loader._read_csv(
            "icu/icustays.csv.gz", usecols=["stay_id"]
        )
        first_5 = set(stays["stay_id"].unique()[:5])

        vitals = loader.load_vitals(stay_ids=first_5)

        assert "stay_id" in vitals.columns
        assert "vital_name" in vitals.columns
        assert "valuenum" in vitals.columns
        assert len(vitals) > 0

    def test_load_sofa_labs(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        sofa_labs = loader.load_sofa_labs()

        assert "hadm_id" in sofa_labs.columns
        assert "charttime" in sofa_labs.columns
        assert len(sofa_labs) > 0

    def test_load_antibiotics(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        abx = loader.load_antibiotics()

        assert "subject_id" in abx.columns
        assert "hadm_id" in abx.columns
        assert "starttime" in abx.columns
        assert "drug" in abx.columns
        assert len(abx) > 0

    def test_load_cultures(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        cultures = loader.load_cultures()

        assert "subject_id" in cultures.columns
        assert "hadm_id" in cultures.columns
        assert "charttime" in cultures.columns
        assert "spec_type_desc" in cultures.columns
        assert len(cultures) > 0

    def test_derive_sepsis3_labels(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        labels = loader.derive_sepsis_labels()

        assert "stay_id" in labels.columns
        assert "sepsis_label" in labels.columns
        assert "label_source" in labels.columns
        assert len(labels) > 0

        # Should have a mix of 0s and 1s
        assert set(labels["sepsis_label"].unique()) == {0, 1}

    def test_build_training_dataset_small(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=5)

        assert "patient_id" in df.columns
        assert "sepsis_label" in df.columns
        assert "label_source" in df.columns
        assert len(df) > 0

        # Should have some vital columns present
        vital_cols = ["temperature", "heart_rate", "resp_rate", "sbp"]
        has_any_vital = any(c in df.columns for c in vital_cols)
        assert has_any_vital

    def test_build_training_dataset_has_epoch_column(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=5)
        assert "timestamp" in df.columns or "epoch" in df.columns

    def test_build_training_dataset_no_duplicate_epochs(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=5)
        time_col = "timestamp" if "timestamp" in df.columns else "epoch"
        dupes = df.duplicated(subset=["patient_id", time_col])
        assert not dupes.any(), f"Found {dupes.sum()} duplicate (patient_id, time) pairs"
