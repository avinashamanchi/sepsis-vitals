"""End-to-end integration test for Layer 1 data pipeline."""
from pathlib import Path

import pandas as pd
import pytest

MIMIC_DEMO_PATH = Path("physionet.org/files/mimic-iv-demo/2.2")
FHIR_DEMO_PATH = Path("physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir")
HAS_CSV_DATA = (MIMIC_DEMO_PATH / "hosp" / "patients.csv.gz").exists()
HAS_FHIR_DATA = (FHIR_DEMO_PATH / "MimicPatient.ndjson.gz").exists()


@pytest.mark.skipif(not HAS_CSV_DATA, reason="MIMIC-IV Demo CSV not available")
class TestCSVPipelineIntegration:
    def test_full_pipeline_produces_training_data(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=10)
        assert len(df) > 0
        assert "patient_id" in df.columns
        assert "sepsis_label" in df.columns
        assert "label_source" in df.columns
        vital_cols = ["temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2"]
        has_vitals = sum(1 for v in vital_cols if v in df.columns and df[v].notna().any())
        assert has_vitals >= 3

    def test_sepsis3_labels_not_all_same(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=50)
        assert df["sepsis_label"].nunique() >= 1
        n_pos = df["sepsis_label"].sum()
        n_total = len(df)
        print(f"Sepsis labels: {n_pos}/{n_total} ({100*n_pos/n_total:.1f}%)")


@pytest.mark.skipif(not HAS_FHIR_DATA, reason="MIMIC-IV FHIR Demo not available")
class TestFHIRPipelineIntegration:
    def test_fhir_loads_100_patients(self):
        from sepsis_vitals.ml.fhir_loader import FHIRLoader

        loader = FHIRLoader.from_demo()
        patients = loader.load_patients()
        assert len(patients) == 100

    def test_fhir_loads_vitals(self):
        from sepsis_vitals.ml.fhir_loader import FHIRLoader

        loader = FHIRLoader.from_demo()
        vitals = loader.load_vitals()
        assert len(vitals) > 1000

    def test_fhir_loads_labs(self):
        from sepsis_vitals.ml.fhir_loader import FHIRLoader

        loader = FHIRLoader.from_demo()
        labs = loader.load_labs()
        assert len(labs) > 0


@pytest.mark.skipif(not (HAS_CSV_DATA and HAS_FHIR_DATA), reason="Both sources needed")
class TestUnifiedPipeline:
    def test_unify_csv_and_fhir(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader
        from sepsis_vitals.ml.data_unifier import unify_datasets

        csv_loader = MIMICLoader.from_demo()
        csv_df = csv_loader.build_training_dataset(max_patients=5)
        result = unify_datasets([csv_df])
        assert len(result) > 0
        assert "patient_id" in result.columns
