"""Tests for FHIR NDJSON streaming loader."""
from pathlib import Path
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

class TestFHIRPatientParsing:
    def test_parse_patients_from_ndjson(self):
        from sepsis_vitals.ml.fhir_loader import parse_patients
        patients = parse_patients(FIXTURES / "fhir_patient.ndjson")
        assert len(patients) == 2
        assert patients.iloc[0]["subject_id"] == 10007795
        assert patients.iloc[0]["gender"] == "female"
        assert patients.iloc[1]["subject_id"] == 10003400
        assert patients.iloc[1]["gender"] == "male"

    def test_parse_patients_has_uuid(self):
        from sepsis_vitals.ml.fhir_loader import parse_patients
        patients = parse_patients(FIXTURES / "fhir_patient.ndjson")
        assert patients.iloc[0]["fhir_id"] == "patient-001"
        assert patients.iloc[1]["fhir_id"] == "patient-002"

class TestFHIRObservationParsing:
    def test_parse_observations_from_ndjson(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations
        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        assert len(obs) == 3
        assert "patient_fhir_id" in obs.columns
        assert "itemid" in obs.columns
        assert "valuenum" in obs.columns
        assert "charttime" in obs.columns

    def test_fahrenheit_to_celsius_conversion(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations
        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        temp_obs = obs[obs["itemid"] == 223761]
        assert len(temp_obs) == 1
        assert abs(temp_obs.iloc[0]["valuenum"] - 37.0) < 0.1  # 98.6°F = 37.0°C

    def test_vital_name_mapping(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations
        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        hr_obs = obs[obs["vital_name"] == "heart_rate"]
        assert len(hr_obs) == 2

    def test_patient_linkage(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations
        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        patient1_obs = obs[obs["patient_fhir_id"] == "patient-001"]
        assert len(patient1_obs) == 2

class TestStreamingParsing:
    def test_stream_ndjson_yields_dicts(self):
        from sepsis_vitals.ml.fhir_loader import stream_ndjson
        records = list(stream_ndjson(FIXTURES / "fhir_patient.ndjson"))
        assert len(records) == 2
        assert records[0]["resourceType"] == "Patient"

    def test_stream_ndjson_gzipped(self):
        from sepsis_vitals.ml.fhir_loader import stream_ndjson
        gz_path = Path("physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/MimicPatient.ndjson.gz")
        if not gz_path.exists():
            pytest.skip("FHIR demo data not available")
        records = list(stream_ndjson(gz_path))
        assert len(records) == 100
