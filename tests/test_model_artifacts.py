"""
tests/test_model_artifacts.py — artifact integrity, compatibility and the
liveness / readiness / prediction-readiness split.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None or not (MODELS / "manifest.json").exists(),
    reason="scikit-learn or committed model artifacts unavailable",
)


@pytest.fixture()
def model_copy(tmp_path):
    dest = tmp_path / "models"
    dest.mkdir()
    for name in ("sepsis_model.joblib", "model_metadata.json", "imputation_medians.json", "manifest.json"):
        shutil.copy(MODELS / name, dest / name)
    return dest


def test_committed_model_verifies_but_is_not_clinically_ready():
    from sepsis_vitals.ml.artifacts import verify_artifacts

    status = verify_artifacts(MODELS)
    assert status.state == "ready"
    assert status.validation_status == "synthetic-development"
    assert status.as_dict()["clinically_ready"] is False
    assert status.as_dict()["clinical_use"] == "not-permitted"


def test_tampered_artifact_is_rejected_before_unpickling(model_copy):
    from sepsis_vitals.ml.artifacts import ModelArtifactError, verify_artifacts

    with open(model_copy / "sepsis_model.joblib", "ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(ModelArtifactError) as err:
        verify_artifacts(model_copy)
    assert err.value.state == "invalid" and "Checksum" in err.value.reason


def test_missing_manifest_is_refused_unless_explicitly_allowed(model_copy):
    from sepsis_vitals.ml.artifacts import ModelArtifactError, verify_artifacts

    (model_copy / "manifest.json").unlink()
    with pytest.raises(ModelArtifactError):
        verify_artifacts(model_copy)
    status = verify_artifacts(model_copy, allow_unverified=True)
    assert status.state == "unverified" and status.validation_status == "unvalidated"


def test_runtime_version_mismatch_is_incompatible(model_copy):
    from sepsis_vitals.ml.artifacts import ModelArtifactError, verify_artifacts

    manifest = json.loads((model_copy / "manifest.json").read_text())
    manifest["runtime"]["scikit_learn"] = "0.0.1"
    (model_copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ModelArtifactError) as err:
        verify_artifacts(model_copy)
    assert err.value.state == "incompatible"


def test_feature_schema_change_is_incompatible(model_copy):
    from sepsis_vitals.ml.artifacts import ModelArtifactError, verify_artifacts

    manifest = json.loads((model_copy / "manifest.json").read_text())
    manifest["artifacts"].pop("model_metadata.json")  # isolate the schema check
    (model_copy / "manifest.json").write_text(json.dumps(manifest))
    metadata = json.loads((model_copy / "model_metadata.json").read_text())
    metadata["feature_names"] = metadata["feature_names"][:-1]
    (model_copy / "model_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ModelArtifactError) as err:
        verify_artifacts(model_copy)
    assert err.value.state == "incompatible"


def test_unknown_validation_status_cannot_be_written():
    from sepsis_vitals.ml.artifacts import build_manifest

    with pytest.raises(ValueError):
        build_manifest(MODELS, "clinically-validated")


# -- API: liveness, readiness, prediction readiness -----------------------------

@pytest.fixture()
def client_with_models(monkeypatch, tmp_path):
    def _make(model_dir):
        from fastapi.testclient import TestClient

        import sepsis_vitals.api as api

        monkeypatch.setenv("SEPSIS_MODEL_DIR", str(model_dir))
        monkeypatch.setenv("SEPSIS_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setattr(api, "_predictor", None)
        monkeypatch.setattr(api, "_auth_enabled", False)
        for dep in (api.check_rate_limit, api.check_ml_rate_limit):
            api.app.dependency_overrides[dep] = lambda: None
        return TestClient(api.app)

    yield _make
    import sepsis_vitals.api as api

    api.app.dependency_overrides.clear()


def test_model_present_reports_ready_but_never_clinically_ready(client_with_models, model_copy):
    with client_with_models(model_copy) as client:
        status = client.get("/model/status").json()
        assert status["state"] == "ready" and status["prediction_ready"] is True
        assert status["clinically_ready"] is False
        resp = client.post("/predict", json={
            "patient_id": "artifact-probe",
            "vitals": {"heart_rate": 118, "resp_rate": 24, "sbp": 96, "temperature": 38.6, "lactate": 2.4},
            "age_years": 70,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["validation_status"] == "synthetic-development"
        assert body["clinical_use"] == "not-permitted"
        assert body["provenance"]["artifact_sha256"] == status["artifact_sha256"]
        # rules flag this patient; the model may not lower that
        assert body["risk_level"] in {"high", "critical"} and body["alert"] is True


def test_model_absent_is_explicit_and_api_stays_live(client_with_models, tmp_path):
    empty = tmp_path / "no-models"
    empty.mkdir()
    with client_with_models(empty) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200  # API serves without a model
        status = client.get("/model/status").json()
        assert status["state"] == "absent" and status["prediction_ready"] is False
        resp = client.post("/predict", json={
            "patient_id": "x", "vitals": {"heart_rate": 90, "resp_rate": 18, "sbp": 120},
        })
        assert resp.status_code == 503
        assert resp.json()["detail"]["model_state"] == "absent"


def test_tampered_model_is_not_loaded_by_the_api(client_with_models, model_copy):
    with open(model_copy / "sepsis_model.joblib", "ab") as fh:
        fh.write(b"\x00")
    with client_with_models(model_copy) as client:
        status = client.get("/model/status").json()
        assert status["state"] == "invalid" and status["prediction_ready"] is False
