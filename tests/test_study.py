"""
tests/test_study.py — outcome recording and local evaluation for a
silent-mode study. Synthetic patients only.
"""

from __future__ import annotations

import importlib.util
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None or not (MODELS / "manifest.json").exists(),
    reason="scikit-learn or committed model artifacts unavailable",
)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    import json

    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api

    dest = tmp_path / "models"
    dest.mkdir()
    for name in list(json.loads((MODELS / "manifest.json").read_text())["artifacts"]) + ["manifest.json"]:
        shutil.copy(MODELS / name, dest / name)
    monkeypatch.setenv("SEPSIS_MODEL_DIR", str(dest))
    monkeypatch.setenv("SEPSIS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SEPSIS_MODEL_DISPLAY", "silent")
    monkeypatch.setattr(api, "_predictor", None)
    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", False)
    monkeypatch.setattr(api.ws_manager, "broadcast", AsyncMock())
    for dep in (api.check_rate_limit, api.check_ml_rate_limit):
        api.app.dependency_overrides[dep] = lambda: None
    with TestClient(api.app) as c:
        yield c
    api.app.dependency_overrides.clear()


SICK = {"heart_rate": 128, "resp_rate": 31, "sbp": 84, "temperature": 39.2, "spo2": 90, "gcs": 13}
WELL = {"heart_rate": 74, "resp_rate": 14, "sbp": 126, "temperature": 36.8, "spo2": 98, "gcs": 15}


def _patient(client, n, site):
    resp = client.post("/patients", json={"external_id": f"SYNTH-STUDY-{site}-{n}", "site_id": site,
                                          "age_years": 50 + n})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_outcomes_and_local_evaluation(client):
    from sepsis_vitals.db import SessionLocal
    from sepsis_vitals.study.evaluation import evaluate_local_study

    site = f"STUDY-{datetime.now().strftime('%H%M%S%f')}"
    now = datetime.now(timezone.utc)
    for n in range(12):
        pid = _patient(client, n, site)
        septic = n % 2 == 0
        resp = client.post("/predict", json={"patient_id": pid, "vitals": SICK if septic else WELL})
        assert resp.status_code == 200 and resp.json()["model_output_withheld"] is True
        body = {"patient_id": pid, "sepsis": septic, "definition": "Synthetic reference standard v0"}
        if septic:
            body["onset_at"] = (now + timedelta(seconds=1)).isoformat()
        assert client.post("/study/outcomes", json=body).status_code == 201, body

    summary = client.get("/study/summary", params={"site_id": site}).json()
    assert summary == {"site_id": site, "patients_with_predictions": 12, "patients_with_outcome": 12,
                       "sepsis_positive": 6}
    db = SessionLocal()
    try:
        report = evaluate_local_study(db, site_id=site, n_boot=50)
    finally:
        db.close()
    assert report["status"] == "computed" and report["patients"] == 12
    assert set(report["discrimination"]["arms"]) >= {"model", "uva", "news2", "qsofa"}
    assert report["discrimination"]["arms"]["uva"]["auroc"]["estimate"] == 1.0  # synthetic extremes
    assert report["alerting"]["positives_with_onset"] == 6
    assert "SYNTH-STUDY" not in str(report)  # aggregates only


@pytest.mark.parametrize("body, detail", [
    ({"sepsis": False, "onset_at": "2026-01-01T00:00:00Z", "definition": "v0"}, "onset_at is only"),
    ({"sepsis": True, "onset_at": "2026-01-01T00:00:00", "definition": "v0"}, "UTC offset"),
    ({"sepsis": True, "definition": "v0", "notes": "free text"}, "Extra inputs"),
])
def test_outcome_validation(client, body, detail):
    pid = _patient(client, 99, f"STUDY-V-{datetime.now().strftime('%f')}")
    resp = client.post("/study/outcomes", json={"patient_id": pid, **body})
    assert resp.status_code == 422 and detail in resp.text


def test_outcomes_need_a_study_role(client, monkeypatch):
    monkeypatch.setattr("sepsis_vitals.dependencies._anonymous_user",
                        lambda: {"id": "anonymous", "role": "nurse", "org_id": "WARD-A"})
    resp = client.post("/study/outcomes", json={"patient_id": "00000000-0000-0000-0000-000000000001",
                                                "sepsis": True, "definition": "v0"})
    assert resp.status_code == 403


def test_insufficient_outcomes_are_reported_not_computed():
    from sepsis_vitals.db import SessionLocal, init_db
    from sepsis_vitals.study.evaluation import evaluate_local_study

    init_db()
    db = SessionLocal()
    try:
        report = evaluate_local_study(db, site_id="NO-SUCH-SITE", n_boot=10)
    finally:
        db.close()
    assert report["status"].startswith("insufficient")


def test_study_cli_writes_aggregates_only(tmp_path):
    import json

    from sepsis_vitals.study.__main__ import main

    assert main(["--site", "NO-SUCH-SITE", "--bootstrap", "10", "--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "local_study_evaluation.json").read_text())
    assert report["status"].startswith("insufficient") and "provenance" in report
