"""
tests/test_audited_inference.py — every risk result is traceable and recorded
before it is released, and nothing is released when it cannot be.

Covers /predict, /predict/timeseries, /predict/batch and the continuous
monitor: audit record per inference and alert decision, append-only and
tamper-evident records, fail-closed behaviour on model and audit failures
(no default or stale score), the time-series payload contract, and the
request log. Synthetic inputs only. Runs on SQLite and, in CI, PostgreSQL.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
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

VITALS = {"heart_rate": 118, "resp_rate": 24, "sbp": 96, "temperature": 38.6, "spo2": 93}


@pytest.fixture()
def model_dir(tmp_path):
    dest = tmp_path / "models"
    dest.mkdir()
    for name in ("sepsis_model.joblib", "model_metadata.json", "imputation_medians.json", "manifest.json"):
        shutil.copy(MODELS / name, dest / name)
    return dest


@pytest.fixture()
def client(monkeypatch, tmp_path, model_dir):
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api

    monkeypatch.setenv("SEPSIS_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SEPSIS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(api, "_predictor", None)
    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", False)
    monkeypatch.setattr(api.ws_manager, "broadcast", AsyncMock())
    for dep in (api.check_rate_limit, api.check_ml_rate_limit):
        api.app.dependency_overrides[dep] = lambda: None
    with TestClient(api.app) as c:
        yield c
    api.app.dependency_overrides.clear()


def _record(record_id):
    from sepsis_vitals.db import PredictionRecord, SessionLocal

    db = SessionLocal()
    try:
        record = db.get(PredictionRecord, record_id)
        if record is not None:
            db.expunge(record)
        return record
    finally:
        db.close()


def _count_records():
    from sepsis_vitals.db import PredictionRecord, SessionLocal

    db = SessionLocal()
    try:
        return db.query(PredictionRecord).count()
    finally:
        db.close()


# -- one sealed record per inference -------------------------------------------

def test_predict_returns_only_a_recorded_traceable_result(client):
    from sepsis_vitals.audit_records import sha256_of, verify_record

    resp = client.post("/predict", json={
        "patient_id": "synthetic-trace-1", "vitals": VITALS, "age_years": 70,
        "comorbidities": {"has_diabetes": 1},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert resp.headers["X-Request-ID"] == body["request_id"]

    record = _record(body["audit_record_id"])
    assert record is not None
    assert record.request_id == body["request_id"] and record.source == "api"
    assert record.risk_probability == body["risk_probability"]
    assert record.risk_level == body["risk_level"] and record.alert_fired == body["alert"]
    assert record.threshold_policy == body["threshold_policy"]["id"] == "placeholder-bands-v1"
    assert verify_record(record) is True

    payload = json.loads(record.input_vitals)
    assert payload["observation"] == VITALS and payload["age_years"] == 70
    assert payload["comorbidities"]["has_diabetes"] == 1
    assert record.input_sha256 == sha256_of(payload)

    output = json.loads(record.output_scores)
    assert output["versions"] == body["versions"]
    for key in ("model_version", "artifact_sha256", "feature_schema_sha256",
                "preprocessing_version", "threshold_policy_id", "threshold_policy_sha256"):
        assert body["versions"][key], key
    assert output["clinical_use"] == "not-permitted"


def test_response_states_decision_support_and_no_invalid_interval(client):
    body = client.post("/predict", json={"patient_id": "synthetic-ds", "vitals": VITALS}).json()
    assert "not a diagnosis" in body["decision_support_notice"]
    assert body["clinical_use"] == "not-permitted"
    assert body["confidence_interval"] is None
    assert body["uncertainty"]["method"] == "none"
    assert body["explanation"]["method"].startswith("median substitution")
    assert body["threshold_policy"]["status"].startswith("engineering-placeholder")
    codes = {w["code"] for w in body["data_quality"]["warnings"]}
    assert {"age_not_supplied", "comorbidities_not_supplied", "no_history"} <= codes


def test_explanation_is_per_patient(client):
    a = client.post("/predict", json={"patient_id": "synthetic-a", "vitals": VITALS, "age_years": 80}).json()
    b = client.post("/predict", json={
        "patient_id": "synthetic-b",
        "vitals": {"heart_rate": 72, "resp_rate": 14, "sbp": 130, "lactate": 4.1}, "age_years": 30,
    }).json()
    features_a = {f["feature"] for f in a["top_risk_factors"]}
    features_b = {f["feature"] for f in b["top_risk_factors"]}
    # Only inputs the patient supplied are attributed
    assert features_a == {"heart_rate", "resp_rate", "sbp", "temperature", "spo2", "age_years"}
    assert features_b == {"heart_rate", "resp_rate", "sbp", "lactate", "age_years"}
    for factor in a["top_risk_factors"]:
        assert set(factor) >= {"contribution", "direction", "current_value", "reference_value"}


# -- fail closed ----------------------------------------------------------------

def test_audit_failure_withholds_the_result(client, monkeypatch):
    import sepsis_vitals.api as api
    from sepsis_vitals.audit_records import AuditWriteError

    def refuse(**_):
        raise AuditWriteError("OperationalError")

    monkeypatch.setattr("sepsis_vitals.audit_records.record_prediction", refuse)
    before = _count_records()
    resp = client.post("/predict", json={"patient_id": "synthetic-audit-down", "vitals": VITALS})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail.startswith("No risk calculated") and "not a low-risk result" in detail
    assert "risk_probability" not in resp.text
    api.ws_manager.broadcast.assert_not_awaited()
    assert _count_records() == before
    trend = api._predictor.get_patient_trend("synthetic-audit-down")
    assert trend is None or trend["n_observations"] == 0  # nothing tracked


def test_model_failure_is_no_risk_not_a_default_score(client, monkeypatch):
    import sepsis_vitals.api as api

    client.get("/model/status")  # load the model

    def broken(**_):
        raise ValueError("synthetic model failure")

    monkeypatch.setattr(api._predictor, "predict", broken)
    before = _count_records()
    resp = client.post("/predict", json={"patient_id": "synthetic-model-down", "vitals": VITALS})
    assert resp.status_code == 503
    assert resp.json()["detail"].startswith("No risk calculated")
    assert "synthetic model failure" not in resp.text
    assert _count_records() == before
    assert any(i["outcome"] == "model_error" for i in api._recent_inferences)


def test_record_prediction_raises_when_the_database_refuses(monkeypatch):
    from sqlalchemy.exc import OperationalError

    from sepsis_vitals import db
    from sepsis_vitals.audit_records import AuditWriteError, record_prediction

    class Broken:
        def add(self, _):
            pass

        def commit(self):
            raise OperationalError("INSERT", {}, Exception("disk I/O error"))

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(db, "SessionLocal", Broken)
    with pytest.raises(AuditWriteError, match="OperationalError"):
        record_prediction(patient_id="synthetic", user_id=None, source="api", request_id="r1",
                          result={"risk_probability": 0.1, "risk_level": "low"}, payload={})


# -- append-only and tamper-evident ---------------------------------------------

def test_records_cannot_be_changed_or_removed(client):
    from sqlalchemy import text

    from sepsis_vitals.db import ImmutableRecordError, PredictionRecord, SessionLocal

    record_id = client.post("/predict", json={"patient_id": "synthetic-imm", "vitals": VITALS}).json()["audit_record_id"]
    db = SessionLocal()
    try:
        record = db.get(PredictionRecord, record_id)
        record.risk_level = "low"
        with pytest.raises(ImmutableRecordError):
            db.flush()
        db.rollback()
        with pytest.raises(ImmutableRecordError):
            db.query(PredictionRecord).filter(PredictionRecord.id == record_id).delete()
        db.rollback()
        # Below the ORM, the database itself refuses (trigger).
        for statement in ("UPDATE prediction_records SET risk_level = 'low'",
                          "DELETE FROM prediction_records"):
            with pytest.raises(Exception, match="append-only"):
                db.execute(text(statement))
            db.rollback()
        if db.bind.dialect.name == "postgresql":
            with pytest.raises(Exception, match="append-only"):
                db.execute(text("TRUNCATE prediction_records"))
            db.rollback()
    finally:
        db.close()
    from sepsis_vitals.audit_records import verify_record
    assert verify_record(_record(record_id)) is True  # the row is intact


def test_tampering_is_detected():
    from types import SimpleNamespace

    from sepsis_vitals.audit_records import SEALED_FIELDS, _seal, verify_record

    values = {name: None for name in SEALED_FIELDS}
    values.update(id="a", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), patient_id="synthetic",
                  risk_probability=0.2, risk_level="low", alert_fired=False,
                  input_vitals='{"observation":{}}', output_scores="{}", top_risk_factors="[]")
    record = SimpleNamespace(**values, record_sha256=_seal(values))
    assert verify_record(record) is True
    record.risk_level = "high"
    assert verify_record(record) is False
    assert verify_record(SimpleNamespace(record_sha256=None)) is None  # pre-006 row


# -- unknown fields are refused, never silently dropped --------------------------

@pytest.mark.parametrize("body", [
    {"patient_id": "s", "vitals": VITALS, "comorbidities": {"hypertension": 1}},
    {"patient_id": "s", "vitals": {**VITALS, "heartrate": 90}},
    {"patient_id": "s", "vitals": VITALS, "sex": "F"},
])
def test_unknown_fields_are_rejected(client, body):
    assert client.post("/predict", json=body).status_code == 422


# -- time-series contract --------------------------------------------------------

def _series(n=3, start=None, step=timedelta(hours=1)):
    start = start or datetime.now(timezone.utc) - step * n
    return [
        {"timestamp": (start + step * i).isoformat(), "heart_rate": 90 + 8 * i,
         "resp_rate": 18 + 2 * i, "sbp": 120 - 8 * i, "temperature": 37.2 + 0.4 * i}
        for i in range(n)
    ]


def test_timeseries_scores_the_last_observation_with_its_history(client):
    obs = _series(4)
    resp = client.post("/predict/timeseries", json={
        "patient_id": "synthetic-ts", "age_years": 64, "observations": obs,
        "units": {"temperature": "Cel", "heart_rate": "/min", "sbp": "mm[Hg]"},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data_quality"]["observations_used"] == 4
    assert body["data_quality"]["history_span_minutes"] == pytest.approx(180, abs=0.5)
    assert datetime.fromisoformat(body["observed_at"].replace("Z", "+00:00")) == \
        datetime.fromisoformat(obs[-1]["timestamp"])
    record = _record(body["audit_record_id"])
    assert record.source == "api-timeseries"
    assert len(json.loads(record.input_vitals)["history"]) == 3


@pytest.mark.parametrize("mutate, reason", [
    (lambda r: r["observations"].reverse(), "ascending time order"),
    (lambda r: r["observations"].__setitem__(1, {**r["observations"][1], "timestamp": r["observations"][0]["timestamp"]}),
     "duplicates the timestamp"),
    (lambda r: r["observations"][0].__setitem__("timestamp", "2026-01-01T00:00:00"), "UTC offset"),
    (lambda r: r["observations"][-1].__setitem__(
        "timestamp", (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()), "in the future"),
    (lambda r: r["observations"][1].__setitem__("patient_id", "someone-else"), "does not match"),
    (lambda r: r.__setitem__("units", {"temperature": "[degF]"}), "send values in Cel"),
    (lambda r: r.__setitem__("units", {"glucose": "mg/dL"}), "unknown field"),
    (lambda r: r["observations"].__setitem__(-1, {"timestamp": r["observations"][-1]["timestamp"], "heart_rate": 90}),
     "at least 3 measurements"),
    (lambda r: r["observations"][-1].__setitem__("temperature", 98.6), "less than or equal to 45"),
    (lambda r: r["observations"][-1].__setitem__("glucose", 5.1), "Extra inputs are not permitted"),
    (lambda r: r.__setitem__("observations", _series(49)), "at most 48"),
])
def test_timeseries_rejects_inconsistent_payloads(client, mutate, reason):
    request = {"patient_id": "synthetic-ts-bad", "observations": _series(3)}
    mutate(request)
    before = _count_records()
    resp = client.post("/predict/timeseries", json=request)
    assert resp.status_code == 422, resp.text
    assert reason in resp.text
    assert _count_records() == before


# -- batch ------------------------------------------------------------------------

def test_batch_withholds_only_the_unrecorded_patient(client, monkeypatch):
    import sepsis_vitals.audit_records as audit

    real = audit.record_prediction

    def flaky(**kwargs):
        if kwargs["patient_id"] == "synthetic-batch-2":
            raise audit.AuditWriteError("OperationalError")
        return real(**kwargs)

    monkeypatch.setattr(audit, "record_prediction", flaky)
    body = client.post("/predict/batch", json={"patients": [
        {"patient_id": "synthetic-batch-1", "vitals": VITALS},
        {"patient_id": "synthetic-batch-2", "vitals": VITALS},
    ]}).json()
    assert [p["patient_id"] for p in body["predictions"]] == ["synthetic-batch-1"]
    assert body["errors"] == [{"index": 1, "error": audit_error_text()}]
    assert all(p["audit_record_id"] for p in body["predictions"])


def audit_error_text():
    from sepsis_vitals.api import NO_RISK_AUDIT_UNAVAILABLE
    return NO_RISK_AUDIT_UNAVAILABLE


# -- continuous monitor -----------------------------------------------------------

@pytest.fixture()
def ingester(model_dir, tmp_path):
    from sepsis_vitals.ml.monitor import DeteriorationTracker, PatientRegistry, VitalsIngester
    from sepsis_vitals.ml.predictor import SepsisPredictor

    predictor = SepsisPredictor(model_dir=str(model_dir), state_dir=str(tmp_path / "state"))
    predictor.load()
    ws = AsyncMock()
    ws.broadcast = AsyncMock()
    written = []

    def writer(**kwargs):
        written.append(kwargs)
        return f"record-{len(written)}"

    ing = VitalsIngester(predictor, PatientRegistry(debounce_seconds=0), DeteriorationTracker(), ws,
                         audit_writer=writer)
    ing.written = written
    return ing


def test_monitor_records_the_alert_decision_before_broadcasting(ingester):
    result = asyncio.run(ingester.ingest_single("synthetic-mon", VITALS, demographics={"age": 70}))
    assert result["audit_record_id"] == "record-1"
    recorded = ingester.written[0]["result"]
    assert recorded["monitor_decision"]["message_type"] in {"patient_update", "sepsis_alert"}
    assert recorded["monitor_decision"]["policy"]["status"].startswith("engineering-placeholder")
    message = ingester.ws_manager.broadcast.await_args.args[0]
    assert message["audit_record_id"] == "record-1"
    assert ingester.written[0]["payload"]["age_years"] == 70
    info = ingester.registry.get_patient_info("synthetic-mon")
    assert info["risk_status"] == "scored" and info["risk_probability"] is not None


def test_monitor_uses_registered_demographics_for_later_readings(ingester):
    ingester.registry.register("synthetic-reg", demographics={"age": 81}, comorbidities={"has_ckd": 1})
    asyncio.run(ingester.ingest_single("synthetic-reg", VITALS))
    payload = ingester.written[0]["payload"]
    assert payload["age_years"] == 81 and payload["comorbidities"] == {"has_ckd": 1}


@pytest.mark.parametrize("failure", ["audit", "model"])
def test_monitor_failure_shows_no_risk_and_clears_the_old_score(ingester, monkeypatch, failure):
    from sepsis_vitals.audit_records import AuditWriteError

    asyncio.run(ingester.ingest_single("synthetic-fail", VITALS))
    assert ingester.registry.get_patient_info("synthetic-fail")["risk_status"] == "scored"
    window = list(ingester.tracker.get_window("synthetic-fail"))

    if failure == "audit":
        def refuse(**_):
            raise AuditWriteError("OperationalError")
        ingester.audit_writer = refuse
    else:
        def broken(**_):
            raise ValueError("synthetic model failure")
        monkeypatch.setattr(ingester.predictor, "predict", broken)

    assert asyncio.run(ingester.ingest_single("synthetic-fail", VITALS)) is None
    info = ingester.registry.get_patient_info("synthetic-fail")
    assert info["risk_status"] == "unavailable"
    assert info["risk_probability"] is None and info["risk_level"] == "unknown"
    message = ingester.ws_manager.broadcast.await_args.args[0]
    assert message["type"] == "risk_unavailable" and "not a low-risk result" in message["message"]
    assert "risk_probability" not in message
    assert ingester.tracker.get_window("synthetic-fail") == window  # restored


def test_api_monitor_always_has_an_audit_writer(client):
    import sepsis_vitals.api as api

    client.get("/model/status")
    _, _, ingester = api._get_monitor_components()
    if ingester is not None:
        assert ingester.audit_writer is api._monitor_audit_writer


# -- request log ------------------------------------------------------------------

def test_request_log_has_route_and_outcome_but_no_identifiers(client, caplog):
    caplog.set_level(logging.INFO, logger="sepsis_vitals.requests")
    resp = client.post("/predict", json={"patient_id": "MRN-SYNTH-0042", "vitals": VITALS},
                       headers={"X-Request-ID": "demo-request-0001"})
    assert resp.headers["X-Request-ID"] == "demo-request-0001"
    lines = [r.getMessage() for r in caplog.records if r.name == "sepsis_vitals.requests"]
    entry = json.loads(lines[-1].split(" ", 1)[1])
    assert entry["route"] == "/predict" and entry["status"] == 200
    assert entry["request_id"] == "demo-request-0001"
    assert entry["inference"][0]["outcome"] == "scored" and entry["inference"][0]["model_version"]
    assert "MRN-SYNTH-0042" not in caplog.text


def test_malformed_request_ids_are_replaced(client):
    resp = client.post("/predict", json={"patient_id": "s", "vitals": VITALS},
                       headers={"X-Request-ID": "bad id; with spaces"})
    assert resp.headers["X-Request-ID"] != "bad id; with spaces"
    assert len(resp.headers["X-Request-ID"]) == 32
