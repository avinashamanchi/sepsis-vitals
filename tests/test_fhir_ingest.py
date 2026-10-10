"""
tests/test_fhir_ingest.py — the FHIR write endpoints (Patient, Observation,
Bundle, $process-vitals): execution model, transactions, duplicates, input
validation, tenant isolation and audit.

The four handlers had no tests before stage 4. All data here is synthetic.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi missing")

LOINC = "http://loinc.org"
FHIR_JSON = "application/fhir+json"


def patient_res(mrn: str, rid: str = "p1", birth: str = "1960-01-01", gender: str = "female") -> dict:
    return {
        "resourceType": "Patient", "id": rid, "gender": gender, "birthDate": birth,
        "identifier": [{"type": {"coding": [{"code": "MR"}]}, "value": mrn}],
    }


def obs_res(subject: str, value: float = 88, code: str = "8867-4", unit: str = "/min",
            when: str | None = "2026-10-01T10:00:00Z", rid: str = "o1") -> dict:
    res = {
        "resourceType": "Observation", "id": rid, "status": "final",
        "code": {"coding": [{"system": LOINC, "code": code}]},
        "subject": {"reference": f"Patient/{subject}"},
        "valueQuantity": {"value": value, "unit": unit},
    }
    if when is not None:
        res["effectiveDateTime"] = when
    return res


def bundle_res(*resources: dict) -> dict:
    return {"resourceType": "Bundle", "type": "transaction", "entry": [{"resource": r} for r in resources]}


@pytest.fixture()
def env(monkeypatch):
    """Two sites with a nurse each, and a nurse with no site."""
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api
    from sepsis_vitals.auth.service import register_user
    from sepsis_vitals.auth.tokens import create_access_token
    from sepsis_vitals.db import SessionLocal, init_db
    from sepsis_vitals.patients import service

    monkeypatch.setenv("SEPSIS_JWT_SECRET", "fhir-ingest-test-secret-0123456789abcdef")
    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", True)
    for dep in (api.check_rate_limit, api.check_auth_rate_limit, api.check_ml_rate_limit):
        api.app.dependency_overrides[dep] = lambda: None
    init_db()
    run = uuid.uuid4().hex[:8]
    site_a, site_b = f"FA{run}", f"FB{run}"
    db = SessionLocal()
    try:
        users = {}
        for label, site, role in (("a", site_a, "nurse"), ("b", site_b, "nurse"), ("none", None, "nurse")):
            email = f"fhir-{label}-{run}@example.org"
            user = register_user(email, "Correct-Horse-Battery-9!", role, site, db)["user"]
            user_id = user["id"] if isinstance(user, dict) else user.id
            users[label] = {
                "token": create_access_token(user_id, email, role, site),
                "principal": {"id": user_id, "email": email, "role": role, "org_id": site},
            }
        patient_b = service.create_patient(f"MRN-B-{run}", site_b, 34, "M", db).id
    finally:
        db.close()

    with TestClient(api.app) as client:
        yield {
            "client": client, "run": run, "site_a": site_a, "site_b": site_b,
            "patient_b": patient_b, "mrn_b": f"MRN-B-{run}",
            "h": {k: {"Authorization": f"Bearer {v['token']}", "Content-Type": FHIR_JSON} for k, v in users.items()},
            "user": {k: v["principal"] for k, v in users.items()},
        }
    api.app.dependency_overrides.clear()


def _post(env, path, body, who="a"):
    data = body if isinstance(body, (bytes, str)) else json.dumps(body)
    return env["client"].post(path, content=data, headers=env["h"][who])


def _count(model, **filters) -> int:
    from sepsis_vitals.db import SessionLocal

    db = SessionLocal()
    try:
        query = db.query(model)
        for name, value in filters.items():
            query = query.filter(getattr(model, name) == value)
        return query.count()
    finally:
        db.close()


def _patients_with_mrn(site: str, mrn: str) -> int:
    from sepsis_vitals.db import Patient
    from sepsis_vitals.security import compute_blind_index

    return _count(Patient, site_id=site, external_id_hash=compute_blind_index(mrn))


def _create_patient(env, mrn: str, who="a") -> str:
    resp = _post(env, "/fhir/Patient", patient_res(mrn), who)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


# -- idempotency and duplicates -------------------------------------------------------

def test_patient_post_creates_then_updates(env):
    mrn = f"MRN-A-{env['run']}"
    first = _post(env, "/fhir/Patient", patient_res(mrn))
    second = _post(env, "/fhir/Patient", patient_res(mrn, birth="1961-01-01"))
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["id"] == second.json()["id"]
    assert first.headers["content-type"].startswith(FHIR_JSON)
    assert _patients_with_mrn(env["site_a"], mrn) == 1


def test_concurrent_creates_of_one_mrn_make_one_patient(env, monkeypatch):
    """Both requests look the MRN up before either inserts (forced with a
    barrier); the loser hits the unique constraint, retries and updates."""
    from sepsis_vitals.fhir import ingest
    from sepsis_vitals.fhir.resources import FHIRPatient

    mrn = f"MRN-RACE-{env['run']}"
    barrier = threading.Barrier(2)
    waited = threading.local()
    real_lookup = ingest._lookup

    def lookup_then_wait(db, internal):
        found = real_lookup(db, internal)
        if not getattr(waited, "done", False):
            waited.done = True
            barrier.wait(timeout=10)
        return found

    monkeypatch.setattr(ingest, "_lookup", lookup_then_wait)
    results, errors = [], []

    def create():
        try:
            work = ingest.patient_work(FHIRPatient.from_fhir(patient_res(mrn)), env["user"]["a"])
            results.append(ingest.run_in_worker(work))
        except Exception as exc:  # pragma: no cover - reported by the assertion
            errors.append(repr(exc))

    threads = [threading.Thread(target=create) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert sorted(created for _, created in results) == [False, True]
    assert len({res["id"] for res, _ in results}) == 1
    assert _patients_with_mrn(env["site_a"], mrn) == 1


def test_replayed_observation_is_acknowledged_not_duplicated(env):
    from sepsis_vitals.db import VitalReading

    pid = _create_patient(env, f"MRN-OBS-{env['run']}")
    first = _post(env, "/fhir/Observation", obs_res(pid))
    replay = _post(env, "/fhir/Observation", obs_res(pid))
    assert (first.status_code, replay.status_code) == (201, 200)
    assert _count(VitalReading, patient_id=pid) == 1


def test_distinct_readings_are_all_kept(env):
    from sepsis_vitals.db import VitalReading

    pid = _create_patient(env, f"MRN-DISTINCT-{env['run']}")
    codes = [
        obs_res(pid, 88),                                   # HR 88 at 10:00
        obs_res(pid, 90),                                   # HR 90 at 10:00
        obs_res(pid, 20, code="9279-1"),                    # RR 20 at 10:00
        obs_res(pid, 88, when="2026-10-01T10:05:00Z"),      # HR 88 at 10:05
    ]
    assert [_post(env, "/fhir/Observation", o).status_code for o in codes] == [201] * 4
    assert _count(VitalReading, patient_id=pid) == 4


def test_observations_without_a_time_cannot_be_recognised_as_replays(env):
    """Documented limitation: the receipt time differs on each delivery."""
    from sepsis_vitals.db import VitalReading

    pid = _create_patient(env, f"MRN-NOTIME-{env['run']}")
    for _ in range(2):
        assert _post(env, "/fhir/Observation", obs_res(pid, when=None)).status_code == 201
    assert _count(VitalReading, patient_id=pid) == 2


def test_concurrent_identical_observations_store_one_reading(env):
    from sepsis_vitals.db import VitalReading
    from sepsis_vitals.fhir import ingest
    from sepsis_vitals.fhir.resources import FHIRObservation

    pid = _create_patient(env, f"MRN-CONC-{env['run']}")
    barrier = threading.Barrier(6)
    created, errors = [], []

    def send():
        try:
            work = ingest.observation_work(FHIRObservation.from_fhir(obs_res(pid)), env["user"]["a"])
            barrier.wait(timeout=10)
            created.append(ingest.run_in_worker(work).created)
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))

    threads = [threading.Thread(target=send) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert sorted(created) == [False] * 5 + [True]
    assert _count(VitalReading, patient_id=pid) == 1


def test_replayed_bundle_and_process_vitals_store_once(env):
    from sepsis_vitals.db import Score, VitalReading

    mrn = f"MRN-BUNDLE-{env['run']}"
    bundle = bundle_res(patient_res(mrn, rid="px"), obs_res("px", 112), obs_res("px", 26, code="9279-1"))
    first = _post(env, "/fhir/Bundle", bundle)
    replay = _post(env, "/fhir/Bundle", bundle)
    assert first.status_code == replay.status_code == 200
    statuses = [[e["response"]["status"] for e in r.json()["entry"]] for r in (first, replay)]
    assert statuses == [["201 Created", "201 Created", "201 Created"], ["200 OK", "200 OK", "200 OK"]]
    pid = first.json()["entry"][0]["response"]["location"].split("/")[1]
    assert _count(VitalReading, patient_id=pid) == 2

    vitals = bundle_res(patient_res(mrn, rid="px"), obs_res("px", 120, when="2026-10-01T11:00:00Z"),
                        obs_res("px", 28, code="9279-1", when="2026-10-01T11:00:00Z"))
    for _ in range(2):
        resp = _post(env, "/fhir/$process-vitals", vitals)
        assert resp.status_code == 200 and resp.json()["resourceType"] == "RiskAssessment"
    assert _count(VitalReading, patient_id=pid) == 3
    assert _count(Score) >= 1 and _scores_for(pid) == 1


def _scores_for(patient_id: str) -> int:
    from sepsis_vitals.db import Score, SessionLocal, VitalReading

    db = SessionLocal()
    try:
        return db.query(Score).join(VitalReading, Score.vital_id == VitalReading.id).filter(
            VitalReading.patient_id == patient_id).count()
    finally:
        db.close()


# -- invalid input --------------------------------------------------------------------

@pytest.mark.parametrize("path,body,status,code", [
    ("/fhir/Patient", b"{not json", 400, "invalid"),
    ("/fhir/Patient", b"[]", 400, "structure"),                      # was a 500 (AttributeError)
    ("/fhir/Bundle", b'"text"', 400, "structure"),
    ("/fhir/Patient", {"resourceType": "Observation"}, 400, "structure"),
    ("/fhir/Bundle", {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Patient", "identifier": "x"}}]}, 400, "structure"),
])
def test_malformed_bodies_are_rejected_without_writes(env, path, body, status, code):
    resp = _post(env, path, body)
    assert resp.status_code == status, resp.text
    assert resp.json()["resourceType"] == "OperationOutcome"
    assert resp.json()["issue"][0]["code"] == code


def test_invalid_observations_are_rejected_without_writes(env):
    from sepsis_vitals.db import VitalReading

    pid = _create_patient(env, f"MRN-INVALID-{env['run']}")
    cases = [
        (obs_res(pid, code="0000-0"), 422),                       # not a vital we track
        (obs_res(pid, 1000), 422),                                # outside the API's input bounds
        (obs_res(pid, -5), 422),
        (obs_res(pid, when="yesterday"), 400),                    # was silently stored as "now"
        (obs_res(str(uuid.uuid4())), 404),                        # unknown patient
        ({**obs_res(pid), "valueQuantity": {"unit": "/min"}}, 400),  # no value
    ]
    for body, status in cases:
        resp = _post(env, "/fhir/Observation", body)
        assert resp.status_code == status, (body, resp.text)
        assert resp.json()["resourceType"] == "OperationOutcome"
    nan = json.dumps(obs_res(pid)).replace('"value": 88', '"value": NaN')
    assert _post(env, "/fhir/Observation", nan).status_code == 422
    assert _count(VitalReading, patient_id=pid) == 0


# -- transactions -------------------------------------------------------------------

def test_failure_mid_bundle_rolls_everything_back(env, monkeypatch):
    import sqlalchemy.exc as sa_exc

    from sepsis_vitals.db import VitalReading, engine
    from sepsis_vitals.fhir import ingest

    calls = {"n": 0}
    real_record = ingest._record

    def fail_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sa_exc.OperationalError("INSERT INTO vitals", {}, Exception("disk I/O error"))
        return real_record(*args, **kwargs)

    monkeypatch.setattr(ingest, "_record", fail_second)
    mrn = f"MRN-ROLLBACK-{env['run']}"
    resp = _post(env, "/fhir/Bundle", bundle_res(patient_res(mrn, rid="pr"), obs_res("pr", 100), obs_res("pr", 22, code="9279-1")))
    assert resp.status_code == 503
    assert resp.json()["issue"][0]["code"] == "transient"
    assert "disk" not in resp.text  # driver details stay in the server log
    assert _patients_with_mrn(env["site_a"], mrn) == 0, "patient insert was not rolled back"
    assert _count(VitalReading, recorded_at=None) == 0
    assert engine.pool.checkedout() == 0, "a session was left open"


def test_cancelled_request_finishes_its_transaction_and_closes_the_session(env):
    """A client disconnect cancels the awaiting coroutine; the worker thread
    still commits or rolls back as a whole and closes its session."""
    from sepsis_vitals.db import VitalReading, engine
    from sepsis_vitals.fhir import ingest
    from sepsis_vitals.fhir.resources import FHIRObservation

    pid = _create_patient(env, f"MRN-CANCEL-{env['run']}")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    inner = ingest.observation_work(FHIRObservation.from_fhir(obs_res(pid)), env["user"]["a"])

    def slow_work(txn):
        started.set()
        release.wait(timeout=10)
        return inner(txn)

    def tracked(work):
        try:
            return ingest.run_in_worker(work)
        finally:
            finished.set()

    async def scenario():
        task = asyncio.create_task(asyncio.to_thread(tracked, slow_work))
        await asyncio.to_thread(started.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    release.set()
    assert finished.wait(timeout=10)
    assert _count(VitalReading, patient_id=pid) == 1
    assert engine.pool.checkedout() == 0


def test_ingestion_does_not_block_the_event_loop(env, monkeypatch):
    """Simulate a slow database write and check the loop keeps serving."""
    from starlette.requests import Request

    from sepsis_vitals.fhir import ingest
    from sepsis_vitals.fhir import router as fhir

    pid = _create_patient(env, f"MRN-LOOP-{env['run']}")
    real_record = ingest._record

    def slow_record(*args, **kwargs):
        time.sleep(0.4)  # stands in for a blocking database round trip
        return real_record(*args, **kwargs)

    monkeypatch.setattr(ingest, "_record", slow_record)
    body = json.dumps(obs_res(pid)).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def scenario():
        request = Request({"type": "http", "method": "POST", "path": "/fhir/Observation",
                           "headers": [(b"content-type", FHIR_JSON.encode())], "query_string": b""}, receive)
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        tick_task = asyncio.create_task(ticker())
        resp = await fhir.create_observation(request, current_user=env["user"]["a"])
        done.set()
        await tick_task
        return resp, ticks

    resp, ticks = asyncio.run(scenario())
    assert resp.status_code == 201
    assert ticks >= 15, f"event loop was blocked (only {ticks} ticks during a 0.4 s write)"


# -- tenant isolation ------------------------------------------------------------------

def test_observation_for_another_sites_patient_is_not_found(env):
    from sepsis_vitals.db import VitalReading

    for ref in (env["patient_b"], env["mrn_b"]):
        resp = _post(env, "/fhir/Observation", obs_res(ref))
        assert resp.status_code == 404, resp.text
    assert _count(VitalReading, patient_id=env["patient_b"]) == 0


def test_same_mrn_at_another_site_creates_a_separate_patient(env):
    from sepsis_vitals.db import Patient, SessionLocal

    resp = _post(env, "/fhir/Patient", patient_res(env["mrn_b"], birth="2000-01-01"))
    assert resp.status_code == 201
    assert resp.json()["id"] != env["patient_b"]
    db = SessionLocal()
    try:
        assert db.get(Patient, env["patient_b"]).age_years == 34  # site B's record untouched
    finally:
        db.close()


def test_bundle_cannot_write_to_another_sites_patient(env):
    from sepsis_vitals.db import VitalReading

    resp = _post(env, "/fhir/Bundle", bundle_res(obs_res(env["patient_b"], 120)))
    assert resp.status_code == 200
    assert resp.json()["entry"][0]["response"]["status"] == "404 Not Found"
    assert _count(VitalReading, patient_id=env["patient_b"]) == 0


def test_unassigned_user_cannot_ingest(env):
    resp = _post(env, "/fhir/Patient", patient_res(f"MRN-NONE-{env['run']}"), who="none")
    assert resp.status_code == 403
    assert _patients_with_mrn("fhir", f"MRN-NONE-{env['run']}") == 0


def test_unauthenticated_requests_are_rejected(env):
    resp = env["client"].post("/fhir/Observation", content=json.dumps(obs_res("x")),
                              headers={"Content-Type": FHIR_JSON})
    assert resp.status_code == 401


# -- audit --------------------------------------------------------------------------

def test_fhir_access_is_audited_without_logging_mrns(env, caplog):
    pid = _create_patient(env, f"MRN-AUDIT-{env['run']}")
    with caplog.at_level(logging.INFO, logger="sepsis_vitals.api"):
        _post(env, "/fhir/Observation", obs_res(pid))
        env["client"].get(f"/fhir/Patient/MRN-AUDIT-{env['run']}", headers=env["h"]["a"])
        env["client"].get(f"/fhir/Patient/{pid}", headers=env["h"]["a"])
    events = [json.loads(r.getMessage().split("HIPAA_AUDIT ", 1)[1])
              for r in caplog.records if r.getMessage().startswith("HIPAA_AUDIT ")]
    actions = [(e["action"], e["resource_id"]) for e in events]
    assert ("fhir_write", None) in actions
    assert ("fhir_read", pid) in actions
    assert all(e["user_id"] == env["user"]["a"]["id"] for e in events)
    assert f"MRN-AUDIT-{env['run']}" not in caplog.text
