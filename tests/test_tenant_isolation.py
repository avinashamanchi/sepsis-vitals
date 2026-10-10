"""
tests/test_tenant_isolation.py — Two-site integration tests for tenant scoping.

A nurse at site A must not be able to read, write, re-home, or discover a
patient at site B through the patients, FHIR, dashboard, monitor, or profile
endpoints, and a user without a site assignment must see nothing.
"""

from __future__ import annotations

import importlib.util
import uuid

import pytest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture()
def two_sites(monkeypatch):
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api
    from sepsis_vitals.auth.service import register_user
    from sepsis_vitals.auth.tokens import create_access_token
    from sepsis_vitals.db import SessionLocal, init_db
    from sepsis_vitals.patients import service

    monkeypatch.setenv("SEPSIS_JWT_SECRET", "tenant-isolation-test-secret-0123456789")
    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", True)
    for dep in (api.check_rate_limit, api.check_auth_rate_limit, api.check_ml_rate_limit):
        api.app.dependency_overrides[dep] = lambda: None

    init_db()
    run = uuid.uuid4().hex[:8]
    site_a, site_b = f"A{run}", f"B{run}"
    db = SessionLocal()
    try:
        def make_user(label: str, site: str | None, role: str = "nurse") -> str:
            email = f"{label}-{run}@example.org"
            user = register_user(email, "Correct-Horse-Battery-9!", role, site, db)["user"]
            user_id = user["id"] if isinstance(user, dict) else user.id
            token = create_access_token(user_id, email, role, site)
            return token

        tokens = {
            "a": make_user("nurse-a", site_a),
            "b": make_user("nurse-b", site_b),
            "none": make_user("nurse-unassigned", None),
        }
        patient_a = service.create_patient(f"MRN-A-{run}", site_a, 61, "F", db).id
        patient_b = service.create_patient(f"MRN-B-{run}", site_b, 34, "M", db).id
    finally:
        db.close()

    with TestClient(api.app) as client:
        yield {
            "client": client,
            "site_a": site_a,
            "site_b": site_b,
            "patient_a": patient_a,
            "patient_b": patient_b,
            "mrn_b": f"MRN-B-{run}",
            "headers": {k: {"Authorization": f"Bearer {v}"} for k, v in tokens.items()},
        }
    api.app.dependency_overrides.clear()


def test_list_only_returns_own_site(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.get("/patients", headers=h)
    assert resp.status_code == 200
    sites = {p["site_id"] for p in resp.json()}
    assert sites == {two_sites["site_a"]}


def test_cannot_filter_to_another_site(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    assert c.get(f"/patients?site_id={two_sites['site_b']}", headers=h).status_code == 404
    assert c.get(
        f"/patients/dashboard/stats?site_id={two_sites['site_b']}", headers=h
    ).status_code == 404


def test_dashboard_defaults_to_own_site(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.get("/patients/dashboard/stats", headers=h)
    assert resp.status_code == 200
    assert resp.json()["patient_count"] == 1


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "/patients/{pid}", None),
        ("get", "/patients/{pid}/vitals", None),
        ("get", "/patients/{pid}/history", None),
        ("post", "/patients/{pid}/vitals", {"heart_rate": 130, "temperature": 39.5}),
        ("put", "/patients/{pid}", {"site_id": "{site_a}"}),
        ("get", "/fhir/Patient/{pid}", None),
        ("get", "/fhir/Patient/{pid}/observations", None),
        ("get", "/fhir/RiskAssessment/{pid}", None),
        ("get", "/patient/{pid}/trend", None),
        ("delete", "/monitor/{pid}", None),
    ],
)
def test_cross_site_patient_is_not_found(two_sites, method, path, body):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    url = path.format(pid=two_sites["patient_b"])
    if body is not None:
        body = {k: (v.format(site_a=two_sites["site_a"]) if isinstance(v, str) else v)
                for k, v in body.items()}
    resp = getattr(c, method)(url, headers=h, **({"json": body} if body is not None else {}))
    assert resp.status_code == 404, (url, resp.status_code, resp.text)


def test_fhir_lookup_by_other_sites_mrn_is_not_found(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    assert c.get(f"/fhir/Patient/{two_sites['mrn_b']}", headers=h).status_code == 404


def test_own_patient_is_accessible(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    pid = two_sites["patient_a"]
    assert c.get(f"/patients/{pid}", headers=h).status_code == 200
    assert c.post(f"/patients/{pid}/vitals", headers=h,
                  json={"heart_rate": 88, "temperature": 37.0}).status_code == 201


def test_cannot_create_patient_at_another_site(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.post("/patients", headers=h, json={
        "external_id": f"MRN-X-{uuid.uuid4().hex[:6]}", "site_id": two_sites["site_b"],
    })
    assert resp.status_code == 403


def test_nurse_cannot_move_own_patient_between_sites(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.put(f"/patients/{two_sites['patient_a']}", headers=h,
                 json={"site_id": two_sites["site_b"]})
    assert resp.status_code == 403


def test_user_cannot_switch_own_site(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.put("/auth/me", headers=h, json={"site_id": two_sites["site_b"]})
    assert resp.status_code == 403
    # and the patient at site B is still hidden afterwards
    assert c.get(f"/patients/{two_sites['patient_b']}", headers=h).status_code == 404


def test_unassigned_user_sees_nothing(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["none"]
    assert c.get("/patients", headers=h).status_code == 403
    assert c.get(f"/patients/{two_sites['patient_a']}", headers=h).status_code == 403


def test_only_admin_can_assign_sites(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.put("/auth/users/some-user/site", headers=h, json={"site_id": two_sites["site_b"]})
    assert resp.status_code == 403


def test_cannot_write_predictions_into_another_sites_patient(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    body = {
        "patient_id": two_sites["patient_b"],
        "vitals": {"heart_rate": 120, "resp_rate": 26, "sbp": 95, "temperature": 38.9},
    }
    resp = c.post("/predict", headers=h, json=body)
    # 404 when the model is loaded; 503 is acceptable only if no model is present
    assert resp.status_code in (404, 503)
    if resp.status_code == 503:
        pytest.skip("model artifact not loaded")


def test_same_mrn_may_exist_at_two_sites(two_sites):
    """MRNs are only unique per site; site A may register B's MRN for its own patient."""
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.post("/patients", headers=h, json={
        "external_id": two_sites["mrn_b"], "site_id": two_sites["site_a"],
    })
    assert resp.status_code == 201, resp.text
    # and a duplicate within the same site is still rejected, without echoing the MRN
    dup = c.post("/patients", headers=h, json={
        "external_id": two_sites["mrn_b"], "site_id": two_sites["site_a"],
    })
    assert dup.status_code == 409
    assert two_sites["mrn_b"] not in dup.text


def test_patient_list_reports_latest_observation_or_null(two_sites):
    """Unobserved patients must come back with null risk, never a default 'low'."""
    c, h = two_sites["client"], two_sites["headers"]["a"]
    before = {p["id"]: p for p in c.get("/patients", headers=h).json()}
    assert before[two_sites["patient_a"]]["latest_risk_level"] is None
    assert before[two_sites["patient_a"]]["latest_vitals"] is None

    c.post(f"/patients/{two_sites['patient_a']}/vitals", headers=h,
           json={"heart_rate": 125, "resp_rate": 26, "sbp": 88, "temperature": 39.2})
    after = {p["id"]: p for p in c.get("/patients", headers=h).json()}[two_sites["patient_a"]]
    assert after["latest_vitals"]["heart_rate"] == 125
    assert after["latest_risk_level"] in {"high", "critical"}
    assert after["latest_recorded_at"] is not None


def _ws_token(role: str, org_id, lifetime_s: int) -> str:
    import time

    import jwt as pyjwt

    from sepsis_vitals.auth import tokens

    now = int(time.time())
    payload = {"sub": f"ws-{uuid.uuid4().hex[:6]}", "email": "ws@example.org", "role": role,
               "org_id": org_id, "type": "access", "jti": uuid.uuid4().hex,
               "iat": now, "exp": now + lifetime_s}
    return pyjwt.encode(payload, tokens._get_signing_key(), algorithm=tokens._ALGORITHM)


def test_websocket_rejects_orgless_non_admin(two_sites):
    from starlette.websockets import WebSocketDisconnect

    c = two_sites["client"]
    token = _ws_token("nurse", None, 600)
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/alerts", subprotocols=["sepsis-vitals", f"bearer.{token}"]) as ws:
            ws.receive_text()


def test_websocket_closes_when_token_expires(two_sites):
    from starlette.websockets import WebSocketDisconnect

    c = two_sites["client"]
    token = _ws_token("nurse", two_sites["site_a"], 2)
    with c.websocket_connect("/ws/alerts", subprotocols=["sepsis-vitals", f"bearer.{token}"]) as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()  # nothing is sent; the server closes at expiry
    assert closed.value.code == 1008


# -- alert workflow scoping (S4, S5) ---------------------------------------------

def test_alert_ack_is_attributed_to_authenticated_user_and_scoped(two_sites):
    c, h = two_sites["client"], two_sites["headers"]
    pid = two_sites["patient_a"]
    c.post(f"/patients/{pid}/vitals", headers=h["a"],
           json={"heart_rate": 135, "resp_rate": 30, "sbp": 82, "temperature": 39.6})
    alerts = [a for a in c.get("/patients/alerts", headers=h["a"]).json() if a["patient_id"] == pid]
    assert alerts, "high-risk vitals should open an alert"
    alert_id = alerts[0]["id"]

    # nurse B cannot see or acknowledge site A's alert
    assert all(a["patient_id"] != pid for a in c.get("/patients/alerts", headers=h["b"]).json())
    other = c.put(f"/patients/alerts/{alert_id}/acknowledge", headers=h["b"],
                  json={"reason": "not my patient"})
    assert other.status_code == 404

    # a caller-supplied user_id is ignored; the authenticated user is recorded
    me = c.get("/auth/me", headers=h["a"]).json()
    acked = c.put(f"/patients/alerts/{alert_id}/acknowledge", headers=h["a"],
                  json={"user_id": "someone-else", "reason": "reviewed at bedside"})
    assert acked.status_code == 200
    assert acked.json()["action_by"] == me["id"]


def test_notification_contacts_are_owner_scoped(two_sites):
    c, h = two_sites["client"], two_sites["headers"]
    created = c.post("/alerts/contacts", headers=h["a"],
                     json={"channel": "websocket", "destination": "ward-a-console"})
    assert created.status_code == 201, created.text
    contact_id = created.json()["id"]

    listed_by_b = c.get("/alerts/contacts", headers=h["b"]).json()["contacts"]
    assert all(x["id"] != contact_id for x in listed_by_b)
    assert c.delete(f"/alerts/contacts/{contact_id}", headers=h["b"]).status_code == 404
    spoofed = c.post("/alerts/contacts", headers=h["b"],
                     json={"user_id": "someone-else", "channel": "websocket", "destination": "x"})
    assert spoofed.status_code == 403


def test_test_alerts_and_delivery_history_are_admin_only(two_sites):
    c, h = two_sites["client"], two_sites["headers"]["a"]
    assert c.post("/alerts/test", headers=h,
                  json={"channel": "sms", "destination": "+15550000000"}).status_code == 403
    assert c.get("/alerts/history", headers=h).status_code == 403


@pytest.mark.parametrize("endpoint,ok", [
    ("https://fcm.googleapis.com/fcm/send/abc", True),
    ("https://updates.push.services.mozilla.com/wpush/v2/abc", True),
    ("https://web.push.apple.com/abc", True),
    ("http://fcm.googleapis.com/fcm/send/abc", False),           # not https
    ("https://169.254.169.254/latest/meta-data", False),        # cloud metadata (SSRF)
    ("https://fcm.googleapis.com.attacker.example/x", False),   # suffix trick
    ("https://internal-service:8080/hook", False),
])
def test_push_endpoint_allowlist(endpoint, ok):
    from fastapi import HTTPException

    from sepsis_vitals.alerts.router import _validate_push_endpoint

    if ok:
        _validate_push_endpoint(endpoint)
    else:
        with pytest.raises(HTTPException):
            _validate_push_endpoint(endpoint)


def test_active_alert_list_is_routable(two_sites):
    """Regression: /patients/{patient_id} shadowed GET /patients/alerts."""
    c, h = two_sites["client"], two_sites["headers"]["a"]
    resp = c.get("/patients/alerts", headers=h)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
