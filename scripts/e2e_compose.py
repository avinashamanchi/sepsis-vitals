#!/usr/bin/env python3
"""End-to-end checks against the running compose stack (CI compose-smoke job).

    cd docker && docker compose up -d --wait db redis api dashboard
    python ../scripts/e2e_compose.py           # from docker/, needs Playwright + Chromium

Everything goes through nginx on http://localhost:8000 (the dashboard and
/api), the way a browser reaches the stack. All accounts, MRNs and passwords
are created here for this run; nothing is real. Output never includes
tokens or passwords.

Covers: SPA rendering at "/" (N39); login and the EULA gate; FHIR ingestion
through to the patient list and dashboard, and replay de-duplication; tenant
boundaries in the UI; no demo data in live mode; Predict showing rule and
model levels separately and "Clinical use: not permitted"; the
password-reset token lifecycle in the browser; refresh rotation, replay and
logout revocation; admin-only test alerts; no API path that changes
clinical use; and liveness versus readiness during a database outage.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

BASE = "http://localhost:8000"
API = BASE + "/api"
RUN = secrets.token_hex(3)
PASSWORD = f"E2e-{secrets.token_urlsafe(12)}-9!"
NEW_PASSWORD = f"E2e-new-{secrets.token_urlsafe(12)}-7!"
USERS = {"a": (f"e2e-nurse-a-{RUN}@example.org", f"E2E-A-{RUN}"),
         "b": (f"e2e-nurse-b-{RUN}@example.org", f"E2E-B-{RUN}")}
MRN = f"MRN-E2E-{RUN}"
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("ok:   " if condition else "FAIL: ") + message, flush=True)
    if not condition:
        FAILURES.append(message)


def call(method: str, path: str, body: Any = None, token: Optional[str] = None,
         content_type: str = "application/json") -> Tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310 - local stack
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw) if raw else None
        except ValueError:
            return err.code, raw.decode(errors="replace")


def api_python(code: str) -> str:
    """Run Python inside the API container (seeding, reset tokens)."""
    out = subprocess.run(["docker", "compose", "exec", "-T", "api", "python", "-c", code],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"api container command failed: {out.stderr[-400:]}")
    return out.stdout.strip()


def wait_for(predicate, what: str, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except OSError:
            pass
        time.sleep(1)
    check(False, f"timed out waiting for {what}")
    return False


def seed() -> None:
    lines = ["from sepsis_vitals.auth.service import register_user",
             "from sepsis_vitals.db import SessionLocal", "db = SessionLocal()"]
    for email, site in USERS.values():
        lines.append(f"register_user({email!r}, {PASSWORD!r}, 'nurse', {site!r}, db)")
    lines.append("db.close()")
    api_python("\n".join(lines))


def login(key: str, password: str = PASSWORD) -> Dict[str, str]:
    status, body = call("POST", "/auth/login", {"email": USERS[key][0], "password": password})
    check(status == 200, f"API login for nurse {key.upper()} (HTTP {status})")
    return body or {}


def bundle() -> Dict[str, Any]:
    def obs(rid: str, code: str, value: float) -> Dict[str, Any]:
        return {"resource": {
            "resourceType": "Observation", "id": rid, "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": code}]},
            "subject": {"reference": "Patient/pe2e"}, "effectiveDateTime": "2026-10-01T10:00:00Z",
            "valueQuantity": {"value": value}}}

    return {"resourceType": "Bundle", "type": "transaction", "entry": [
        {"resource": {"resourceType": "Patient", "id": "pe2e", "gender": "female", "birthDate": "1958-04-02",
                      "identifier": [{"type": {"coding": [{"code": "MR"}]}, "value": MRN}]}},
        obs("o1", "8867-4", 112), obs("o2", "9279-1", 26)]}


def api_checks(tokens: Dict[str, Dict[str, str]]) -> None:
    a = tokens["a"]["access_token"]
    check(call("GET", "/health")[0] == 200, "liveness /health")
    status, ready = call("GET", "/ready")
    check(status == 200 and ready.get("migrations") == "at-head", f"readiness /ready ({ready})")
    model = call("GET", "/model/status")[1]
    check(model.get("state") == "ready" and model.get("prediction_ready") is True,
          "prediction readiness /model/status")
    check(model.get("clinically_ready") is False and model.get("clinical_use") == "not-permitted",
          "model status never reports clinical readiness")

    first = call("POST", "/fhir/Bundle", bundle(), a, "application/fhir+json")
    replay = call("POST", "/fhir/Bundle", bundle(), a, "application/fhir+json")
    statuses = [[e["response"]["status"] for e in r[1]["entry"]] for r in (first, replay)]
    check(first[0] == 200 and statuses[0] == ["201 Created"] * 3, f"FHIR bundle ingested ({statuses[0]})")
    check(replay[0] == 200 and statuses[1] == ["200 OK"] * 3, f"replayed bundle not stored twice ({statuses[1]})")

    patients = call("GET", "/patients", token=a)[1]
    mine = [p for p in patients if p["external_id"] == MRN]
    vitals = (mine[0].get("latest_vitals") or {}) if len(mine) == 1 else {}
    check(vitals.get("heart_rate") == 112 and vitals.get("resp_rate") == 26,
          f"ingested patient listed with both vitals recorded together ({vitals})")
    b_patients = call("GET", "/patients", token=tokens["b"]["access_token"])[1]
    check(all(p["external_id"] != MRN for p in b_patients), "site B cannot see site A's patient (API)")

    code, pred = call("POST", "/predict", {"patient_id": f"e2e-{RUN}", "age_years": 70,
                                           "vitals": {"heart_rate": 118, "resp_rate": 24, "sbp": 96,
                                                      "temperature": 38.6, "lactate": 2.4}}, a)
    check(code == 200 and pred.get("clinical_use") == "not-permitted"
          and pred.get("rule_risk_level") and pred.get("model_risk_level"),
          "prediction carries rule and model levels and clinical_use=not-permitted")
    for method in ("POST", "PUT", "PATCH"):
        code = call(method, "/model/status", {"clinical_use": "permitted"}, a)[0]
        check(code in (404, 405), f"{method} /model/status cannot change clinical use (HTTP {code})")
    code = call("POST", "/alerts/test", {"channel": "websocket"}, a)[0]
    check(code == 403, f"test alerts are admin-only (nurse got HTTP {code})")

    # Refresh rotation and replay, then logout revocation.
    fresh = login("a")
    code, rotated = call("POST", "/auth/refresh", {"refresh_token": fresh["refresh_token"]})
    check(code == 200 and rotated.get("access_token"), "refresh token rotates")
    replayed = call("POST", "/auth/refresh", {"refresh_token": fresh["refresh_token"]})[0]
    check(replayed == 401, f"replayed refresh token is rejected (HTTP {replayed})")
    after_replay = call("GET", "/patients", token=rotated["access_token"])[0]
    check(after_replay == 401, f"replay revokes the whole session family (HTTP {after_replay})")
    session = login("a")
    check(call("POST", "/auth/logout", None, session["access_token"])[0] == 200, "logout")
    revoked = call("GET", "/patients", token=session["access_token"])[0]
    check(revoked == 401, f"access token is rejected after logout (HTTP {revoked})")


def browser_checks() -> None:
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        console: list[str] = []

        def new_page():
            context = browser.new_context()
            page = context.new_page()
            page.on("console", lambda msg: console.append(msg.text) if msg.type in ("error", "warning") else None)
            page.set_default_timeout(20_000)
            return page

        def sign_in(page, key: str, password: str = PASSWORD) -> None:
            page.goto(BASE + "/login")
            page.fill("#email", USERS[key][0])
            page.fill("#password", password)
            page.get_by_role("button", name="Sign in").click()
            agree = page.get_by_role("button", name="I Agree — Enter Application")
            agree.wait_for()
            page.locator("input[type=checkbox]").first.check()
            agree.click()
            page.wait_for_url("**/dashboard")

        page = new_page()
        page.goto(BASE + "/")
        expect(page.get_by_text("Find the patients who need a")).to_be_visible()
        check(not any("basename" in m for m in console), "SPA renders at / (no router basename mismatch)")

        sign_in(page, "a")
        expect(page.get_by_text(MRN)).to_be_visible()
        check(page.get_by_text("P-1042").count() == 0, "dashboard shows no demo patients in live mode")

        page.goto(BASE + "/patients")
        card = page.get_by_role("button", name=MRN)
        expect(card).to_contain_text("112 bpm")
        check(True, "patients page shows the FHIR-ingested patient and its vitals")

        page.goto(BASE + "/predict")
        for field, value in (("#vitals-patient-id", f"e2e-ui-{RUN}"), ("#vitals-heart-rate", "118"),
                             ("#vitals-resp-rate", "24"), ("#vitals-sbp", "96")):
            page.fill(field, value)
        page.get_by_role("button", name="Run Prediction").click()
        expect(page.get_by_test_id("clinical-use")).to_have_text("Clinical use: not permitted")
        expect(page.get_by_label("Risk level sources")).to_contain_text("Rule-based scores")
        check(True, "predict page separates rule and model levels and states clinical use is not permitted")

        page_b = new_page()
        sign_in(page_b, "b")
        page_b.goto(BASE + "/patients")
        expect(page_b.get_by_text("No patients match")).to_be_visible()
        check(page_b.get_by_text(MRN).count() == 0, "site B nurse does not see site A's patient (UI)")

        token = api_python(
            "from sepsis_vitals.auth.service import request_password_reset\n"
            "from sepsis_vitals.db import SessionLocal\n"
            f"db = SessionLocal(); print(request_password_reset({USERS['b'][0]!r}, db)); db.close()"
        )
        reset_page = new_page()
        reset_page.goto(f"{BASE}/login#reset_token={urllib.request.quote(token)}")
        reset_page.fill("#new-password", NEW_PASSWORD)
        reset_page.fill("#confirm-password", NEW_PASSWORD)
        reset_page.get_by_role("button", name="Set new password").click()
        expect(reset_page.get_by_role("status")).to_contain_text("Password updated")
        check("reset_token" not in reset_page.url, "reset token removed from the address bar")
        reuse = call("POST", "/auth/password-reset/confirm", {"token": token, "new_password": PASSWORD})[0]
        check(reuse in (400, 401), f"reset token cannot be reused (HTTP {reuse})")
        check(call("POST", "/auth/login", {"email": USERS["b"][0], "password": PASSWORD})[0] == 401,
              "old password no longer works")
        sign_in(new_page(), "b", NEW_PASSWORD)
        check(True, "sign-in with the new password")
        browser.close()


def outage_checks() -> None:
    subprocess.run(["docker", "compose", "stop", "db"], check=True, capture_output=True, timeout=120)
    wait_for(lambda: call("GET", "/ready")[0] == 503, "/ready to report the database outage")
    check(call("GET", "/health")[0] == 200, "liveness stays up while the database is down")
    subprocess.run(["docker", "compose", "start", "db"], check=True, capture_output=True, timeout=120)
    wait_for(lambda: call("GET", "/ready")[0] == 200, "/ready to recover after the database returns")
    check(True, "readiness recovers")


def main() -> int:
    seed()
    tokens = {key: login(key) for key in USERS}
    api_checks(tokens)
    browser_checks()
    outage_checks()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall end-to-end checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
