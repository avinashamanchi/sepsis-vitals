"""
tests/test_mfa.py — TOTP enrollment, login enforcement, recovery codes, role
policy, bypass resistance and administrator reset.
"""

from __future__ import annotations

import importlib.util
import uuid

import pytest

HAS_DEPS = all(importlib.util.find_spec(m) for m in ("fastapi", "pyotp"))
pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="fastapi/pyotp not installed")

PASSWORD = "Correct-Horse-Battery-9!"


class _Clock:
    """Controlled TOTP clock: tests move between 30 s steps without sleeping."""

    def __init__(self) -> None:
        self.t = 1_800_000_015.0  # middle of a step

    def __call__(self) -> float:
        return self.t

    def advance(self, steps: int = 1) -> None:
        self.t += 30 * steps


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr("sepsis_vitals.auth.jwt._now", c)
    return c


def _code(secret: str, clock: "_Clock", offset_steps: int = 0) -> str:
    import pyotp

    return pyotp.TOTP(secret).at(clock.t + 30 * offset_steps)


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api
    from sepsis_vitals.db import init_db

    monkeypatch.setenv("SEPSIS_JWT_SECRET", "mfa-test-secret-0123456789abcdef")
    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", True)
    for dep in (api.check_rate_limit, api.check_auth_rate_limit):
        api.app.dependency_overrides[dep] = lambda: None
    init_db()
    with TestClient(api.app) as c:
        yield c
    api.app.dependency_overrides.clear()


def _user(role: str = "nurse") -> str:
    from sepsis_vitals.auth.service import register_user
    from sepsis_vitals.db import SessionLocal

    email = f"mfa-{uuid.uuid4().hex[:8]}@example.org"
    db = SessionLocal()
    try:
        try:
            register_user(email, PASSWORD, role, "SITE-M", db)
        except Exception:  # enrollment-required policy may refuse to issue tokens
            pass
    finally:
        db.close()
    return email


def _clear_lockout(email):
    """Simulate the backoff window elapsing after a deliberate failure."""
    from sepsis_vitals.db import SessionLocal, User
    from sepsis_vitals.security import compute_blind_index

    db = SessionLocal()
    user = db.query(User).filter(User.email_hash == compute_blind_index(email)).first()
    user.locked_until = None
    db.commit()
    db.close()


def _login(client, email, otp=None):
    body = {"email": email, "password": PASSWORD}
    if otp is not None:
        body["otp"] = otp
    return client.post("/auth/login", json=body)


def _enroll(client, token, clock):
    """Enroll and confirm at the current step (which the confirmation consumes)."""
    auth = {"Authorization": f"Bearer {token}"}
    started = client.post("/auth/mfa/enroll", headers=auth)
    assert started.status_code == 200, started.text
    secret = started.json()["secret"]
    assert started.json()["otpauth_uri"].startswith("otpauth://totp/")
    confirmed = client.post("/auth/mfa/confirm", headers=auth, json={"code": _code(secret, clock)})
    assert confirmed.status_code == 200, confirmed.text
    clock.advance()  # the confirming code is used up; sign-ins need the next one
    return secret, confirmed.json()["recovery_codes"]


def test_enrolled_user_needs_a_code_and_codes_are_checked(client, clock):
    email = _user()
    token = _login(client, email).json()["access_token"]
    secret, codes = _enroll(client, token, clock)
    assert len(codes) == 10

    assert _login(client, email).json()["detail"] == "mfa_required"
    assert _login(client, email, otp="000000").status_code == 401  # counts as a failure
    _clear_lockout(email)
    ok = _login(client, email, otp=_code(secret, clock))
    assert ok.status_code == 200 and ok.json()["access_token"]


def test_enabling_mfa_ends_existing_sessions(client, clock):
    email = _user()
    token = _login(client, email).json()["access_token"]
    _enroll(client, token, clock)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_recovery_codes_work_once(client, clock):
    email = _user()
    token = _login(client, email).json()["access_token"]
    _, codes = _enroll(client, token, clock)
    assert _login(client, email, otp=codes[0]).status_code == 200
    assert _login(client, email, otp=codes[0]).status_code == 401  # already used
    _clear_lockout(email)
    assert _login(client, email, otp=codes[1].lower().replace("-", "")).status_code == 200


def test_required_role_gets_only_an_enrollment_token(client, monkeypatch, clock):
    monkeypatch.setenv("SEPSIS_MFA_REQUIRED_ROLES", "nurse")
    email = _user("nurse")
    resp = _login(client, email)
    assert resp.status_code == 403 and resp.json()["detail"] == "mfa_enrollment_required"
    enrollment = resp.json()["enrollment_token"]
    assert "access_token" not in resp.json()

    # The enrollment token opens nothing but the MFA endpoints.
    auth = {"Authorization": f"Bearer {enrollment}"}
    assert client.get("/auth/me", headers=auth).status_code == 401
    assert client.get("/patients", headers=auth).status_code == 401

    secret, _ = _enroll(client, enrollment, clock)
    assert _login(client, email, otp=_code(secret, clock)).status_code == 200


def test_refresh_cannot_extend_a_pre_enforcement_session(client, monkeypatch):
    email = _user("nurse")
    refresh = _login(client, email).json()["refresh_token"]
    monkeypatch.setenv("SEPSIS_MFA_REQUIRED_ROLES", "nurse")
    assert client.post("/auth/refresh", json={"refresh_token": refresh}).status_code == 401


def test_policy_is_off_by_default(client, monkeypatch):
    monkeypatch.delenv("SEPSIS_MFA_REQUIRED_ROLES", raising=False)
    assert _login(client, _user("nurse")).status_code == 200


def test_required_role_cannot_disable_mfa(client, monkeypatch, clock):
    email = _user("nurse")
    token = _login(client, email).json()["access_token"]
    secret, _ = _enroll(client, token, clock)
    monkeypatch.setenv("SEPSIS_MFA_REQUIRED_ROLES", "nurse")
    session = _login(client, email, otp=_code(secret, clock)).json()["access_token"]
    clock.advance()
    resp = client.post("/auth/mfa/disable", headers={"Authorization": f"Bearer {session}"},
                       json={"password": PASSWORD, "code": _code(secret, clock)})
    assert resp.status_code == 403


def test_admin_reset_clears_mfa_but_not_for_self(client, clock):
    from sepsis_vitals.auth.tokens import create_access_token
    from sepsis_vitals.db import SessionLocal, User

    email = _user("nurse")
    token = _login(client, email).json()["access_token"]
    _enroll(client, token, clock)
    db = SessionLocal()
    target = db.query(User).filter(User.mfa_enabled.is_(True)).order_by(User.created_at.desc()).first()
    admin_email = _user("researcher")
    admin = db.query(User).filter(User.email_hash.isnot(None)).order_by(User.created_at.desc()).first()
    admin.role = "system_admin"
    db.commit()
    admin_token = create_access_token(admin.id, admin_email, "system_admin", None)
    auth = {"Authorization": f"Bearer {admin_token}"}
    assert client.put(f"/auth/users/{admin.id}/mfa/reset", headers=auth).status_code == 403
    assert client.put(f"/auth/users/{target.id}/mfa/reset", headers=auth).status_code == 200
    db.close()
    assert _login(client, email).status_code == 200  # password alone works again


def test_wrong_codes_trigger_the_lockout(client, clock):
    email = _user()
    token = _login(client, email).json()["access_token"]
    _enroll(client, token, clock)
    assert _login(client, email, otp="000000").status_code == 401
    assert _login(client, email, otp="000000").status_code == 403  # backoff in effect


def test_mfa_status_reports_counts_without_identities(client, monkeypatch, capsys, clock):
    """Rollout check: how many accounts in the enforced roles still need to enroll."""
    from sepsis_vitals.auth import admin_cli
    from sepsis_vitals.db import SessionLocal

    db = SessionLocal()
    try:
        before = admin_cli.mfa_status(db, frozenset({"nurse"}))
    finally:
        db.close()
    email = _user()
    _enroll(client, _login(client, email).json()["access_token"], clock)
    _user()  # a nurse without MFA
    db = SessionLocal()
    try:
        after = admin_cli.mfa_status(db, frozenset({"nurse"}))
    finally:
        db.close()
    nurses_before = before["roles"].get("nurse", {"accounts": 0, "mfa_enabled": 0, "with_recovery_codes": 0})
    assert after["roles"]["nurse"]["accounts"] == nurses_before["accounts"] + 2
    assert after["roles"]["nurse"]["mfa_enabled"] == nurses_before["mfa_enabled"] + 1
    assert after["roles"]["nurse"]["with_recovery_codes"] == nurses_before["with_recovery_codes"] + 1
    assert after["would_need_enrollment"] == before["would_need_enrollment"] + 1
    assert after["ready_to_enforce"] is False

    assert admin_cli.main(["mfa-status", "--roles", "nurse"]) == 0
    out = capsys.readouterr().out
    assert "@" not in out and email not in out


def test_totp_allows_one_step_of_clock_drift_and_no_more(clock):
    """A code from the adjacent step is accepted (RFC 6238 drift); older is not."""
    import pyotp

    from sepsis_vitals.auth.jwt import current_totp_step, match_totp_step, verify_totp

    secret = pyotp.random_base32()
    step = current_totp_step()
    assert match_totp_step(secret, _code(secret, clock)) == step
    assert match_totp_step(secret, _code(secret, clock, -1)) == step - 1
    assert match_totp_step(secret, _code(secret, clock, +1)) == step + 1
    assert not verify_totp(secret, _code(secret, clock, -2))
    assert not verify_totp(secret, _code(secret, clock, +2))
