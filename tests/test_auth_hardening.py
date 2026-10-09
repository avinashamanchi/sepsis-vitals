"""
tests/test_auth_hardening.py — lockout cap, single-use reset tokens, refresh
replay detection, and password-reset email delivery.
"""

from __future__ import annotations

import importlib.util
import logging
import uuid

import pytest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")

PASSWORD = "Correct-Horse-Battery-9!"


@pytest.fixture()
def db(monkeypatch):
    monkeypatch.setenv("SEPSIS_JWT_SECRET", "auth-hardening-test-secret-0123456789")
    from sepsis_vitals.db import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    yield session
    session.close()


def _register(db, role: str = "nurse") -> str:
    from sepsis_vitals.auth.service import register_user

    email = f"user-{uuid.uuid4().hex[:8]}@example.org"
    register_user(email, PASSWORD, role, "SITE-T", db)
    return email


# -- lockout ------------------------------------------------------------------

def test_lockout_is_capped_and_never_overflows():
    from sepsis_vitals.auth.jwt import MAX_LOCKOUT_SECONDS, lockout_duration

    assert lockout_duration(0) == 0
    assert lockout_duration(3) == 4
    assert lockout_duration(50) == MAX_LOCKOUT_SECONDS
    assert lockout_duration(5000) == MAX_LOCKOUT_SECONDS  # 2**4999 would overflow float


# -- password reset -------------------------------------------------------------

def test_reset_token_is_single_use(db):
    from sepsis_vitals.auth.service import (
        InvalidTokenError,
        request_password_reset,
        reset_password,
    )

    email = _register(db)
    token = request_password_reset(email, db)
    assert reset_password(token, "Another-Strong-Passphrase-7", db) is True
    with pytest.raises(InvalidTokenError):
        reset_password(token, "Yet-Another-Passphrase-8!", db)


def test_tampered_reset_token_rejected(db):
    from sepsis_vitals.auth.service import InvalidTokenError, request_password_reset, reset_password

    token = request_password_reset(_register(db), db)
    user_id, expiry, sig = token.split(":")
    forged = f"{user_id}:{int(expiry) + 86400}:{sig}"
    with pytest.raises(InvalidTokenError):
        reset_password(forged, "Another-Strong-Passphrase-7", db)


def test_reset_email_carries_token_in_url_fragment(monkeypatch):
    from sepsis_vitals.auth import mailer

    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, context):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = user

        def send_message(self, msg):
            sent["msg"] = msg

    monkeypatch.setenv("SMTP_HOST", "smtp.example.org")
    monkeypatch.setenv("SMTP_USERNAME", "mailer")
    monkeypatch.setenv("SEPSIS_APP_URL", "https://study.example.org/app/")
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)

    assert mailer.send_password_reset("nurse@example.org", "u1:123:abc") is True
    body = sent["msg"].get_content()
    assert "https://study.example.org/app/login#reset_token=u1%3A123%3Aabc" in body
    assert sent["tls"] is True and sent["msg"]["To"] == "nurse@example.org"


def test_reset_email_not_sent_without_smtp(monkeypatch):
    from sepsis_vitals.auth import mailer

    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert mailer.send_password_reset("nurse@example.org", "t") is False


def test_reset_request_does_not_log_email_and_is_uniform(db, monkeypatch, caplog):
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api
    from sepsis_vitals.auth import router as auth_router

    sent: list[str] = []
    monkeypatch.setattr(auth_router, "send_password_reset", lambda to, tok: sent.append(to))
    for dep in (api.check_rate_limit, api.check_auth_rate_limit):
        api.app.dependency_overrides[dep] = lambda: None
    email = _register(db)
    try:
        with TestClient(api.app) as client, caplog.at_level(logging.DEBUG):
            known = client.post("/auth/password-reset/request", json={"email": email})
            unknown = client.post(
                "/auth/password-reset/request", json={"email": "nobody@example.org"}
            )
    finally:
        api.app.dependency_overrides.clear()
    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json()
    assert sent == [email]
    assert email not in caplog.text


# -- refresh rotation -------------------------------------------------------------

def test_refresh_token_replay_revokes_all_sessions(db):
    from sepsis_vitals.auth.service import InvalidTokenError, login_user, refresh_access_token
    from sepsis_vitals.auth.tokens import TokenError, decode_token

    email = _register(db)
    first = login_user(email, PASSWORD, db)
    rotated = refresh_access_token(first["refresh_token"], db)

    with pytest.raises(InvalidTokenError, match="reuse"):
        refresh_access_token(first["refresh_token"], db)  # stolen token replayed

    with pytest.raises(InvalidTokenError):
        refresh_access_token(rotated["refresh_token"], db)  # family is dead too
    with pytest.raises(TokenError):
        decode_token(rotated["access_token"])


def test_lockout_check_accepts_naive_timestamps_from_sqlite():
    """Regression: SQLite returns naive datetimes; comparing them raised TypeError (login 500)."""
    from datetime import datetime, timedelta

    from sepsis_vitals.auth.jwt import is_locked_out

    naive_future = datetime.utcnow() + timedelta(minutes=5)
    naive_past = datetime.utcnow() - timedelta(minutes=5)
    assert is_locked_out(naive_future) is True
    assert is_locked_out(naive_past) is False


def test_session_issued_right_after_revocation_is_valid():
    """Regression: whole-second revocation rejected tokens issued in the same second."""
    from sepsis_vitals.auth.tokens import TokenBlacklist

    bl = TokenBlacklist()
    revoked_ms = 1_800_000_000_500
    bl._user_revoked_before["u-1"] = revoked_ms
    assert bl.is_revoked("j1", user_id="u-1", issued_at_ms=revoked_ms - 1) is True
    assert bl.is_revoked("j2", user_id="u-1", issued_at_ms=revoked_ms + 1) is False
    # tokens without iat_ms: same second as the revocation counts as revoked
    assert bl.is_revoked("j3", user_id="u-1", issued_at=1_800_000_000) is True
    assert bl.is_revoked("j4", user_id="u-1", issued_at=1_800_000_001) is False


def test_revocations_stored_in_seconds_still_apply():
    from sepsis_vitals.auth.tokens import TokenBlacklist

    bl = TokenBlacklist()
    bl._user_revoked_before["u-2"] = 1_800_000_000  # legacy value in seconds
    assert bl.is_revoked("j", user_id="u-2", issued_at_ms=1_799_999_999_000) is True
    assert bl.is_revoked("j", user_id="u-2", issued_at_ms=1_800_000_001_000) is False
