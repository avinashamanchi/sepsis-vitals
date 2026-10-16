"""
tests/test_mfa_replay.py — N52: each TOTP time step is accepted once per
secret, on every path that checks a code (enrollment confirmation, login,
self-disable), across concurrent requests, workers and restarts; recovery
codes are single-use under concurrency; failures deny access.

A controlled clock (tests.test_mfa._Clock) replaces real time; nothing sleeps.
"""

from __future__ import annotations

import importlib.util
import threading

import pytest

HAS_DEPS = all(importlib.util.find_spec(m) for m in ("fastapi", "pyotp"))
pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="fastapi/pyotp not installed")

from tests.test_mfa import (  # noqa: E402,F401  (fixtures are used by name)
    PASSWORD,
    _clear_lockout,
    _code,
    _enroll,
    _login,
    _user,
    client,
    clock,
)


def _enrolled(client, clock):
    """A user with MFA on; the confirmation consumed the current step."""
    email = _user()
    token = _login(client, email).json()["access_token"]
    secret, codes = _enroll(client, token, clock)  # advances the clock one step
    return email, secret, codes


def _db_user(email):
    from sepsis_vitals.db import SessionLocal, User
    from sepsis_vitals.security import blind_index_candidates

    db = SessionLocal()
    user = db.query(User).filter(User.email_hash.in_(blind_index_candidates(email))).one()
    return db, user


def _last_step(email):
    db, user = _db_user(email)
    try:
        return user.totp_last_step
    finally:
        db.close()


# -- login --------------------------------------------------------------------------

def test_a_code_signs_in_once(client, clock):
    email, secret, _ = _enrolled(client, clock)
    code = _code(secret, clock)
    assert _login(client, email, otp=code).status_code == 200
    replay = _login(client, email, otp=code)
    _clear_lockout(email)
    wrong = _login(client, email, otp="000000" if code != "000000" else "111111")
    # a replay is indistinguishable from a wrong code
    assert replay.status_code == wrong.status_code == 401 and replay.json() == wrong.json()
    _clear_lockout(email)
    clock.advance()
    assert _login(client, email, otp=_code(secret, clock)).status_code == 200


def test_the_enrollment_code_cannot_be_reused_to_sign_in(client, clock):
    email = _user()
    token = _login(client, email).json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    secret = client.post("/auth/mfa/enroll", headers=auth).json()["secret"]
    code = _code(secret, clock)
    assert client.post("/auth/mfa/confirm", headers=auth, json={"code": code}).status_code == 200
    assert _login(client, email, otp=code).status_code == 401
    _clear_lockout(email)
    clock.advance()
    assert _login(client, email, otp=_code(secret, clock)).status_code == 200


def test_drift_tolerance_never_reopens_a_used_step(client, clock):
    """Adjacent steps are accepted, but only moving forward: after the next
    step's code (fast client clock) is used, the current step's code is refused.
    (Checked at the service layer: the login route is throttled per account.)"""
    email, secret, _ = _enrolled(client, clock)
    assert _attempt(email, _code(secret, clock, -1)) is False  # the step the confirmation used
    assert _attempt(email, _code(secret, clock, +1)) is True   # next step: within drift
    assert _attempt(email, _code(secret, clock)) is False      # earlier than the used step
    assert _attempt(email, _code(secret, clock, +1)) is False  # the used step itself
    clock.advance(2)
    assert _attempt(email, _code(secret, clock, -1)) is False  # still the used step
    clock.advance()
    assert _attempt(email, _code(secret, clock, -1)) is True   # previous step, newer than used
    assert _login(client, email, otp=_code(secret, clock)).status_code == 200  # and via the route


@pytest.mark.parametrize("bad", ["abcdef", "12345", "1234567", "", "12 34 5", "１２３４５６"])
def test_malformed_and_wrong_codes_consume_nothing(client, clock, bad):
    from sepsis_vitals.auth.service import verify_second_factor

    email, secret, _ = _enrolled(client, clock)
    before = _last_step(email)
    db, user = _db_user(email)
    try:
        assert verify_second_factor(user, bad, db) is False
        wrong = "000000" if _code(secret, clock) != "000000" else "111111"
        assert verify_second_factor(user, wrong, db) is False
        db.commit()
    finally:
        db.close()
    assert _last_step(email) == before
    assert _login(client, email, otp=_code(secret, clock)).status_code == 200


def test_users_without_mfa_are_unaffected(client, clock):
    from sepsis_vitals.auth.service import verify_second_factor

    email = _user()
    assert _login(client, email).status_code == 200
    assert _login(client, email, otp="123456").status_code == 200  # an unrequested code is ignored
    db, user = _db_user(email)
    try:
        assert verify_second_factor(user, "123456", db) is False  # no secret: nothing to match
        assert user.totp_last_step is None
    finally:
        db.close()


# -- other code-checking paths -------------------------------------------------------

def test_disabling_mfa_needs_an_unused_code(client, clock):
    email, secret, _ = _enrolled(client, clock)
    code = _code(secret, clock)
    session = _login(client, email, otp=code).json()["access_token"]
    auth = {"Authorization": f"Bearer {session}"}
    replay = client.post("/auth/mfa/disable", headers=auth, json={"password": PASSWORD, "code": code})
    assert replay.status_code == 401
    clock.advance()
    fresh = client.post("/auth/mfa/disable", headers=auth, json={"password": PASSWORD, "code": _code(secret, clock)})
    assert fresh.status_code == 200


def test_reset_and_reenrollment_start_a_new_replay_history(client, clock):
    from sepsis_vitals.auth.mfa import reset_mfa

    email, secret, _ = _enrolled(client, clock)
    assert _login(client, email, otp=_code(secret, clock)).status_code == 200
    assert _last_step(email) is not None
    db, user = _db_user(email)
    try:
        reset_mfa(user, db)
    finally:
        db.close()
    assert _last_step(email) is None
    token = _login(client, email).json()["access_token"]  # password alone again
    auth = {"Authorization": f"Bearer {token}"}
    new_secret = client.post("/auth/mfa/enroll", headers=auth).json()["secret"]
    assert new_secret != secret
    # the new secret's code for the *same* step is accepted: history was per secret
    assert client.post("/auth/mfa/confirm", headers=auth, json={"code": _code(new_secret, clock)}).status_code == 200
    assert _login(client, email, otp=_code(secret, clock, +1)).status_code == 401  # old secret is dead


def test_a_recovery_code_does_not_reopen_used_totp_steps(client, clock):
    email, secret, codes = _enrolled(client, clock)
    code = _code(secret, clock)
    assert _login(client, email, otp=code).status_code == 200
    assert _login(client, email, otp=codes[0]).status_code == 200
    assert _login(client, email, otp=code).status_code == 401
    _clear_lockout(email)
    assert _login(client, email, otp=codes[0]).status_code == 401


# -- concurrency, workers, restarts ------------------------------------------------------

def _race(n, attempt):
    barrier = threading.Barrier(n)
    results, errors = [], []

    def run():
        try:
            barrier.wait(timeout=10)
            results.append(attempt())
        except Exception as exc:  # pragma: no cover - reported by the assertion
            errors.append(repr(exc))

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    return results


def _attempt(email, code):
    from sepsis_vitals.auth.service import verify_second_factor

    db, user = _db_user(email)
    try:
        ok = verify_second_factor(user, code, db)
        if ok:
            db.commit()  # consumption commits with the (simulated) successful sign-in
        else:
            db.rollback()
        return ok
    finally:
        db.close()


def test_concurrent_submissions_of_one_code_succeed_once(client, clock):
    email, secret, _ = _enrolled(client, clock)
    code = _code(secret, clock)
    assert sorted(_race(6, lambda: _attempt(email, code))) == [False] * 5 + [True]


def test_concurrent_use_of_one_recovery_code_succeeds_once(client, clock):
    email, _, codes = _enrolled(client, clock)
    assert sorted(_race(6, lambda: _attempt(email, codes[0]))) == [False] * 5 + [True]


def test_separate_worker_processes_accept_a_code_once(client, clock, monkeypatch):
    """Independent processes (like uvicorn workers) share only the database."""
    import multiprocessing as mp

    from tests import mfa_replay_worker

    email, secret, _ = _enrolled(client, clock)
    code = _code(secret, clock)
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(4), ctx.Queue()
    procs = [ctx.Process(target=mfa_replay_worker.login_once,
                         args=(email, PASSWORD, code, clock.t, barrier, results)) for _ in range(4)]
    for p in procs:
        p.start()
    outcomes = sorted(results.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    assert outcomes.count("ok") == 1, outcomes
    assert set(outcomes) <= {"ok", "InvalidCredentialsError", "AccountLockedError"}, outcomes


def test_used_steps_survive_a_restart(client, clock):
    """Replay state is in the database, not in process memory."""
    from sepsis_vitals import db as db_module

    email, secret, _ = _enrolled(client, clock)
    code = _code(secret, clock)
    assert _login(client, email, otp=code).status_code == 200
    db_module.engine.dispose()  # drop every pooled connection, as a restart would
    _clear_lockout(email)
    assert _attempt(email, code) is False


# -- transactions and failures --------------------------------------------------------

def test_consumption_is_undone_when_the_sign_in_is_rolled_back(client, clock):
    from sepsis_vitals.auth.service import verify_second_factor

    email, secret, _ = _enrolled(client, clock)
    code = _code(secret, clock)
    db, user = _db_user(email)
    try:
        assert verify_second_factor(user, code, db) is True
        db.rollback()  # the surrounding sign-in failed after the check
    finally:
        db.close()
    assert _attempt(email, code) is True  # not burnt by an attempt that never completed


def test_a_database_failure_denies_access(client, clock, monkeypatch):
    import sqlalchemy.exc as sa_exc
    from sqlalchemy.orm import Session

    email, secret, _ = _enrolled(client, clock)
    real_execute = Session.execute

    def failing_execute(self, statement, *args, **kwargs):
        if "totp_last_step" in str(statement):
            raise sa_exc.OperationalError("UPDATE users", {}, Exception("database unavailable"))
        return real_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "execute", failing_execute)
    with pytest.raises(sa_exc.OperationalError):
        client.post("/auth/login", json={"email": email, "password": PASSWORD, "otp": _code(secret, clock)})
    monkeypatch.setattr(Session, "execute", real_execute)
    assert _attempt(email, _code(secret, clock)) is True  # nothing was consumed


# -- migration ------------------------------------------------------------------------

def test_migration_adds_a_nullable_step_for_existing_accounts():
    """Existing users get NULL (no step used), so their next valid code works."""
    from pathlib import Path

    import sqlalchemy as sa
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("mig005", root / "alembic" / "versions" / "005_totp_replay_protection.py")
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)
    assert mig.down_revision == "d4e5f6a7b8c9"

    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id TEXT PRIMARY KEY, mfa_enabled BOOLEAN)"))
        conn.execute(sa.text("INSERT INTO users (id, mfa_enabled) VALUES ('u1', 1)"))
        with Operations.context(MigrationContext.configure(conn)):
            mig.upgrade()
        assert conn.execute(sa.text("SELECT totp_last_step FROM users")).scalar() is None
        with Operations.context(MigrationContext.configure(conn)):
            mig.downgrade()
        assert "totp_last_step" not in {c["name"] for c in sa.inspect(conn).get_columns("users")}


# -- the flow as a whole ----------------------------------------------------------------

def test_mfa_flows_log_no_secrets_codes_or_recovery_codes(client, clock, caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        email, secret, codes = _enrolled(client, clock)
        code = _code(secret, clock)
        _login(client, email, otp=code)
        _login(client, email, otp=code)            # replay
        _clear_lockout(email)
        _login(client, email, otp=codes[0])        # recovery code
    text = caplog.text
    assert secret not in text and code not in text
    assert not any(c in text or c.replace("-", "") in text for c in codes)
    assert "AUDIT mfa_enabled" in text  # the audit trail is still there


def test_policy_stays_off_unless_configured(client, clock, monkeypatch):
    """No deployment default enables enforcement for existing accounts."""
    from pathlib import Path

    from sepsis_vitals.auth.service import mfa_required_roles

    monkeypatch.delenv("SEPSIS_MFA_REQUIRED_ROLES", raising=False)
    assert mfa_required_roles() == frozenset()
    root = Path(__file__).resolve().parents[1]
    assert "SEPSIS_MFA_REQUIRED_ROLES" not in (root / "docker" / "docker-compose.yml").read_text()
    assert "SEPSIS_MFA_REQUIRED_ROLES" not in (root / "terraform" / "main.tf").read_text()
    env_example = (root / ".env.example").read_text()
    assert "SEPSIS_MFA_REQUIRED_ROLES=\n" in env_example
