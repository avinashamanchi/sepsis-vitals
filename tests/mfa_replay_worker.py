"""Worker for the multi-process TOTP replay test (tests/test_mfa_replay.py).

Each spawned process is an independent "API worker": its own interpreter,
engine and connection pool, against the shared test database.
"""

from __future__ import annotations


def login_once(email: str, password: str, code: str, fixed_time: float, barrier, results) -> None:
    import sepsis_vitals.auth.jwt as jwt_mod

    jwt_mod._now = lambda: fixed_time  # the same TOTP step in every worker
    from sepsis_vitals.auth.service import AuthServiceError, login_user
    from sepsis_vitals.db import SessionLocal

    db = SessionLocal()
    try:
        barrier.wait(timeout=60)
        try:
            login_user(email, password, db, otp=code)
            results.put("ok")
        except AuthServiceError as exc:
            results.put(type(exc).__name__)
    except Exception as exc:  # reported, the parent asserts on it
        results.put(f"error:{type(exc).__name__}")
    finally:
        db.close()
