"""
sepsis_vitals.auth.jwt
Authentication utilities: password hashing, JWT token generation,
RBAC, MFA (TOTP), account lockout, and user management.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import base64


# ---------------------------------------------------------------------------
# Password hashing — uses PBKDF2 from stdlib as bcrypt fallback
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return a secure hash of *password*.

    Uses bcrypt if available, otherwise PBKDF2-SHA256 (stdlib, no deps).
    """
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        # PBKDF2 fallback — no external deps needed
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), 200_000)
        return f"pbkdf2:sha256:200000${salt}${dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Return True if *password* matches the hashed value."""
    if hashed.startswith("pbkdf2:"):
        parts = hashed.split("$")
        if len(parts) != 3:
            return False
        _, salt, stored_hash = parts
        iterations = int(hashed.split(":")[2].split("$")[0])
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), iterations)
        return hmac.compare_digest(dk.hex(), stored_hash)

    try:
        import bcrypt
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Role-Based Access Control (RBAC)
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: dict[str, list[str] | str] = {
    "nurse": ["vital:read", "alert:escalate"],
    "researcher": ["vital:read", "model:read", "report:read"],
    "system_admin": "*",  # wildcard – all permissions
}


class AuthorizationError(Exception):
    """Raised when a role lacks the required permission."""


def check_permission(role: str, permission: str) -> None:
    """Raise :class:`AuthorizationError` if *role* does not have *permission*."""
    perms = ROLE_PERMISSIONS.get(role)
    if perms is None:
        raise AuthorizationError(f"Unknown role: {role}")
    if perms == "*":
        return  # system_admin – everything allowed
    if permission not in perms:
        raise AuthorizationError(
            f"Role '{role}' does not have permission '{permission}'"
        )


# ---------------------------------------------------------------------------
# Multi-Factor Authentication (TOTP)
# ---------------------------------------------------------------------------


def generate_totp_secret() -> str:
    """Return a base32-encoded random secret suitable for TOTP."""
    import pyotp
    return pyotp.random_base32()


def verify_totp(secret: str, code: str) -> bool:
    """Return True if *code* is a valid TOTP token for *secret*."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.verify(code)


def get_totp_uri(secret: str, email: str) -> str:
    """Return an ``otpauth://`` provisioning URI for *email*."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=email, issuer_name="SepsisVitals")


# ---------------------------------------------------------------------------
# Account lockout
# ---------------------------------------------------------------------------


def lockout_duration(failures: int) -> float:
    """Return lockout duration in seconds using exponential backoff.

    0 failures -> 0 seconds.  Each subsequent failure doubles the duration
    starting from a 1-second base.
    """
    if failures <= 0:
        return 0.0
    return float(2 ** (failures - 1))


def is_locked_out(lockout_until: Optional[datetime]) -> bool:
    """Return True if the account is currently locked out."""
    if lockout_until is None:
        return False
    return datetime.now(timezone.utc) < lockout_until


# ---------------------------------------------------------------------------
# JWT token generation and verification (HMAC-SHA256, no external deps)
# ---------------------------------------------------------------------------


def _get_jwt_secret() -> str:
    """Return the JWT signing secret from env. Raises in production if unset."""
    secret = os.environ.get("SEPSIS_JWT_SECRET")
    if secret:
        return secret
    env = os.environ.get("SEPSIS_ENV", "development")
    if env == "production":
        raise RuntimeError(
            "SEPSIS_JWT_SECRET must be set in production. "
            "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    # Dev-only: generate a per-process ephemeral secret (never hardcoded)
    if not hasattr(_get_jwt_secret, "_ephemeral"):
        _get_jwt_secret._ephemeral = secrets.token_hex(32)  # type: ignore[attr-defined]
        import logging
        logging.getLogger(__name__).warning(
            "SEPSIS_JWT_SECRET not set — using ephemeral secret (tokens won't survive restart)"
        )
    return _get_jwt_secret._ephemeral  # type: ignore[attr-defined]


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    s += "=" * (padding % 4)
    return base64.urlsafe_b64decode(s)


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    expires_minutes: int = 60,
) -> str:
    """Create a JWT access token (HMAC-SHA256 signed).

    No external JWT library needed — uses stdlib hmac + json + base64.
    """
    secret = _get_jwt_secret()
    now = time.time()
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": int(now),
        "exp": int(now + expires_minutes * 60),
    }

    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url_encode(json.dumps(payload).encode())
    signature_input = f"{header}.{body}"
    sig = hmac.new(secret.encode(), signature_input.encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(sig)}"


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token. Returns payload dict or None if invalid."""
    secret = _get_jwt_secret()
    parts = token.split(".")
    if len(parts) != 3:
        return None

    header_b64, body_b64, sig_b64 = parts
    signature_input = f"{header_b64}.{body_b64}"
    expected_sig = hmac.new(secret.encode(), signature_input.encode(), hashlib.sha256).digest()

    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception:
        return None

    if not hmac.compare_digest(expected_sig, actual_sig):
        return None

    try:
        payload = json.loads(_b64url_decode(body_b64))
    except Exception:
        return None

    # Check expiration
    if payload.get("exp", 0) < time.time():
        return None

    return payload


# ---------------------------------------------------------------------------
# SQLite-backed user database
# ---------------------------------------------------------------------------


class UserStore:
    """Database-backed user management with bcrypt/PBKDF2 password hashing.

    Replaces the in-memory API_KEYS dict with a real persistent user table.
    """

    def __init__(self, db_path: str = "models/users.db"):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'nurse',
                is_active INTEGER NOT NULL DEFAULT 1,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                lockout_until TEXT,
                totp_secret TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """)
        self._conn.commit()

    def create_user(
        self, email: str, password: str, role: str = "nurse"
    ) -> Optional[dict]:
        """Create a new user. Returns user dict or None if email exists."""
        user_id = secrets.token_hex(16)
        pw_hash = hash_password(password)
        now = time.time()
        try:
            self._conn.execute(
                "INSERT INTO users (id, email, password_hash, role, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, email.lower(), pw_hash, role, now, now),
            )
            self._conn.commit()
            return {"id": user_id, "email": email.lower(), "role": role}
        except sqlite3.IntegrityError:
            return None

    def authenticate(self, email: str, password: str) -> Optional[dict]:
        """Authenticate a user. Returns user dict with JWT token, or None."""
        row = self._conn.execute(
            "SELECT id, email, password_hash, role, is_active, failed_attempts, lockout_until "
            "FROM users WHERE email = ?",
            (email.lower(),),
        ).fetchone()

        if row is None:
            return None

        user_id, user_email, pw_hash, role, is_active, failed_attempts, lockout_str = row

        if not is_active:
            return None

        # Check lockout
        if lockout_str:
            lockout_until = datetime.fromisoformat(lockout_str)
            if is_locked_out(lockout_until):
                return None

        if not verify_password(password, pw_hash):
            # Increment failed attempts
            new_failures = failed_attempts + 1
            lock_dur = lockout_duration(new_failures)
            lockout = None
            if lock_dur > 0:
                lockout = (datetime.now(timezone.utc) + timedelta(seconds=lock_dur)).isoformat()
            self._conn.execute(
                "UPDATE users SET failed_attempts = ?, lockout_until = ?, updated_at = ? WHERE id = ?",
                (new_failures, lockout, time.time(), user_id),
            )
            self._conn.commit()
            return None

        # Success — reset failed attempts
        self._conn.execute(
            "UPDATE users SET failed_attempts = 0, lockout_until = NULL, updated_at = ? WHERE id = ?",
            (time.time(), user_id),
        )
        self._conn.commit()

        token = create_access_token(user_id, user_email, role)
        return {"id": user_id, "email": user_email, "role": role, "token": token}

    def get_user(self, user_id: str) -> Optional[dict]:
        """Look up a user by ID."""
        row = self._conn.execute(
            "SELECT id, email, role, is_active FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "email": row[1], "role": row[2], "is_active": bool(row[3])}

    def close(self):
        self._conn.close()
