"""
sepsis_vitals.auth.jwt
Authentication utilities: password hashing, RBAC, MFA (TOTP), and account lockout.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    import bcrypt
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Return True if *password* matches the bcrypt *hashed* value."""
    import bcrypt
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


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
