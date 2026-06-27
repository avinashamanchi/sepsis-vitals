"""
sepsis_vitals.auth.tokens
JWT access and refresh token creation, verification, and revocation.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SECRET_KEY: str | None = None


def _get_secret_key() -> str:
    """Return the JWT signing key, reading from the environment on first call."""
    global _SECRET_KEY  # noqa: PLW0603
    if _SECRET_KEY is None:
        _SECRET_KEY = os.environ.get("SEPSIS_JWT_SECRET", "")
        if not _SECRET_KEY:
            raise RuntimeError(
                "SEPSIS_JWT_SECRET environment variable is not set. "
                "Generate a strong random value and export it before starting the server."
            )
    return _SECRET_KEY


_ALGORITHM = "HS256"


# ---------------------------------------------------------------------------
# Token blacklist (revocation)
# ---------------------------------------------------------------------------

class TokenBlacklist:
    """In-process token blacklist with optional Redis backend.

    Stores revoked JTI (JWT ID) values so that tokens can be invalidated
    before their natural expiry (logout, password reset, etc.).

    Also supports revoking all tokens for a user issued before a given
    timestamp (``revoke_all_for_user``).
    """

    _redis_client = None
    _redis_checked = False

    def __init__(self) -> None:
        self._blacklisted_jtis: set[str] = set()
        # user_id -> unix timestamp; tokens issued before this time are invalid
        self._user_revoked_before: dict[str, int] = {}

    @classmethod
    def _get_redis(cls):
        if not cls._redis_checked:
            cls._redis_checked = True
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                try:
                    import redis
                    cls._redis_client = redis.from_url(redis_url, decode_responses=True)
                    cls._redis_client.ping()
                except Exception:
                    cls._redis_client = None
        return cls._redis_client

    def revoke(self, jti: str, ttl_seconds: int = 1800) -> None:
        """Revoke a single token by its JTI."""
        r = self._get_redis()
        if r:
            try:
                r.setex(f"revoked:{jti}", ttl_seconds, "1")
                return
            except Exception:
                pass
        self._blacklisted_jtis.add(jti)

    def revoke_all_for_user(self, user_id: str) -> None:
        """Revoke all tokens for a user issued before now."""
        now = int(time.time())
        r = self._get_redis()
        if r:
            try:
                r.set(f"user_revoked:{user_id}", str(now))
                return
            except Exception:
                pass
        self._user_revoked_before[user_id] = now

    def is_revoked(self, jti: str, user_id: str | None = None, issued_at: int | None = None) -> bool:
        """Check if a token is revoked (by JTI or by user-level revocation)."""
        r = self._get_redis()
        if r:
            try:
                if r.exists(f"revoked:{jti}"):
                    return True
                if user_id and issued_at is not None:
                    revoked_before = r.get(f"user_revoked:{user_id}")
                    if revoked_before and issued_at <= int(revoked_before):
                        return True
                return False
            except Exception:
                pass
        # In-memory fallback
        if jti in self._blacklisted_jtis:
            return True
        if user_id and issued_at is not None:
            revoked_before = self._user_revoked_before.get(user_id)
            if revoked_before and issued_at <= revoked_before:
                return True
        return False


# Module-level singleton
_blacklist = TokenBlacklist()


def get_blacklist() -> TokenBlacklist:
    """Return the module-level token blacklist singleton."""
    return _blacklist


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    org_id: str | None,
    expires_minutes: int = 15,
) -> str:
    """Create a signed JWT access token.

    Parameters
    ----------
    user_id:
        The user's primary-key UUID.
    email:
        The user's email address.
    role:
        The user's RBAC role (``nurse``, ``researcher``, ``system_admin``).
    org_id:
        The organisation (site) identifier the user belongs to.
    expires_minutes:
        Token lifetime in minutes.  Defaults to 15 (HIPAA best practice).

    Returns
    -------
    str
        An encoded JWT string.
    """
    import jwt as pyjwt

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "role": role,
        "org_id": org_id,
        "type": "access",
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + expires_minutes * 60,
    }
    return pyjwt.encode(payload, _get_secret_key(), algorithm=_ALGORITHM)


def create_refresh_token(
    user_id: str,
    expires_days: int = 7,
    family_id: str | None = None,
) -> str:
    """Create a signed JWT refresh token.

    The refresh token carries only the subject (user id) and token type so it
    cannot be used directly as an access token.

    Parameters
    ----------
    user_id:
        The user's primary-key UUID.
    expires_days:
        Token lifetime in days.  Defaults to 7.
    family_id:
        Token family ID for refresh token rotation.  If not provided, a new
        family is created.

    Returns
    -------
    str
        An encoded JWT string.
    """
    import jwt as pyjwt

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "type": "refresh",
        "jti": uuid.uuid4().hex,
        "fid": family_id or uuid.uuid4().hex,
        "iat": now,
        "exp": now + expires_days * 86400,
    }
    return pyjwt.encode(payload, _get_secret_key(), algorithm=_ALGORITHM)


# ---------------------------------------------------------------------------
# Token decoding
# ---------------------------------------------------------------------------


class TokenError(Exception):
    """Raised when a token is invalid, expired, or malformed."""


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token.

    Parameters
    ----------
    token:
        The raw JWT string.

    Returns
    -------
    dict
        The decoded payload dictionary.

    Raises
    ------
    TokenError
        If the token is expired, has an invalid signature, revoked, or
        otherwise malformed.
    """
    import jwt as pyjwt

    try:
        payload: dict[str, Any] = pyjwt.decode(
            token,
            _get_secret_key(),
            algorithms=[_ALGORITHM],
            options={"require": ["sub", "exp", "iat", "type"]},
        )
    except pyjwt.ExpiredSignatureError:
        raise TokenError("Token has expired")
    except pyjwt.InvalidTokenError as exc:
        raise TokenError(f"Invalid token: {exc}")

    # Check blacklist
    jti = payload.get("jti")
    if jti and _blacklist.is_revoked(
        jti=jti,
        user_id=payload.get("sub"),
        issued_at=payload.get("iat"),
    ):
        raise TokenError("Token has been revoked")

    return payload
