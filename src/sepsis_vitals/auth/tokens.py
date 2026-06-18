"""
sepsis_vitals.auth.tokens
JWT access and refresh token creation and verification.
"""

from __future__ import annotations

import os
import time
from typing import Any


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
# Token creation
# ---------------------------------------------------------------------------


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    org_id: str | None,
    expires_minutes: int = 30,
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
        Token lifetime in minutes.  Defaults to 30.

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
        "iat": now,
        "exp": now + expires_minutes * 60,
    }
    return pyjwt.encode(payload, _get_secret_key(), algorithm=_ALGORITHM)


def create_refresh_token(
    user_id: str,
    expires_days: int = 7,
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
        If the token is expired, has an invalid signature, or is otherwise
        malformed.
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

    return payload
