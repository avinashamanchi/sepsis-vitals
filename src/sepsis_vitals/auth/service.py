"""
sepsis_vitals.auth.service
High-level authentication service: registration, login, password reset, and
email verification.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from sepsis_vitals.auth.jwt import (
    hash_password,
    is_locked_out,
    lockout_duration,
    verify_password,
)
from sepsis_vitals.auth.tokens import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from sepsis_vitals.db import User
from sepsis_vitals.security import compute_blind_index

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AuthServiceError(Exception):
    """Base exception for authentication service errors."""


class DuplicateEmailError(AuthServiceError):
    """Raised when registering with an email that already exists."""


class InvalidCredentialsError(AuthServiceError):
    """Raised on incorrect email or password."""


class AccountLockedError(AuthServiceError):
    """Raised when a login attempt is made against a locked account."""


class WeakPasswordError(AuthServiceError):
    """Raised when a password does not meet complexity requirements."""


class InvalidTokenError(AuthServiceError):
    """Raised when a verification or reset token is invalid."""


class BreakGlassError(AuthServiceError):
    """Raised when a break-glass emergency access attempt fails."""


# ---------------------------------------------------------------------------
# Password policy
# ---------------------------------------------------------------------------

_MIN_PASSWORD_LENGTH = 8
_PASSWORD_UPPER_RE = re.compile(r"[A-Z]")
_PASSWORD_DIGIT_RE = re.compile(r"\d")


def _validate_password_strength(password: str) -> None:
    """Raise :class:`WeakPasswordError` if *password* is too weak.

    Requirements:
    - At least 8 characters.
    - At least one uppercase letter.
    - At least one digit.
    """
    errors: list[str] = []
    if len(password) < _MIN_PASSWORD_LENGTH:
        errors.append(f"at least {_MIN_PASSWORD_LENGTH} characters")
    if not _PASSWORD_UPPER_RE.search(password):
        errors.append("at least one uppercase letter")
    if not _PASSWORD_DIGIT_RE.search(password):
        errors.append("at least one digit")
    if errors:
        raise WeakPasswordError(
            "Password does not meet requirements: " + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# Login rate limiting (in-process; for horizontal scaling use Redis)
# ---------------------------------------------------------------------------

_login_attempts: dict[str, list[float]] = {}
_MAX_LOGIN_ATTEMPTS_PER_MINUTE = 5


def _check_login_rate_limit(email: str) -> None:
    """Raise :class:`AuthServiceError` if the email has exceeded the login
    attempt rate limit (5 attempts per 60-second window).
    """
    now = time.monotonic()
    window = 60.0
    attempts = _login_attempts.setdefault(email, [])
    # Discard attempts older than the window.
    _login_attempts[email] = [t for t in attempts if now - t < window]
    if len(_login_attempts[email]) >= _MAX_LOGIN_ATTEMPTS_PER_MINUTE:
        raise AuthServiceError(
            "Too many login attempts. Please wait before trying again."
        )
    _login_attempts[email].append(now)


# ---------------------------------------------------------------------------
# HMAC-based token helpers (password-reset / email-verification)
# ---------------------------------------------------------------------------

_TOKEN_SECRET: str | None = None


def _get_token_secret() -> str:
    """Return the HMAC key used for password-reset and email-verification
    tokens.  Falls back to SEPSIS_JWT_SECRET if SEPSIS_TOKEN_SECRET is unset.
    """
    global _TOKEN_SECRET  # noqa: PLW0603
    if _TOKEN_SECRET is None:
        _TOKEN_SECRET = os.environ.get(
            "SEPSIS_TOKEN_SECRET",
            os.environ.get("SEPSIS_JWT_SECRET", ""),
        )
        if not _TOKEN_SECRET:
            raise RuntimeError(
                "Neither SEPSIS_TOKEN_SECRET nor SEPSIS_JWT_SECRET is set."
            )
    return _TOKEN_SECRET


def _make_hmac_token(user_id: str, purpose: str, expires_seconds: int) -> str:
    """Build a compact HMAC token of the form ``user_id:expiry:signature``.

    The token is not stored in the database; it is self-validating via the
    HMAC signature.
    """
    expiry = int(time.time()) + expires_seconds
    message = f"{user_id}:{purpose}:{expiry}"
    sig = hmac.new(
        _get_token_secret().encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{user_id}:{expiry}:{sig}"


def _verify_hmac_token(token: str, purpose: str) -> str:
    """Validate an HMAC token and return the embedded ``user_id``.

    Raises :class:`InvalidTokenError` on any failure (bad format, expired,
    tampered).
    """
    parts = token.split(":")
    if len(parts) != 3:
        raise InvalidTokenError("Malformed token")

    user_id, expiry_str, provided_sig = parts

    try:
        expiry = int(expiry_str)
    except ValueError:
        raise InvalidTokenError("Malformed token")

    if time.time() > expiry:
        raise InvalidTokenError("Token has expired")

    message = f"{user_id}:{purpose}:{expiry}"
    expected_sig = hmac.new(
        _get_token_secret().encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise InvalidTokenError("Invalid token signature")

    return user_id


# ---------------------------------------------------------------------------
# Token-pair helper
# ---------------------------------------------------------------------------


def _issue_tokens(user: User) -> dict[str, str]:
    """Return a dict with ``access_token`` and ``refresh_token`` for *user*."""
    return {
        "access_token": create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            org_id=user.site_id,
        ),
        "refresh_token": create_refresh_token(user_id=user.id),
        "token_type": "bearer",
    }


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


def register_user(
    email: str,
    password: str,
    role: str,
    org_id: str | None,
    db_session: Session,
) -> dict[str, Any]:
    """Register a new user account.

    Parameters
    ----------
    email:
        Unique email address.
    password:
        Plain-text password (validated for strength).
    role:
        One of ``nurse``, ``researcher``, ``system_admin``.
    org_id:
        Organisation / site identifier.
    db_session:
        Active SQLAlchemy session.

    Returns
    -------
    dict
        ``{"user": <User>, "access_token": ..., "refresh_token": ..., "token_type": ...}``

    Raises
    ------
    DuplicateEmailError
        If the email is already registered.
    WeakPasswordError
        If the password does not meet complexity requirements.
    """
    _validate_password_strength(password)

    email_hash = compute_blind_index(email)
    existing = db_session.query(User).filter(User.email_hash == email_hash).first()
    if existing is not None:
        raise DuplicateEmailError(f"Email {email!r} is already registered")

    valid_roles = {"nurse", "researcher", "system_admin"}
    if role not in valid_roles:
        raise AuthServiceError(
            f"Invalid role {role!r}. Must be one of {sorted(valid_roles)}."
        )

    user = User(
        email=email,
        email_hash=email_hash,
        password_hash=hash_password(password),
        role=role,
        site_id=org_id,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    tokens = _issue_tokens(user)
    return {"user": user, **tokens}


def login_user(
    email: str,
    password: str,
    db_session: Session,
) -> dict[str, Any]:
    """Authenticate a user by email and password.

    Parameters
    ----------
    email:
        The user's email address.
    password:
        The plain-text password.
    db_session:
        Active SQLAlchemy session.

    Returns
    -------
    dict
        ``{"user": <User>, "access_token": ..., "refresh_token": ..., "token_type": ...}``

    Raises
    ------
    AuthServiceError
        If rate-limited.
    AccountLockedError
        If the account is currently locked out.
    InvalidCredentialsError
        If the email does not exist or the password is wrong.
    """
    _check_login_rate_limit(email)

    email_hash = compute_blind_index(email)
    user = db_session.query(User).filter(User.email_hash == email_hash).first()
    if user is None:
        raise InvalidCredentialsError("Invalid email or password")

    # Check lockout before verifying the password.
    if is_locked_out(user.locked_until):
        raise AccountLockedError(
            "Account is temporarily locked due to repeated failed attempts. "
            "Please try again later."
        )

    if not verify_password(password, user.password_hash):
        # Increment failed attempts and set lockout window.
        user.failed_attempts = (user.failed_attempts or 0) + 1
        lock_secs = lockout_duration(user.failed_attempts)
        if lock_secs > 0:
            user.locked_until = datetime.now(timezone.utc) + timedelta(
                seconds=lock_secs
            )
        db_session.commit()
        raise InvalidCredentialsError("Invalid email or password")

    # Successful login — reset failure counters and update last_login.
    user.failed_attempts = 0
    user.locked_until = None
    user.last_login = datetime.now(timezone.utc)
    db_session.commit()
    db_session.refresh(user)

    return {"user": user, **_issue_tokens(user)}


def refresh_access_token(
    refresh_token: str,
    db_session: Session,
) -> dict[str, str]:
    """Exchange a valid refresh token for a new access token.

    Parameters
    ----------
    refresh_token:
        A JWT refresh token issued by :func:`login_user` or
        :func:`register_user`.
    db_session:
        Active SQLAlchemy session.

    Returns
    -------
    dict
        ``{"access_token": ..., "token_type": "bearer"}``

    Raises
    ------
    InvalidTokenError
        If the refresh token is expired, invalid, or the user no longer exists.
    """
    try:
        payload = decode_token(refresh_token)
    except TokenError as exc:
        raise InvalidTokenError(str(exc))

    if payload.get("type") != "refresh":
        raise InvalidTokenError("Token is not a refresh token")

    user_id = payload["sub"]
    user = db_session.query(User).filter(User.id == user_id).first()
    if user is None:
        raise InvalidTokenError("User no longer exists")

    access = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
        org_id=user.site_id,
    )
    return {"access_token": access, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

_PASSWORD_RESET_EXPIRY_SECONDS = 3600  # 1 hour


def request_password_reset(
    email: str,
    db_session: Session,
) -> str | None:
    """Generate a password-reset token for the given email.

    If the email is not registered the function returns ``None`` silently to
    avoid leaking account existence.

    Parameters
    ----------
    email:
        The user's email address.
    db_session:
        Active SQLAlchemy session.

    Returns
    -------
    str or None
        An HMAC-based reset token if the user exists, otherwise ``None``.
    """
    email_hash = compute_blind_index(email)
    user = db_session.query(User).filter(User.email_hash == email_hash).first()
    if user is None:
        return None
    return _make_hmac_token(
        user.id, "password_reset", _PASSWORD_RESET_EXPIRY_SECONDS
    )


def reset_password(
    token: str,
    new_password: str,
    db_session: Session,
) -> bool:
    """Reset the user's password using a valid reset token.

    Parameters
    ----------
    token:
        The HMAC reset token from :func:`request_password_reset`.
    new_password:
        The new plain-text password.
    db_session:
        Active SQLAlchemy session.

    Returns
    -------
    bool
        ``True`` on success.

    Raises
    ------
    InvalidTokenError
        If the token is expired, invalid, or the user no longer exists.
    WeakPasswordError
        If the new password does not meet complexity requirements.
    """
    user_id = _verify_hmac_token(token, "password_reset")
    _validate_password_strength(new_password)

    user = db_session.query(User).filter(User.id == user_id).first()
    if user is None:
        raise InvalidTokenError("User no longer exists")

    user.password_hash = hash_password(new_password)
    user.failed_attempts = 0
    user.locked_until = None
    db_session.commit()

    return True


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------

_EMAIL_VERIFY_EXPIRY_SECONDS = 86400  # 24 hours


def create_email_verification_token(user_id: str) -> str:
    """Generate an HMAC token for email verification.

    Parameters
    ----------
    user_id:
        The user's primary-key UUID.

    Returns
    -------
    str
        An HMAC-based verification token.
    """
    return _make_hmac_token(user_id, "email_verify", _EMAIL_VERIFY_EXPIRY_SECONDS)


def verify_email(
    token: str,
    db_session: Session,
) -> bool:
    """Verify a user's email address using a verification token.

    Since the current ``User`` model does not have an ``email_verified``
    column, this function validates the token and confirms the user exists.
    Callers can extend it once the column is added.

    Parameters
    ----------
    token:
        The HMAC verification token.
    db_session:
        Active SQLAlchemy session.

    Returns
    -------
    bool
        ``True`` if the token is valid and the user exists.

    Raises
    ------
    InvalidTokenError
        If the token is expired, invalid, or the user no longer exists.
    """
    user_id = _verify_hmac_token(token, "email_verify")

    user = db_session.query(User).filter(User.id == user_id).first()
    if user is None:
        raise InvalidTokenError("User no longer exists")

    # When an ``email_verified`` column is added to the User model, set it
    # here:  user.email_verified = True
    db_session.commit()

    return True


# ---------------------------------------------------------------------------
# Break-glass emergency access — HIPAA § 164.312(a)(2)(ii)
# ---------------------------------------------------------------------------

_BREAK_GLASS_TOKEN_HASH: str | None = None
_BREAK_GLASS_EXPIRY_MINUTES = 60  # 1 hour read-only access


def _get_break_glass_hash() -> str | None:
    """Return the SHA-256 hash of the sealed-envelope emergency token.

    Read from BREAK_GLASS_TOKEN_HASH env var. Returns None if not configured.
    """
    global _BREAK_GLASS_TOKEN_HASH  # noqa: PLW0603
    if _BREAK_GLASS_TOKEN_HASH is None:
        _BREAK_GLASS_TOKEN_HASH = os.environ.get("BREAK_GLASS_TOKEN_HASH", "")
    return _BREAK_GLASS_TOKEN_HASH or None


def break_glass_login(
    emergency_token: str,
    reason: str,
    ip_address: str,
) -> dict[str, Any]:
    """Authenticate via break-glass emergency override.

    HIPAA § 164.312(a)(2)(ii) requires an emergency access procedure.
    This implements a sealed-envelope token approach:

    1. A pre-shared emergency token is generated offline and its SHA-256
       hash is stored in BREAK_GLASS_TOKEN_HASH.
    2. The physical token is kept in a sealed envelope in the ward.
    3. When used, this function validates the token, issues a 1-hour
       read-only JWT, and fires compliance alerts.

    Parameters
    ----------
    emergency_token:
        The raw emergency token from the sealed envelope.
    reason:
        Free-text clinical justification (required, audited).
    ip_address:
        The requestor's IP address (for audit trail).

    Returns
    -------
    dict
        ``{"access_token": ..., "token_type": "bearer", "expires_minutes": 60}``

    Raises
    ------
    BreakGlassError
        If the token is invalid, not configured, or reason is empty.
    """
    import logging

    bg_logger = logging.getLogger("sepsis_vitals.break_glass")

    # Always log the attempt, even if it fails
    bg_logger.critical(
        "BREAK-GLASS ATTEMPT | ip=%s | reason=%s",
        ip_address,
        reason[:200],
    )

    expected_hash = _get_break_glass_hash()
    if expected_hash is None:
        bg_logger.error(
            "BREAK-GLASS DENIED — not configured | ip=%s", ip_address
        )
        raise BreakGlassError(
            "Emergency access is not configured. Contact system administrator."
        )

    if not reason or len(reason.strip()) < 10:
        bg_logger.warning(
            "BREAK-GLASS DENIED — insufficient reason | ip=%s | reason=%s",
            ip_address,
            reason[:200],
        )
        raise BreakGlassError(
            "A clinical justification of at least 10 characters is required."
        )

    # Validate token: SHA-256 hash comparison (constant-time)
    token_hash = hashlib.sha256(emergency_token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(token_hash, expected_hash):
        bg_logger.critical(
            "BREAK-GLASS DENIED — invalid token | ip=%s", ip_address
        )
        raise BreakGlassError("Invalid emergency token.")

    # Token is valid — issue a restricted 1-hour read-only JWT
    access_token = create_access_token(
        user_id="break-glass-emergency",
        email="emergency-override@system",
        role="nurse",  # read-only clinical role, not admin
        org_id=None,
        expires_minutes=_BREAK_GLASS_EXPIRY_MINUTES,
    )

    bg_logger.critical(
        "BREAK-GLASS GRANTED | ip=%s | reason=%s | expires_minutes=%d",
        ip_address,
        reason[:200],
        _BREAK_GLASS_EXPIRY_MINUTES,
    )

    # Fire compliance alerts (in production, this would send SMS/email
    # to the compliance officer and system administrators)
    _fire_break_glass_alerts(reason, ip_address)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_minutes": _BREAK_GLASS_EXPIRY_MINUTES,
        "role": "nurse",
        "warning": "Emergency read-only access granted. All actions are being audited. "
                   "This access will expire in 60 minutes.",
    }


def _fire_break_glass_alerts(reason: str, ip_address: str) -> None:
    """Fire high-priority compliance alerts for break-glass access.

    In production this would integrate with SMS (Twilio), email, and
    PagerDuty. For now it logs at CRITICAL level for log aggregator
    alerting (Grafana/PagerDuty alert rules on this log pattern).
    """
    import logging

    alert_logger = logging.getLogger("sepsis_vitals.compliance_alert")
    alert_logger.critical(
        "COMPLIANCE ALERT: Break-glass emergency access activated | "
        "ip=%s | reason=%s | "
        "ACTION REQUIRED: Verify clinical justification within 24 hours",
        ip_address,
        reason[:200],
    )
