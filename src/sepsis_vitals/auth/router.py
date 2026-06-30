"""
sepsis_vitals.auth.router
FastAPI router for authentication endpoints: registration, login, token
refresh, password reset, and user profile management.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from sepsis_vitals.auth.middleware import get_current_user
from sepsis_vitals.auth.service import (
    AccountLockedError,
    AuthServiceError,
    BreakGlassError,
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidTokenError,
    WeakPasswordError,
    break_glass_login,
    login_user,
    refresh_access_token,
    register_user,
    request_password_reset,
    reset_password,
    verify_email,
)
from sepsis_vitals.db import User, get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """Payload for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)
    role: str = Field(
        "nurse",
        pattern=r"^(nurse|researcher)$",
        description="User role: nurse or researcher",
    )
    org_id: Optional[str] = Field(
        None,
        max_length=32,
        description="Organisation / site identifier",
    )


class LoginRequest(BaseModel):
    """Payload for user login."""

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    """Payload to refresh an access token."""

    refresh_token: str


class PasswordResetRequestBody(BaseModel):
    """Payload to request a password-reset token."""

    email: EmailStr


class PasswordResetConfirmBody(BaseModel):
    """Payload to confirm a password reset."""

    token: str
    new_password: str = Field(..., min_length=12, max_length=128)


class EmailVerifyBody(BaseModel):
    """Payload to verify an email address."""

    token: str


class BreakGlassRequest(BaseModel):
    """Payload for HIPAA § 164.312(a)(2)(ii) emergency access."""

    emergency_token: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="The sealed-envelope emergency token",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Clinical justification for emergency access",
    )


class BreakGlassResponse(BaseModel):
    """Response from break-glass emergency access."""

    access_token: str
    token_type: str = "bearer"
    expires_minutes: int
    role: str
    warning: str


class ProfileUpdateRequest(BaseModel):
    """Payload for updating the current user's profile."""

    site_id: Optional[str] = Field(
        None, max_length=32, description="Organisation / site identifier"
    )


class TokenResponse(BaseModel):
    """Response containing JWT tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AccessTokenResponse(BaseModel):
    """Response containing a single access token (used on refresh)."""

    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Public representation of a user."""

    id: str
    email: str
    role: str
    org_id: Optional[str]
    mfa_enabled: bool
    created_at: Optional[str]
    last_login: Optional[str]


class RegisterResponse(BaseModel):
    """Response returned after successful registration."""

    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    """Generic success message."""

    detail: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_to_response(user: User) -> UserResponse:
    """Convert a SQLAlchemy ``User`` instance to a ``UserResponse``."""
    return UserResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        org_id=user.site_id,
        mfa_enabled=user.mfa_enabled,
        created_at=user.created_at.isoformat() if user.created_at else None,
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
def auth_register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
) -> RegisterResponse:
    """Create a new user account and return JWT tokens."""
    try:
        result = register_user(
            email=body.email,
            password=body.password,
            role=body.role,
            org_id=body.org_id,
            db_session=db,
        )
    except DuplicateEmailError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    except WeakPasswordError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except AuthServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    user: User = result["user"]
    return RegisterResponse(
        user=_user_to_response(user),
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        token_type=result["token_type"],
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
)
def auth_login(
    body: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Authenticate and return access + refresh tokens."""
    try:
        result = login_user(
            email=body.email,
            password=body.password,
            db_session=db,
        )
    except AccountLockedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is temporarily locked due to repeated failed attempts. "
            "Please try again later.",
        )
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthServiceError as exc:
        # Rate-limit exceeded.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        )

    return TokenResponse(
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        token_type=result["token_type"],
    )


class RefreshResponse(BaseModel):
    """Response containing rotated token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Refresh an access token",
)
def auth_refresh(
    body: RefreshRequest,
    db: Session = Depends(get_db),
) -> RefreshResponse:
    """Exchange a refresh token for a new access + refresh token pair.

    The old refresh token is revoked (single-use). Replaying a revoked
    refresh token will invalidate the user's entire token family.
    """
    try:
        result = refresh_access_token(
            refresh_token=body.refresh_token,
            db_session=db,
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return RefreshResponse(
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        token_type=result["token_type"],
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout and revoke tokens",
)
def auth_logout(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> MessageResponse:
    """Revoke the current access token and invalidate the session.

    Returns 200 even if the token was already revoked (idempotent).
    """
    from sepsis_vitals.auth.tokens import decode_token, get_blacklist

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            payload = decode_token(auth_header[7:])
            jti = payload.get("jti")
            if jti:
                # Revoke for remaining token lifetime
                ttl = max(0, payload.get("exp", 0) - int(__import__("time").time()))
                get_blacklist().revoke(jti, ttl_seconds=ttl or 900)
        except Exception:
            pass  # Token already expired/invalid — still return 200

    return MessageResponse(detail="Logged out successfully")


@router.post(
    "/password-reset/request",
    response_model=MessageResponse,
    summary="Request a password-reset token",
)
def auth_password_reset_request(
    body: PasswordResetRequestBody,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Generate a password-reset token.

    Always returns 200 regardless of whether the email exists, to prevent
    account enumeration.  In production the token would be sent via email.
    """
    token = request_password_reset(email=body.email, db_session=db)
    if token is not None:
        smtp_configured = bool(os.getenv("SMTP_HOST"))
        if smtp_configured:
            # TODO: send email via SMTP when configured
            logger.info("Password-reset token generated for %s", body.email)
        else:
            logger.warning(
                "Password-reset token generated for %s but SMTP is not configured — "
                "token cannot be delivered. Set SMTP_HOST to enable email delivery.",
                body.email,
            )
    return MessageResponse(
        detail="If an account with that email exists, a password-reset link has been sent."
    )


@router.post(
    "/password-reset/confirm",
    response_model=MessageResponse,
    summary="Reset password using a token",
)
def auth_password_reset_confirm(
    body: PasswordResetConfirmBody,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Reset the user's password using the reset token."""
    try:
        reset_password(
            token=body.token,
            new_password=body.new_password,
            db_session=db,
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
    except WeakPasswordError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return MessageResponse(detail="Password has been reset successfully")


@router.post(
    "/email/verify",
    response_model=MessageResponse,
    summary="Verify email address",
)
def auth_verify_email(
    body: EmailVerifyBody,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Verify the user's email address using a verification token."""
    try:
        verify_email(token=body.token, db_session=db)
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )

    return MessageResponse(detail="Email verified successfully")


@router.post(
    "/break-glass",
    response_model=BreakGlassResponse,
    summary="Emergency access — HIPAA § 164.312(a)(2)(ii)",
)
def auth_break_glass(
    body: BreakGlassRequest,
    request: Request,
) -> BreakGlassResponse:
    """Activate break-glass emergency access.

    This endpoint grants 1-hour read-only access for clinical emergencies
    when normal authentication is unavailable. All usage is heavily audited
    and triggers immediate compliance alerts.

    The emergency token is a pre-shared secret kept in a sealed envelope
    in the ward. Its SHA-256 hash is stored in BREAK_GLASS_TOKEN_HASH.
    """
    ip = request.client.host if request.client else "unknown"
    try:
        result = break_glass_login(
            emergency_token=body.emergency_token,
            reason=body.reason,
            ip_address=ip,
        )
    except BreakGlassError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )

    return BreakGlassResponse(
        access_token=result["access_token"],
        token_type=result["token_type"],
        expires_minutes=result["expires_minutes"],
        role=result["role"],
        warning=result["warning"],
    )


@router.post("/ping", summary="Session keep-alive")
async def session_ping(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Lightweight keep-alive endpoint that resets the session idle timer."""
    return {"status": "ok"}


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def auth_me(
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    user = db.query(User).filter(User.id == current_user["id"]).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )
    return _user_to_response(user)


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
)
def auth_update_me(
    body: ProfileUpdateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Update the current user's profile fields."""
    user = db.query(User).filter(User.id == current_user["id"]).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )

    if body.site_id is not None:
        user.site_id = body.site_id

    db.commit()
    db.refresh(user)

    return _user_to_response(user)
