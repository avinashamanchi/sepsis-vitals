"""
sepsis_vitals.auth.router
FastAPI router for authentication endpoints: registration, login, token
refresh, password reset, and user profile management.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from sepsis_vitals.auth.mailer import send_password_reset
from sepsis_vitals.auth.middleware import get_current_user, require_role
from sepsis_vitals.auth.service import (
    AccountLockedError,
    AuthServiceError,
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidTokenError,
    MFAEnrollmentRequired,
    MFARequiredError,
    WeakPasswordError,
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


class LoginRequest(BaseModel):
    """Payload for user login."""

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)
    otp: Optional[str] = Field(
        None, max_length=32, description="TOTP or recovery code (required once MFA is enabled)"
    )


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
    """Payload for updating the current user's profile.

    ``site_id`` is accepted only from ``system_admin`` users. Tenant
    assignment for everyone else goes through ``PUT /auth/users/{id}/site``.
    """

    site_id: Optional[str] = Field(
        None, max_length=32, description="Organisation / site identifier (admins only)"
    )


class SiteAssignmentRequest(BaseModel):
    """Payload for an administrator assigning a user to a site."""

    site_id: Optional[str] = Field(
        ..., max_length=32, description="Site identifier, or null to remove access"
    )


class TokenResponse(BaseModel):
    """Response containing JWT tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


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


def _enrollment_required(exc: MFAEnrollmentRequired) -> JSONResponse:
    """403 carrying a token that only the /auth/mfa endpoints accept."""
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": "mfa_enrollment_required", "enrollment_token": exc.enrollment_token},
    )


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
) -> RegisterResponse | JSONResponse:
    """Create a new user account and return JWT tokens."""
    if (
        os.getenv("SEPSIS_ENV", "development") == "production"
        and os.getenv("SEPSIS_ALLOW_SELF_REGISTRATION", "false").lower() != "true"
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration requires an administrator invitation.",
        )

    try:
        result = register_user(
            email=body.email,
            password=body.password,
            # Public input must never choose tenant or privilege. Approved
            # assignments happen through the administrative provisioning path.
            role="researcher",
            org_id=None,
            db_session=db,
        )
    except MFAEnrollmentRequired as exc:
        return _enrollment_required(exc)
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
) -> TokenResponse | JSONResponse:
    """Authenticate and return access + refresh tokens."""
    try:
        result = login_user(
            email=body.email,
            password=body.password,
            db_session=db,
            otp=body.otp,
        )
    except MFARequiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="mfa_required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except MFAEnrollmentRequired as exc:
        return _enrollment_required(exc)
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
        user=_user_to_response(result["user"]).model_dump(),
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
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Email a single-use password-reset link.

    Always returns the same 200 response, and sends mail after the response,
    so neither content nor timing reveals whether the account exists.
    """
    token = request_password_reset(email=body.email, db_session=db)
    if token is not None:
        background_tasks.add_task(send_password_reset, body.email, token)
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
    summary="Emergency access (disabled pending an approved policy)",
    status_code=status.HTTP_403_FORBIDDEN,
)
def auth_break_glass(body: BreakGlassRequest, request: Request) -> None:
    """Reject every emergency-access request.

    The previous implementation issued a token for a user that does not
    exist and with no site, so every endpoint rejected it: the feature looked
    available but could not work. Emergency access needs an approved policy
    (who may invoke it, for which site, with what scope, duration, review and
    notification) before it is rebuilt; see PROJECT_REVIEW.md (N18). The
    attempt is still logged for security monitoring, without the token.
    """
    ip = request.client.host if request.client else "unknown"
    logger.warning("BREAK-GLASS attempt rejected (feature disabled) | ip=%s", ip)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Emergency access is disabled: no approved emergency-access policy is "
            "configured. Contact your system administrator."
        ),
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
        # Users must not choose their own tenant: that would let anyone read
        # another hospital's patients by switching site_id.
        if user.role != "system_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Site assignment is managed by an administrator",
            )
        user.site_id = body.site_id

    db.commit()
    db.refresh(user)

    return _user_to_response(user)


@router.put(
    "/users/{user_id}/site",
    response_model=UserResponse,
    summary="Assign a user to a site (system_admin only)",
)
def auth_assign_site(
    user_id: str,
    body: SiteAssignmentRequest,
    admin: dict[str, Any] = Depends(require_role("system_admin")),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Set or clear the site a user is scoped to. Every change is logged."""
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    previous = target.site_id
    target.site_id = body.site_id
    db.commit()
    db.refresh(target)
    logger.warning(
        "AUDIT site_assignment admin=%s user=%s from=%s to=%s",
        admin.get("id"), target.id, previous, target.site_id,
    )
    return _user_to_response(target)


@router.put(
    "/users/{user_id}/mfa/reset",
    response_model=UserResponse,
    summary="Reset another user's MFA after device loss (system_admin only)",
)
def auth_reset_mfa(
    user_id: str,
    admin: dict[str, Any] = Depends(require_role("system_admin")),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Clear a user's MFA so they can re-enroll; ends all of their sessions.

    Administrators cannot reset their own MFA here: a second administrator
    (or the operator CLI, sepsis_vitals.auth.admin_cli) must do it, so one
    compromised admin session cannot strip its own second factor.
    """
    from sepsis_vitals.auth.mfa import reset_mfa

    if str(admin.get("id")) == str(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ask another administrator to reset your MFA",
        )
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    reset_mfa(target, db)
    logger.warning("AUDIT mfa_reset admin=%s user=%s", admin.get("id"), target.id)
    return _user_to_response(target)


from sepsis_vitals.auth import mfa as _mfa  # noqa: E402  (mounted under /auth/mfa)

router.include_router(_mfa.router)
