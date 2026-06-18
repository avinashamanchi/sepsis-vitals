"""
sepsis_vitals.auth.router
FastAPI router for authentication endpoints: registration, login, token
refresh, password reset, and user profile management.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from sepsis_vitals.auth.middleware import get_current_user
from sepsis_vitals.auth.service import (
    AccountLockedError,
    AuthServiceError,
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidTokenError,
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
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(
        ...,
        pattern=r"^(nurse|researcher|system_admin)$",
        description="User role: nurse, researcher, or system_admin",
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
    new_password: str = Field(..., min_length=8, max_length=128)


class EmailVerifyBody(BaseModel):
    """Payload to verify an email address."""

    token: str


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


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
    summary="Refresh an access token",
)
def auth_refresh(
    body: RefreshRequest,
    db: Session = Depends(get_db),
) -> AccessTokenResponse:
    """Exchange a refresh token for a new access token."""
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

    return AccessTokenResponse(
        access_token=result["access_token"],
        token_type=result["token_type"],
    )


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
        # In production, send the token via email.
        logger.info("Password-reset token generated for %s", body.email)
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
