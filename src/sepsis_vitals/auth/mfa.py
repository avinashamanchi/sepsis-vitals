"""
sepsis_vitals.auth.mfa
~~~~~~~~~~~~~~~~~~~~~~
TOTP multi-factor authentication: enrollment, confirmation, recovery codes,
and self-service disable.

Policy (see sepsis_vitals.auth.service):

* An enrolled user must present a TOTP or recovery code at every login.
* Roles listed in SEPSIS_MFA_REQUIRED_ROLES (empty by default) cannot obtain
  a session until they enroll; login hands them an enrollment-only token
  that these endpoints accept and every other endpoint rejects.
* Enabling, disabling or administrator reset revokes all existing sessions.
* Each TOTP time step is accepted once per secret (``users.totp_last_step``),
  including the code that confirms enrollment; recovery codes are single-use.
  See docs/mfa_replay_protection.md.

The TOTP secret is shown once at enrollment and stored encrypted
(EncryptedString); recovery codes are shown once and stored as keyed hashes.
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from sepsis_vitals.db import User, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mfa", tags=["mfa"])

RECOVERY_CODE_COUNT = 10
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I ambiguity


class CodeBody(BaseModel):
    code: str = Field(..., min_length=6, max_length=32)


class DisableBody(BaseModel):
    password: str = Field(..., min_length=1, max_length=128)
    code: str = Field(..., min_length=6, max_length=32)


def _principal(request: Request, db: Session = Depends(get_db)) -> User:
    """Accept a normal access token or an MFA enrollment token."""
    from sepsis_vitals.auth.tokens import TokenError, decode_token

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    try:
        payload = decode_token(header[7:])
    except TokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    if payload.get("type") not in ("access", "mfa_enroll"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token type")
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user


def _new_recovery_codes() -> list[str]:
    codes = []
    for _ in range(RECOVERY_CODE_COUNT):
        raw = "".join(secrets.choice(_ALPHABET) for _ in range(10))
        codes.append(f"{raw[:5]}-{raw[5:]}")
    return codes


def _revoke_sessions(user: User) -> None:
    from sepsis_vitals.auth.tokens import get_blacklist

    get_blacklist().revoke_all_for_user(str(user.id))


@router.post("/enroll", summary="Start MFA enrollment (returns the TOTP secret once)")
def mfa_enroll(user: User = Depends(_principal), db: Session = Depends(get_db)) -> dict[str, Any]:
    from sepsis_vitals.auth.jwt import generate_totp_secret, get_totp_uri
    from sepsis_vitals.auth.service import lock_user_row

    user = lock_user_row(db, user.id)
    if user.mfa_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "MFA is already enabled; disable it first")
    secret = generate_totp_secret()
    user.totp_secret = secret  # encrypted at rest; not active until confirmed
    user.totp_last_step = None  # replay state belongs to the previous secret
    db.commit()
    logger.info("MFA enrollment started for user %s", user.id)
    return {
        "secret": secret,
        "otpauth_uri": get_totp_uri(secret, user.email),
        "detail": "Add this to an authenticator app, then confirm with a current code.",
    }


@router.post("/confirm", summary="Confirm enrollment with a TOTP code; returns recovery codes once")
def mfa_confirm(
    body: CodeBody, user: User = Depends(_principal), db: Session = Depends(get_db)
) -> dict[str, Any]:
    from sepsis_vitals.auth.jwt import match_totp_step
    from sepsis_vitals.auth.service import consume_totp_step, lock_user_row
    from sepsis_vitals.security import compute_blind_index

    # Row lock: a concurrent enroll cannot swap the secret between this check
    # and enabling MFA.
    user = lock_user_row(db, user.id)
    if user.mfa_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "MFA is already enabled")
    if not user.totp_secret:
        raise HTTPException(status.HTTP_409_CONFLICT, "Start enrollment first")
    # The confirming code is consumed: it cannot then be reused to sign in.
    step = match_totp_step(user.totp_secret, body.code.strip())
    if step is None or not consume_totp_step(user, step, db):
        db.rollback()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid verification code")
    codes = _new_recovery_codes()
    user.mfa_recovery_hashes = json.dumps(
        [compute_blind_index(c.replace("-", "")) for c in codes]
    )
    user.mfa_enabled = True
    db.commit()
    _revoke_sessions(user)
    logger.warning("AUDIT mfa_enabled user=%s", user.id)
    return {
        "recovery_codes": codes,
        "detail": "MFA enabled. Store these single-use recovery codes safely, then sign in again.",
    }


@router.post("/disable", summary="Disable MFA (password and a current code required)")
def mfa_disable(
    body: DisableBody, user: User = Depends(_principal), db: Session = Depends(get_db)
) -> dict[str, str]:
    from sepsis_vitals.auth.jwt import verify_password
    from sepsis_vitals.auth.service import mfa_required_roles, verify_second_factor

    if user.role in mfa_required_roles():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "MFA is required for your role")
    if not user.mfa_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "MFA is not enabled")
    if not verify_password(body.password, user.password_hash) or not verify_second_factor(
        user, body.code, db
    ):
        db.rollback()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid password or verification code")
    reset_mfa(user, db)
    logger.warning("AUDIT mfa_disabled_by_user user=%s", user.id)
    return {"detail": "MFA disabled. Sign in again."}


def reset_mfa(user: User, db: Session) -> None:
    """Clear MFA state and end every session (self-disable, admin or operator reset)."""
    user.mfa_enabled = False
    user.totp_secret = None
    user.totp_last_step = None
    user.mfa_recovery_hashes = None
    db.commit()
    _revoke_sessions(user)
