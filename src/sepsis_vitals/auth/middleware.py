"""
sepsis_vitals.auth.middleware
FastAPI dependencies for JWT authentication, role checking, and
organisation-level access control.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from sepsis_vitals.auth.tokens import TokenError, decode_token
from sepsis_vitals.db import User, get_db


# ---------------------------------------------------------------------------
# Current-user extraction
# ---------------------------------------------------------------------------


def _extract_bearer_token(request: Request) -> str:
    """Return the raw JWT from the ``Authorization: Bearer <token>`` header.

    Raises :class:`HTTPException` (401) if the header is missing or malformed.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Use 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_header[7:]


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """FastAPI dependency that extracts and validates the JWT from the request.

    Returns a dictionary with the user's ``id``, ``email``, ``role``, and
    ``org_id`` fields.  The underlying database row is also fetched to confirm
    the user still exists.

    Raises
    ------
    HTTPException (401)
        If the token is missing, expired, or invalid, or the user no longer
        exists.
    """
    token = _extract_bearer_token(request)

    try:
        payload = decode_token(token)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not an access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload["sub"]
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "org_id": user.site_id,
        "mfa_enabled": user.mfa_enabled,
    }


# ---------------------------------------------------------------------------
# Optional auth (returns None when no token is provided)
# ---------------------------------------------------------------------------


def get_optional_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[dict[str, Any]]:
    """FastAPI dependency that returns user info when a valid token is present,
    or ``None`` when no ``Authorization`` header is supplied.

    If a token *is* present but invalid, a 401 is still raised so that
    callers can distinguish "anonymous" from "bad credentials".
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    # A token was provided — delegate to the strict dependency.
    return get_current_user(request, db)


# ---------------------------------------------------------------------------
# Role requirement
# ---------------------------------------------------------------------------


def require_role(*roles: str) -> Callable[..., dict[str, Any]]:
    """Dependency factory that ensures the current user has one of the given
    roles.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(require_role("system_admin"))])
        async def admin_view(): ...

        # Or inject the user:
        @router.get("/staff")
        async def staff_view(user=Depends(require_role("nurse", "system_admin"))): ...

    Parameters
    ----------
    *roles:
        Allowed role names.

    Returns
    -------
    Callable
        A FastAPI dependency that returns the current user dict or raises 403.
    """
    allowed = set(roles)

    def _check(
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        if current_user["role"] not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Insufficient permissions. Required role: "
                    f"{', '.join(sorted(allowed))}"
                ),
            )
        return current_user

    return _check


# ---------------------------------------------------------------------------
# Organisation access
# ---------------------------------------------------------------------------


def require_org_access(org_id_param: str = "org_id") -> Callable[..., dict[str, Any]]:
    """Dependency factory that verifies the current user belongs to the
    organisation identified by the path/query parameter *org_id_param*.

    ``system_admin`` users bypass the check and can access any organisation.

    Usage::

        @router.get("/orgs/{org_id}/patients")
        async def list_patients(
            org_id: str,
            user=Depends(require_org_access("org_id")),
        ): ...

    Parameters
    ----------
    org_id_param:
        The name of the path or query parameter that contains the target
        organisation id.

    Returns
    -------
    Callable
        A FastAPI dependency that returns the current user dict or raises 403.
    """

    def _check(
        request: Request,
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        # System admins may access any organisation.
        if current_user["role"] == "system_admin":
            return current_user

        # Resolve the requested org_id from path parameters first, then query
        # parameters.
        target_org_id = request.path_params.get(org_id_param)
        if target_org_id is None:
            target_org_id = request.query_params.get(org_id_param)

        if target_org_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required parameter: {org_id_param}",
            )

        if current_user["org_id"] != target_org_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this organisation",
            )

        return current_user

    return _check
