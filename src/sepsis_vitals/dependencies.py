"""
sepsis_vitals.dependencies — request dependencies shared by every router.

Authentication, per-IP rate limits, client-IP resolution and the patient
site check live here, not in ``sepsis_vitals.api``. The routers (auth,
patients, alerts, FHIR, billing, bundles and ``sepsis_vitals.routes``) import
them from this module, so none of them imports the application module and
import order no longer matters. ``sepsis_vitals.api`` re-exports every name
here: ``from sepsis_vitals.api import verify_auth`` and
``app.dependency_overrides[api.check_rate_limit]`` refer to the same function
objects.

Patch ``sepsis_vitals.dependencies._auth_enabled`` (not the api module) to
switch authentication off in tests.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from typing import Any, Dict

from fastapi import Depends, HTTPException, Request

from sepsis_vitals.security import RateLimiter

# Same logger name as before the split, so log routing and filters still apply.
logger = logging.getLogger("sepsis_vitals.api")

_is_production = os.getenv("SEPSIS_ENV", "development") == "production"

# ---------------------------------------------------------------------------
# Trusted proxy configuration for X-Forwarded-For validation
# ---------------------------------------------------------------------------

_TRUSTED_PROXIES: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
_raw_trusted = os.getenv("TRUSTED_PROXIES", "")
if _raw_trusted:
    for cidr in _raw_trusted.split(","):
        cidr = cidr.strip()
        if cidr:
            try:
                _TRUSTED_PROXIES.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# 10 req/s burst 20 for general API, 2 req/s burst 5 for expensive ML predict
_api_limiter = RateLimiter(rate=10.0, burst=20)
_ml_limiter = RateLimiter(rate=2.0, burst=5)
_auth_limiter = RateLimiter(rate=3.0, burst=10)   # Auth: 3/s burst 10 (brute-force protection)
_copilot_limiter = RateLimiter(rate=0.5, burst=3)
_billing_limiter = RateLimiter(rate=1.0, burst=3)  # Stripe mutations: 1/s
_webhook_limiter = RateLimiter(rate=5.0, burst=10)  # Stripe webhooks: 5/s


def _client_ip(request: Request) -> str:
    """Extract the real client IP, only trusting X-Forwarded-For when the
    immediate client is in ``TRUSTED_PROXIES``."""
    direct_ip = request.client.host if request.client else "unknown"
    if direct_ip == "unknown":
        return direct_ip

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and _TRUSTED_PROXIES:
        try:
            addr = ipaddress.ip_address(direct_ip)
            if any(addr in net for net in _TRUSTED_PROXIES):
                return forwarded.split(",")[0].strip()
        except ValueError:
            pass
    elif forwarded and not _TRUSTED_PROXIES:
        # No trusted proxies configured — fall back to direct IP
        return direct_ip

    return direct_ip


async def check_rate_limit(request: Request) -> None:
    """General API rate limit — dependency for most endpoints."""
    ip = _client_ip(request)
    if not _api_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again shortly.",
        )


async def check_ml_rate_limit(request: Request) -> None:
    """ML prediction rate limit — more restrictive."""
    ip = _client_ip(request)
    if not _ml_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="ML prediction rate limit exceeded. Max 2 requests/second.",
        )


async def check_auth_rate_limit(request: Request) -> None:
    """Auth endpoint rate limit — brute-force protection."""
    ip = _client_ip(request)
    if not _auth_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="Too many auth requests. Try again shortly.",
        )


# ---------------------------------------------------------------------------
# Authentication (JWT with short-lived access tokens + RBAC)
# ---------------------------------------------------------------------------

_auth_enabled = os.getenv("SEPSIS_AUTH_ENABLED", "true").lower() == "true"
if _is_production and not _auth_enabled:
    logger.warning(
        "SEPSIS_AUTH_ENABLED=false is ignored in production — forcing auth on"
    )
    _auth_enabled = True


def _anonymous_user() -> Dict[str, Any]:
    """Return a synthetic admin user dict when auth is disabled (dev only)."""
    return {"id": "anonymous", "email": "dev@localhost", "role": "system_admin", "org_id": None}


async def verify_auth(request: Request) -> Dict[str, Any]:
    """Verify JWT access token from Authorization header.

    Uses the real JWT middleware (short-lived HS256 tokens issued by
    /auth/login) when auth is enabled.  Falls back to an anonymous
    system_admin identity when SEPSIS_AUTH_ENABLED=false (development only).
    """
    if not _auth_enabled:
        return _anonymous_user()

    try:
        from sepsis_vitals.auth.middleware import get_current_user
        from sepsis_vitals.db import get_db

        def _resolve() -> Dict[str, Any]:
            # Resolve the DB session dependency manually since we're not in
            # a standard Depends() chain for this legacy shim.
            db_gen = get_db()
            db = next(db_gen)
            try:
                return get_current_user(request, db)
            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass

        # The user lookup is a blocking database query; keep it off the event
        # loop, which also serves WebSockets and every other request.
        return await asyncio.to_thread(_resolve)
    except ImportError:
        if _is_production:
            logger.critical("Auth middleware not available in production — rejecting request")
            raise HTTPException(
                status_code=500,
                detail="Authentication service unavailable",
            )
        logger.warning("Auth middleware not available — falling back to anonymous (dev only)")
        return _anonymous_user()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auth verification failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_patient_org(patient_id: str, user: Dict[str, Any], db) -> None:
    """Verify that the patient belongs to the requesting user's org.

    Delegates to :mod:`sepsis_vitals.auth.scope`: only ``system_admin``
    (including the anonymous dev identity when auth is disabled) is
    unscoped. Every other user must have a site assignment that matches the
    patient's ``site_id``; otherwise HTTP 404 is raised so existence at
    another site is not disclosed.
    """
    from sepsis_vitals.auth.scope import is_unscoped, load_patient_for_user

    if is_unscoped(user):
        return  # system_admin, incl. the auth-disabled dev identity
    load_patient_for_user(patient_id, user, db)


def require_role_dep(*roles: str):
    """Dependency factory that ensures the current user has one of the given roles."""
    allowed = set(roles)

    async def _check(user: Dict = Depends(verify_auth)) -> Dict[str, Any]:
        if user.get("role") not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {', '.join(sorted(allowed))}",
            )
        return user

    return _check


async def _verify_patient_org_async(patient_id: str, user: Dict[str, Any]) -> None:
    """Run :func:`verify_patient_org` in a worker thread with its own session."""
    def _check_org():
        from sepsis_vitals.db import SessionLocal
        db = SessionLocal()
        try:
            verify_patient_org(patient_id, user, db)
        finally:
            db.close()
    await asyncio.to_thread(_check_org)
