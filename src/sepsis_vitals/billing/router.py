"""
sepsis_vitals.billing.router — FastAPI billing endpoints.

Provides Stripe Checkout, Billing Portal, webhook processing, subscription
status, and bed-count management.  All mutating endpoints require
authentication.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from sepsis_vitals.billing.models import Invoice, Organization, Subscription
from sepsis_vitals.billing.plans import (
    PLANS,
    Plan,
    PlanTier,
    get_plan_by_tier,
    is_annual_price,
)
from sepsis_vitals.billing.stripe_service import StripeService
from sepsis_vitals.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

# ---------------------------------------------------------------------------
# Rate limiting — imported from api module, or create local instances
# ---------------------------------------------------------------------------

try:
    from sepsis_vitals.api import _billing_limiter, _webhook_limiter, _client_ip
except ImportError:
    from sepsis_vitals.security import RateLimiter
    _billing_limiter = RateLimiter(rate=1.0, burst=3)
    _webhook_limiter = RateLimiter(rate=5.0, burst=10)

    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


async def _check_billing_rate(request: Request) -> None:
    ip = _client_ip(request)
    if not _billing_limiter.allow(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Billing rate limit exceeded. Max 1 request/second.",
        )


async def _check_webhook_rate(request: Request) -> None:
    ip = _client_ip(request)
    if not _webhook_limiter.allow(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Webhook rate limit exceeded.",
        )


# ---------------------------------------------------------------------------
# Stripe webhook IP allowlist (production hardening)
# ---------------------------------------------------------------------------

_STRIPE_WEBHOOK_CIDRS: list[ipaddress.IPv4Network] | None = None

try:
    import ipaddress

    # Stripe's documented webhook IPs — https://docs.stripe.com/ips
    # Converted to /32 networks for CIDR-based validation
    _STRIPE_WEBHOOK_CIDRS = [
        ipaddress.ip_network("3.18.12.63/32", strict=False),
        ipaddress.ip_network("3.130.192.31/32", strict=False),
        ipaddress.ip_network("13.235.14.237/32", strict=False),
        ipaddress.ip_network("13.235.122.149/32", strict=False),
        ipaddress.ip_network("18.211.135.69/32", strict=False),
        ipaddress.ip_network("35.154.171.200/32", strict=False),
        ipaddress.ip_network("52.15.183.38/32", strict=False),
        ipaddress.ip_network("54.88.130.119/32", strict=False),
        ipaddress.ip_network("54.88.130.237/32", strict=False),
        ipaddress.ip_network("54.187.174.169/32", strict=False),
        ipaddress.ip_network("54.187.205.235/32", strict=False),
        ipaddress.ip_network("54.187.216.72/32", strict=False),
    ]
except ImportError:
    pass

# In-memory deduplication cache for webhook events
_processed_webhook_events: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# Auth dependency — reuses the pattern from sepsis_vitals.api
# ---------------------------------------------------------------------------


async def _require_auth(request: Request) -> Dict[str, str]:
    """Extract and verify the authenticated user.

    Mirrors ``verify_auth`` from :mod:`sepsis_vitals.api` so the billing
    router can be mounted independently during tests.
    """
    from sepsis_vitals.api import verify_auth

    user = await verify_auth(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return user


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    """Request body for creating a Stripe Checkout session."""

    org_id: str = Field(..., description="Organization UUID")
    plan_tier: str = Field(..., description="Plan tier: community, clinical, enterprise")
    bed_count: int = Field(..., ge=1, description="Number of beds to subscribe for")
    annual: bool = Field(False, description="Use annual billing (20%% discount)")
    success_url: str = Field(..., description="Redirect URL on successful checkout")
    cancel_url: str = Field(..., description="Redirect URL if checkout is cancelled")


class CheckoutResponse(BaseModel):
    """Stripe Checkout session URL."""

    checkout_url: str
    session_id: str


class PortalRequest(BaseModel):
    """Request body for creating a Stripe Billing Portal session."""

    org_id: str = Field(..., description="Organization UUID")
    return_url: str = Field(..., description="URL to return to after portal")


class PortalResponse(BaseModel):
    """Stripe Billing Portal URL."""

    portal_url: str


class UpdateBedsRequest(BaseModel):
    """Request body for updating bed count."""

    org_id: str = Field(..., description="Organization UUID")
    bed_count: int = Field(..., ge=1, description="New bed count")


class SubscriptionDetail(BaseModel):
    """Subscription status returned to the client."""

    subscription_id: str
    stripe_subscription_id: str
    plan_tier: str
    plan_name: str
    price_id: str
    annual: bool
    bed_count: int
    status: str
    current_period_start: Optional[str] = None
    current_period_end: Optional[str] = None
    cancel_at: Optional[str] = None


class SubscriptionStatusResponse(BaseModel):
    """Full subscription status for an organization."""

    org_id: str
    org_name: str
    is_active: bool
    plan_tier: str
    bed_count: int
    subscription: Optional[SubscriptionDetail] = None
    recent_invoices: List[Dict[str, Any]] = Field(default_factory=list)


class UpdateBedsResponse(BaseModel):
    """Confirmation of a bed-count update."""

    org_id: str
    old_bed_count: int
    new_bed_count: int
    plan_tier: str


class PlanInfo(BaseModel):
    """Plan details for the pricing page."""

    tier: str
    display_name: str
    monthly_price_cents: int
    annual_price_cents: int
    features: List[str]
    max_beds: Optional[int] = None


class WebhookResponse(BaseModel):
    """Response for webhook processing."""

    status: str
    type: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_org(db: Session, org_id: str) -> Organization:
    """Fetch an organization by ID or raise 404."""
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found.",
        )
    return org


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/plans", response_model=List[PlanInfo])
async def list_plans() -> List[PlanInfo]:
    """Return available pricing plans (public, no auth required)."""
    return [
        PlanInfo(
            tier=p.tier.value,
            display_name=p.display_name,
            monthly_price_cents=p.monthly_price_cents,
            annual_price_cents=p.annual_price_cents,
            features=list(p.features),
            max_beds=p.max_beds,
        )
        for p in PLANS
    ]


@router.post(
    "/checkout",
    response_model=CheckoutResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_check_billing_rate)],
)
async def create_checkout(
    body: CheckoutRequest,
    user: Dict[str, str] = Depends(_require_auth),
    db: Session = Depends(get_db),
) -> CheckoutResponse:
    """Create a Stripe Checkout session for a new subscription."""
    org = _get_org(db, body.org_id)

    try:
        plan = get_plan_by_tier(body.plan_tier)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown plan tier: {body.plan_tier}",
        )

    if plan.max_beds is not None and body.bed_count > plan.max_beds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"The {plan.display_name} plan supports a maximum of "
                f"{plan.max_beds} beds. Requested: {body.bed_count}."
            ),
        )

    # Create Stripe customer if not yet associated
    if not org.stripe_customer_id:
        customer = StripeService.create_customer(
            org_name=org.name, email=user.get("email", f"{org.slug}@sepsis-vitals.io")
        )
        org.stripe_customer_id = customer.id
        db.commit()

    price_id = plan.stripe_annual_price_id if body.annual else plan.stripe_monthly_price_id

    session = StripeService.create_checkout_session(
        customer_id=org.stripe_customer_id,
        price_id=price_id,
        bed_count=body.bed_count,
        success_url=body.success_url,
        cancel_url=body.cancel_url,
    )

    return CheckoutResponse(checkout_url=session.url, session_id=session.id)


@router.post(
    "/portal",
    response_model=PortalResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_check_billing_rate)],
)
async def create_portal(
    body: PortalRequest,
    user: Dict[str, str] = Depends(_require_auth),
    db: Session = Depends(get_db),
) -> PortalResponse:
    """Create a Stripe Billing Portal session for self-service management."""
    org = _get_org(db, body.org_id)

    if not org.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization has no Stripe customer. Subscribe first.",
        )

    url = StripeService.create_portal_session(
        customer_id=org.stripe_customer_id, return_url=body.return_url
    )
    return PortalResponse(portal_url=url)


@router.post(
    "/webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_check_webhook_rate)],
)
async def stripe_webhook(
    request: Request, db: Session = Depends(get_db)
) -> WebhookResponse:
    """Receive and process Stripe webhook events.

    This endpoint does **not** require bearer-token auth; instead it verifies
    the ``Stripe-Signature`` header against the webhook endpoint secret.
    Includes IP allowlist check and event deduplication.
    """
    # IP allowlist in production
    if _STRIPE_WEBHOOK_CIDRS and os.getenv("SEPSIS_ENV") == "production":
        client_ip = _client_ip(request)
        try:
            addr = ipaddress.ip_address(client_ip)
            if not any(addr in net for net in _STRIPE_WEBHOOK_CIDRS):
                logger.warning("Webhook from non-Stripe IP: %s", client_ip)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Webhook origin not in allowlist.",
                )
        except ValueError:
            logger.warning("Invalid IP format in webhook request: %s", client_ip)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid client IP format.",
            )

    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")
    if not sig_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header.",
        )

    # Event deduplication — skip already-processed events
    import json as _json
    try:
        event_data = _json.loads(payload)
        event_id = event_data.get("id", "")
        if event_id and event_id in _processed_webhook_events:
            logger.info("Skipping duplicate webhook event: %s", event_id)
            return WebhookResponse(status="duplicate", type=event_data.get("type", "unknown"))
    except (ValueError, KeyError):
        pass

    try:
        result = StripeService.handle_webhook(payload, sig_header, db)
    except ValueError as exc:
        logger.warning("Webhook signature verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature.",
        )

    # Record processed event
    if event_id:
        _processed_webhook_events[event_id] = time.time()
        # Evict old entries (keep last 1000)
        if len(_processed_webhook_events) > 1000:
            oldest = sorted(_processed_webhook_events, key=_processed_webhook_events.get)[:500]  # type: ignore[arg-type]
            for k in oldest:
                _processed_webhook_events.pop(k, None)

    return WebhookResponse(**result)


@router.get(
    "/subscription",
    response_model=SubscriptionStatusResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_check_billing_rate)],
)
async def get_subscription(
    org_id: str,
    user: Dict[str, str] = Depends(_require_auth),
    db: Session = Depends(get_db),
) -> SubscriptionStatusResponse:
    """Return the current subscription status for an organization."""
    org = _get_org(db, org_id)

    # Latest active subscription
    sub = (
        db.query(Subscription)
        .filter(
            Subscription.org_id == org.id,
            Subscription.status.in_(("active", "trialing", "past_due")),
        )
        .order_by(Subscription.created_at.desc())
        .first()
    )

    sub_detail: Optional[SubscriptionDetail] = None
    if sub is not None:
        try:
            plan = get_plan_by_tier(org.plan_tier)
            plan_name = plan.display_name
        except ValueError:
            plan_name = org.plan_tier.title()

        sub_detail = SubscriptionDetail(
            subscription_id=sub.id,
            stripe_subscription_id=sub.stripe_subscription_id,
            plan_tier=org.plan_tier,
            plan_name=plan_name,
            price_id=sub.stripe_price_id,
            annual=is_annual_price(sub.stripe_price_id),
            bed_count=org.bed_count,
            status=sub.status,
            current_period_start=(
                sub.current_period_start.isoformat()
                if sub.current_period_start
                else None
            ),
            current_period_end=(
                sub.current_period_end.isoformat()
                if sub.current_period_end
                else None
            ),
            cancel_at=(
                sub.cancel_at.isoformat() if sub.cancel_at else None
            ),
        )

    # Recent invoices (last 10)
    recent = (
        db.query(Invoice)
        .filter(Invoice.org_id == org.id)
        .order_by(Invoice.created_at.desc())
        .limit(10)
        .all()
    )
    invoices_out = [
        {
            "invoice_id": inv.id,
            "stripe_invoice_id": inv.stripe_invoice_id,
            "amount_cents": inv.amount_cents,
            "currency": inv.currency,
            "status": inv.status,
            "paid_at": inv.paid_at.isoformat() if inv.paid_at else None,
            "pdf_url": inv.pdf_url,
        }
        for inv in recent
    ]

    return SubscriptionStatusResponse(
        org_id=org.id,
        org_name=org.name,
        is_active=org.is_active,
        plan_tier=org.plan_tier,
        bed_count=org.bed_count,
        subscription=sub_detail,
        recent_invoices=invoices_out,
    )


@router.put(
    "/beds",
    response_model=UpdateBedsResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_check_billing_rate)],
)
async def update_beds(
    body: UpdateBedsRequest,
    user: Dict[str, str] = Depends(_require_auth),
    db: Session = Depends(get_db),
) -> UpdateBedsResponse:
    """Update the bed count on an active subscription.

    Adjusts the Stripe subscription quantity and updates the local record.
    """
    org = _get_org(db, body.org_id)

    # Validate against plan max
    try:
        plan = get_plan_by_tier(org.plan_tier)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown plan tier: {org.plan_tier}",
        )

    if plan.max_beds is not None and body.bed_count > plan.max_beds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"The {plan.display_name} plan supports a maximum of "
                f"{plan.max_beds} beds. Requested: {body.bed_count}."
            ),
        )

    # Find active subscription
    sub = (
        db.query(Subscription)
        .filter(
            Subscription.org_id == org.id,
            Subscription.status == "active",
        )
        .order_by(Subscription.created_at.desc())
        .first()
    )
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription found for this organization.",
        )

    old_count = org.bed_count

    # Update on Stripe
    StripeService.update_subscription_beds(
        subscription_id=sub.stripe_subscription_id,
        new_bed_count=body.bed_count,
    )

    # Update locally
    org.bed_count = body.bed_count
    db.commit()

    logger.info(
        "Beds updated for org %s: %d -> %d", org.id, old_count, body.bed_count
    )

    return UpdateBedsResponse(
        org_id=org.id,
        old_bed_count=old_count,
        new_bed_count=body.bed_count,
        plan_tier=org.plan_tier,
    )
