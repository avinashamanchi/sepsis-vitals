"""
sepsis_vitals.billing.plans — SaaS pricing tiers for the sepsis-vitals platform.

Defines three per-bed pricing plans with monthly and annual billing options.
Stripe price IDs are loaded from environment variables so that test / live
environments can be switched without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Plan tier enum
# ---------------------------------------------------------------------------


class PlanTier(str, Enum):
    """Canonical plan tier identifiers."""

    COMMUNITY = "community"
    CLINICAL = "clinical"
    ENTERPRISE = "enterprise"


# ---------------------------------------------------------------------------
# Plan dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Immutable definition of a billing plan.

    Attributes
    ----------
    tier : PlanTier
        Canonical tier identifier.
    display_name : str
        Human-readable plan name shown in UI and invoices.
    monthly_price_cents : int
        Per-bed monthly price in USD cents.
    annual_price_cents : int
        Per-bed annual price in USD cents (reflects ~20 % discount).
    stripe_monthly_price_id : str
        Stripe ``Price`` object ID for monthly billing.
    stripe_annual_price_id : str
        Stripe ``Price`` object ID for annual billing.
    features : tuple[str, ...]
        Feature list surfaced on the pricing page.
    max_beds : Optional[int]
        Maximum beds allowed on this plan.  ``None`` means unlimited.
    """

    tier: PlanTier
    display_name: str
    monthly_price_cents: int
    annual_price_cents: int
    stripe_monthly_price_id: str
    stripe_annual_price_id: str
    features: tuple[str, ...] = field(default_factory=tuple)
    max_beds: Optional[int] = None


# ---------------------------------------------------------------------------
# Plan definitions
# ---------------------------------------------------------------------------

# Monthly prices are per-bed.
# Annual prices apply the ~20 % discount and are expressed *per-bed-per-year*.
#   Community: $15/mo  -> $144/yr  ($12/mo effective)
#   Clinical:  $32/mo  -> $307/yr  ($25.58/mo effective)
#   Enterprise:$50/mo  -> $480/yr  ($40/mo effective)

PLANS: tuple[Plan, ...] = (
    Plan(
        tier=PlanTier.COMMUNITY,
        display_name="Community",
        monthly_price_cents=1500,
        annual_price_cents=14400,
        stripe_monthly_price_id=os.getenv(
            "STRIPE_PRICE_COMMUNITY_MONTHLY", "price_community_monthly"
        ),
        stripe_annual_price_id=os.getenv(
            "STRIPE_PRICE_COMMUNITY_ANNUAL", "price_community_annual"
        ),
        features=(
            "Manual vitals entry",
            "Clinical sepsis scores (qSOFA, SIRS, NEWS2)",
            "Basic dashboards",
            "Email support",
        ),
        max_beds=50,
    ),
    Plan(
        tier=PlanTier.CLINICAL,
        display_name="Clinical",
        monthly_price_cents=3200,
        annual_price_cents=30700,
        stripe_monthly_price_id=os.getenv(
            "STRIPE_PRICE_CLINICAL_MONTHLY", "price_clinical_monthly"
        ),
        stripe_annual_price_id=os.getenv(
            "STRIPE_PRICE_CLINICAL_ANNUAL", "price_clinical_annual"
        ),
        features=(
            "Everything in Community",
            "ML sepsis predictions",
            "AI clinical copilot",
            "Real-time WebSocket alerts",
            "SHAP explanations",
            "Priority support",
        ),
        max_beds=500,
    ),
    Plan(
        tier=PlanTier.ENTERPRISE,
        display_name="Enterprise",
        monthly_price_cents=5000,
        annual_price_cents=48000,
        stripe_monthly_price_id=os.getenv(
            "STRIPE_PRICE_ENTERPRISE_MONTHLY", "price_enterprise_monthly"
        ),
        stripe_annual_price_id=os.getenv(
            "STRIPE_PRICE_ENTERPRISE_ANNUAL", "price_enterprise_annual"
        ),
        features=(
            "Everything in Clinical",
            "EHR / HL7 FHIR integration",
            "99.9 % uptime SLA",
            "Audit logs & compliance reporting",
            "SSO / SAML",
            "Dedicated support engineer",
        ),
        max_beds=None,
    ),
)

# Lookup maps built once at import time.
_BY_TIER: dict[PlanTier, Plan] = {p.tier: p for p in PLANS}
_BY_STRIPE_PRICE: dict[str, Plan] = {}
for _plan in PLANS:
    _BY_STRIPE_PRICE[_plan.stripe_monthly_price_id] = _plan
    _BY_STRIPE_PRICE[_plan.stripe_annual_price_id] = _plan


# ---------------------------------------------------------------------------
# Public lookup helpers
# ---------------------------------------------------------------------------


def get_plan_by_tier(tier: PlanTier | str) -> Plan:
    """Return the :class:`Plan` for the given *tier*.

    Parameters
    ----------
    tier : PlanTier or str
        A :class:`PlanTier` member or its string value (e.g. ``"clinical"``).

    Raises
    ------
    ValueError
        If *tier* does not match any known plan.
    """
    if isinstance(tier, str):
        try:
            tier = PlanTier(tier.lower())
        except ValueError:
            raise ValueError(f"Unknown plan tier: {tier!r}")
    plan = _BY_TIER.get(tier)
    if plan is None:
        raise ValueError(f"Unknown plan tier: {tier!r}")
    return plan


def get_plan_by_stripe_price(price_id: str) -> Plan:
    """Return the :class:`Plan` matching a Stripe price ID.

    Parameters
    ----------
    price_id : str
        A Stripe ``Price`` object ID.

    Raises
    ------
    ValueError
        If *price_id* does not match any known plan.
    """
    plan = _BY_STRIPE_PRICE.get(price_id)
    if plan is None:
        raise ValueError(f"No plan found for Stripe price ID: {price_id!r}")
    return plan


def is_annual_price(price_id: str) -> bool:
    """Return ``True`` if *price_id* corresponds to an annual billing cycle."""
    for plan in PLANS:
        if price_id == plan.stripe_annual_price_id:
            return True
    return False
