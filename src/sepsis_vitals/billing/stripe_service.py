"""
sepsis_vitals.billing.stripe_service — Stripe integration layer.

Encapsulates all Stripe API interactions so the rest of the application never
imports ``stripe`` directly.  API keys are loaded from environment variables
(``STRIPE_SECRET_KEY``, ``STRIPE_WEBHOOK_SECRET``).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from sepsis_vitals.billing.models import Invoice, Organization, Subscription
from sepsis_vitals.billing.plans import get_plan_by_stripe_price

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy Stripe import
# ---------------------------------------------------------------------------

_stripe = None


def _get_stripe():
    """Return the ``stripe`` module, configured with the secret key."""
    global _stripe
    if _stripe is None:
        import stripe

        api_key = os.environ.get("STRIPE_SECRET_KEY")
        if not api_key:
            raise EnvironmentError(
                "STRIPE_SECRET_KEY environment variable is not set"
            )
        stripe.api_key = api_key
        _stripe = stripe
    return _stripe


def _get_webhook_secret() -> str:
    """Return the Stripe webhook endpoint secret."""
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not secret:
        raise EnvironmentError(
            "STRIPE_WEBHOOK_SECRET environment variable is not set"
        )
    return secret


# ---------------------------------------------------------------------------
# Stripe service class
# ---------------------------------------------------------------------------


class StripeService:
    """Thin facade over the Stripe Python SDK.

    Every public method corresponds to a single Stripe API operation and
    returns SDK objects or simple values rather than leaking raw HTTP
    responses.
    """

    # -- Customer operations ------------------------------------------------

    @staticmethod
    def create_customer(org_name: str, email: str) -> Any:
        """Create a Stripe ``Customer`` for the given organization.

        Parameters
        ----------
        org_name : str
            Organization display name stored on the customer record.
        email : str
            Billing contact email.

        Returns
        -------
        stripe.Customer
        """
        s = _get_stripe()
        customer = s.Customer.create(
            name=org_name,
            email=email,
            metadata={"platform": "sepsis-vitals"},
        )
        logger.info("Created Stripe customer %s for %s", customer.id, org_name)
        return customer

    # -- Subscription operations --------------------------------------------

    @staticmethod
    def create_subscription(
        customer_id: str, price_id: str, bed_count: int
    ) -> Any:
        """Create a metered Stripe ``Subscription``.

        The subscription uses a ``quantity`` equal to *bed_count* so Stripe
        bills ``price * quantity`` each period.

        Parameters
        ----------
        customer_id : str
            Stripe customer ID.
        price_id : str
            Stripe price ID for the chosen plan and cadence.
        bed_count : int
            Number of beds to bill for.

        Returns
        -------
        stripe.Subscription
        """
        if bed_count < 1:
            raise ValueError("bed_count must be at least 1")

        s = _get_stripe()
        subscription = s.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id, "quantity": bed_count}],
            payment_behavior="default_incomplete",
            expand=["latest_invoice.payment_intent"],
        )
        logger.info(
            "Created subscription %s for customer %s (%d beds)",
            subscription.id,
            customer_id,
            bed_count,
        )
        return subscription

    @staticmethod
    def update_subscription_beds(
        subscription_id: str, new_bed_count: int
    ) -> Any:
        """Update the bed quantity on an existing subscription.

        Stripe prorates the change automatically based on the subscription's
        proration behaviour.

        Parameters
        ----------
        subscription_id : str
            Stripe subscription ID.
        new_bed_count : int
            Updated number of beds.

        Returns
        -------
        stripe.Subscription
        """
        if new_bed_count < 1:
            raise ValueError("new_bed_count must be at least 1")

        s = _get_stripe()
        subscription = s.Subscription.retrieve(subscription_id)
        item_id = subscription["items"]["data"][0]["id"]
        updated = s.Subscription.modify(
            subscription_id,
            items=[{"id": item_id, "quantity": new_bed_count}],
            proration_behavior="create_prorations",
        )
        logger.info(
            "Updated subscription %s to %d beds", subscription_id, new_bed_count
        )
        return updated

    @staticmethod
    def cancel_subscription(
        subscription_id: str, *, at_period_end: bool = True
    ) -> Any:
        """Cancel a Stripe subscription.

        Parameters
        ----------
        subscription_id : str
            Stripe subscription ID.
        at_period_end : bool
            If ``True`` (default), the subscription remains active until the
            current billing period ends.  If ``False``, cancel immediately.

        Returns
        -------
        stripe.Subscription
        """
        s = _get_stripe()
        if at_period_end:
            updated = s.Subscription.modify(
                subscription_id, cancel_at_period_end=True
            )
        else:
            updated = s.Subscription.cancel(subscription_id)
        logger.info(
            "Cancelled subscription %s (at_period_end=%s)",
            subscription_id,
            at_period_end,
        )
        return updated

    # -- Portal -------------------------------------------------------------

    @staticmethod
    def create_portal_session(customer_id: str, return_url: str) -> str:
        """Create a Stripe Billing Portal session.

        Parameters
        ----------
        customer_id : str
            Stripe customer ID.
        return_url : str
            URL the customer is redirected to after leaving the portal.

        Returns
        -------
        str
            The portal session URL.
        """
        s = _get_stripe()
        session = s.billing_portal.Session.create(
            customer=customer_id, return_url=return_url
        )
        return session.url

    # -- Checkout -----------------------------------------------------------

    @staticmethod
    def create_checkout_session(
        customer_id: str,
        price_id: str,
        bed_count: int,
        success_url: str,
        cancel_url: str,
    ) -> Any:
        """Create a Stripe Checkout session for a new subscription.

        Parameters
        ----------
        customer_id : str
            Stripe customer ID.
        price_id : str
            Stripe price ID.
        bed_count : int
            Number of beds.
        success_url : str
            Redirect on successful payment.
        cancel_url : str
            Redirect on cancelled checkout.

        Returns
        -------
        stripe.checkout.Session
        """
        if bed_count < 1:
            raise ValueError("bed_count must be at least 1")

        s = _get_stripe()
        session = s.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": bed_count}],
            success_url=success_url,
            cancel_url=cancel_url,
        )
        logger.info(
            "Created checkout session %s for customer %s", session.id, customer_id
        )
        return session

    # -- Webhook handling ---------------------------------------------------

    @staticmethod
    def construct_webhook_event(payload: bytes, sig_header: str) -> Any:
        """Verify and parse a Stripe webhook event.

        Parameters
        ----------
        payload : bytes
            Raw request body.
        sig_header : str
            Value of the ``Stripe-Signature`` header.

        Returns
        -------
        stripe.Event

        Raises
        ------
        ValueError
            If the signature is invalid.
        """
        s = _get_stripe()
        secret = _get_webhook_secret()
        event = s.Webhook.construct_event(payload, sig_header, secret)
        return event

    @classmethod
    def handle_webhook(cls, payload: bytes, sig_header: str, db: Session) -> dict[str, str]:
        """Process an inbound Stripe webhook event.

        Dispatches to internal handlers based on event type and persists
        relevant state changes to the database.

        Parameters
        ----------
        payload : bytes
            Raw request body.
        sig_header : str
            ``Stripe-Signature`` header value.
        db : Session
            Active SQLAlchemy session.

        Returns
        -------
        dict
            ``{"status": "handled", "type": "<event_type>"}`` on success or
            ``{"status": "ignored", ...}`` for unhandled event types.
        """
        event = cls.construct_webhook_event(payload, sig_header)
        event_type: str = event["type"]
        data_object: dict = event["data"]["object"]

        handler = _EVENT_HANDLERS.get(event_type)
        if handler is None:
            logger.debug("Ignoring unhandled webhook event type: %s", event_type)
            return {"status": "ignored", "type": event_type}

        handler(data_object, db)
        db.commit()
        logger.info("Processed webhook event %s (%s)", event["id"], event_type)
        return {"status": "handled", "type": event_type}


# ---------------------------------------------------------------------------
# Internal webhook event handlers
# ---------------------------------------------------------------------------


def _handle_checkout_completed(data: dict, db: Session) -> None:
    """Handle ``checkout.session.completed``: create local subscription."""
    customer_id = data.get("customer")
    subscription_id = data.get("subscription")
    if not customer_id or not subscription_id:
        logger.warning("checkout.session.completed missing customer/subscription")
        return

    org = (
        db.query(Organization)
        .filter(Organization.stripe_customer_id == customer_id)
        .first()
    )
    if org is None:
        logger.warning(
            "No organization found for Stripe customer %s", customer_id
        )
        return

    # Retrieve full subscription from Stripe to get price and period
    s = _get_stripe()
    stripe_sub = s.Subscription.retrieve(subscription_id)
    item = stripe_sub["items"]["data"][0]
    price_id = item["price"]["id"]
    quantity = item.get("quantity", 1)

    # Resolve plan tier
    try:
        plan = get_plan_by_stripe_price(price_id)
        tier = plan.tier.value
    except ValueError:
        tier = org.plan_tier

    # Upsert local subscription
    local_sub = (
        db.query(Subscription)
        .filter(Subscription.stripe_subscription_id == subscription_id)
        .first()
    )
    if local_sub is None:
        local_sub = Subscription(
            org_id=org.id,
            stripe_subscription_id=subscription_id,
            stripe_price_id=price_id,
            status=stripe_sub["status"],
            current_period_start=datetime.fromtimestamp(
                stripe_sub["current_period_start"], tz=timezone.utc
            ),
            current_period_end=datetime.fromtimestamp(
                stripe_sub["current_period_end"], tz=timezone.utc
            ),
        )
        db.add(local_sub)
    else:
        local_sub.status = stripe_sub["status"]
        local_sub.stripe_price_id = price_id

    # Update org
    org.plan_tier = tier
    org.bed_count = quantity
    org.is_active = True


def _handle_invoice_paid(data: dict, db: Session) -> None:
    """Handle ``invoice.paid``: record a paid invoice."""
    customer_id = data.get("customer")
    stripe_invoice_id = data.get("id")
    if not customer_id or not stripe_invoice_id:
        return

    org = (
        db.query(Organization)
        .filter(Organization.stripe_customer_id == customer_id)
        .first()
    )
    if org is None:
        logger.warning(
            "No organization found for Stripe customer %s", customer_id
        )
        return

    existing = (
        db.query(Invoice)
        .filter(Invoice.stripe_invoice_id == stripe_invoice_id)
        .first()
    )
    if existing is not None:
        existing.status = "paid"
        existing.paid_at = datetime.now(timezone.utc)
        existing.pdf_url = data.get("invoice_pdf")
        return

    invoice = Invoice(
        org_id=org.id,
        stripe_invoice_id=stripe_invoice_id,
        amount_cents=data.get("amount_paid", 0),
        currency=data.get("currency", "usd"),
        status="paid",
        paid_at=datetime.now(timezone.utc),
        pdf_url=data.get("invoice_pdf"),
    )
    db.add(invoice)


def _handle_invoice_payment_failed(data: dict, db: Session) -> None:
    """Handle ``invoice.payment_failed``: record the failure."""
    customer_id = data.get("customer")
    stripe_invoice_id = data.get("id")
    if not customer_id or not stripe_invoice_id:
        return

    org = (
        db.query(Organization)
        .filter(Organization.stripe_customer_id == customer_id)
        .first()
    )
    if org is None:
        return

    existing = (
        db.query(Invoice)
        .filter(Invoice.stripe_invoice_id == stripe_invoice_id)
        .first()
    )
    if existing is not None:
        existing.status = "payment_failed"
        return

    invoice = Invoice(
        org_id=org.id,
        stripe_invoice_id=stripe_invoice_id,
        amount_cents=data.get("amount_due", 0),
        currency=data.get("currency", "usd"),
        status="payment_failed",
    )
    db.add(invoice)


def _handle_subscription_updated(data: dict, db: Session) -> None:
    """Handle ``customer.subscription.updated``: sync local state."""
    subscription_id = data.get("id")
    if not subscription_id:
        return

    local_sub = (
        db.query(Subscription)
        .filter(Subscription.stripe_subscription_id == subscription_id)
        .first()
    )
    if local_sub is None:
        logger.warning("Subscription %s not found locally", subscription_id)
        return

    local_sub.status = data.get("status", local_sub.status)

    if data.get("current_period_start"):
        local_sub.current_period_start = datetime.fromtimestamp(
            data["current_period_start"], tz=timezone.utc
        )
    if data.get("current_period_end"):
        local_sub.current_period_end = datetime.fromtimestamp(
            data["current_period_end"], tz=timezone.utc
        )

    cancel_at = data.get("cancel_at")
    if cancel_at:
        local_sub.cancel_at = datetime.fromtimestamp(cancel_at, tz=timezone.utc)
    else:
        local_sub.cancel_at = None

    # Sync price / quantity changes
    items = data.get("items", {}).get("data", [])
    if items:
        item = items[0]
        price_id = item.get("price", {}).get("id")
        quantity = item.get("quantity", 1)
        if price_id:
            local_sub.stripe_price_id = price_id
            try:
                plan = get_plan_by_stripe_price(price_id)
                local_sub.organization.plan_tier = plan.tier.value
            except ValueError:
                pass
        local_sub.organization.bed_count = quantity

    # Deactivate org if subscription is no longer paying
    if local_sub.status in ("canceled", "unpaid", "incomplete_expired"):
        local_sub.organization.is_active = False
    elif local_sub.status == "active":
        local_sub.organization.is_active = True


def _handle_subscription_deleted(data: dict, db: Session) -> None:
    """Handle ``customer.subscription.deleted``: mark org inactive."""
    subscription_id = data.get("id")
    if not subscription_id:
        return

    local_sub = (
        db.query(Subscription)
        .filter(Subscription.stripe_subscription_id == subscription_id)
        .first()
    )
    if local_sub is None:
        return

    local_sub.status = "canceled"
    local_sub.organization.is_active = False
    local_sub.organization.plan_tier = "community"


# Handler dispatch table
_EVENT_HANDLERS: dict[str, Any] = {
    "checkout.session.completed": _handle_checkout_completed,
    "invoice.paid": _handle_invoice_paid,
    "invoice.payment_failed": _handle_invoice_payment_failed,
    "customer.subscription.updated": _handle_subscription_updated,
    "customer.subscription.deleted": _handle_subscription_deleted,
}
