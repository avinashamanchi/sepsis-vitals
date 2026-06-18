"""
sepsis_vitals.billing.models — SQLAlchemy ORM models for billing entities.

Mirrors the conventions established in ``sepsis_vitals.db``:
- Portable UUID / JSON column types (SQLite compat)
- ``_uuid_default`` for primary-key generation
- ``mapped_column`` with ``Mapped`` type annotations
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sepsis_vitals.db import Base, UUIDType, _is_sqlite, _uuid_default


# ---------------------------------------------------------------------------
# Organization
# ---------------------------------------------------------------------------


class Organization(Base):
    """A hospital or health-system account that subscribes to the platform."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True
    )
    plan_tier: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="community"
    )
    bed_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true" if not _is_sqlite else "1"
    )
    trial_ends_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        server_default="now()" if not _is_sqlite else None,
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        server_default="now()" if not _is_sqlite else None,
    )

    # Relationships
    subscriptions: Mapped[list[Subscription]] = relationship(
        back_populates="organization", lazy="dynamic"
    )
    invoices: Mapped[list[Invoice]] = relationship(
        back_populates="organization", lazy="dynamic"
    )

    __table_args__ = (
        Index("idx_organizations_slug", "slug"),
        Index("idx_organizations_stripe", "stripe_customer_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<Organization id={self.id!r} name={self.name!r} "
            f"tier={self.plan_tier!r}>"
        )


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------


class Subscription(Base):
    """Tracks a Stripe subscription tied to an organization."""

    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    org_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("organizations.id"), nullable=False
    )
    stripe_subscription_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    stripe_price_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="incomplete"
    )
    current_period_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    current_period_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancel_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        server_default="now()" if not _is_sqlite else None,
    )

    # Relationships
    organization: Mapped[Organization] = relationship(
        back_populates="subscriptions"
    )

    __table_args__ = (
        Index("idx_subscriptions_org", "org_id"),
        Index("idx_subscriptions_stripe", "stripe_subscription_id"),
        Index("idx_subscriptions_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<Subscription id={self.id!r} org_id={self.org_id!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------


class Invoice(Base):
    """Mirrors relevant fields from a Stripe invoice for local querying."""

    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    org_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("organizations.id"), nullable=False
    )
    stripe_invoice_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default="usd"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="draft"
    )
    paid_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pdf_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        server_default="now()" if not _is_sqlite else None,
    )

    # Relationships
    organization: Mapped[Organization] = relationship(
        back_populates="invoices"
    )

    __table_args__ = (
        Index("idx_invoices_org", "org_id"),
        Index("idx_invoices_stripe", "stripe_invoice_id"),
        Index("idx_invoices_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<Invoice id={self.id!r} org_id={self.org_id!r} "
            f"amount={self.amount_cents} status={self.status!r}>"
        )
