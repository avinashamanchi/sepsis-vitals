"""
sepsis_vitals.bundles.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQLAlchemy ORM models for Hour-1 bundle tracking.

These reuse the shared declarative ``Base`` from :mod:`sepsis_vitals.db`, so
``Base.metadata.create_all`` (and the Alembic autogenerate step) pick them up
automatically once this module is imported.  Column-type helpers
(``UUIDType``, ``JsonType`` and the ``_is_sqlite`` switch) are also reused so
the tables behave identically under SQLite (tests/dev) and PostgreSQL (prod).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sepsis_vitals.db import (
    Base,
    UUIDType,
    _is_sqlite,
    _utcnow,
    _uuid_default,
)


class SepsisBundle(Base):
    """A single Hour-1 bundle instance opened for a patient.

    Lifecycle status:

    * ``open``      -- clock running, tasks outstanding.
    * ``completed`` -- every applicable critical task marked done.
    * ``expired``   -- 60-minute window elapsed with critical tasks missing.
    * ``cancelled`` -- clinician closed the bundle (e.g. sepsis ruled out).
    """

    __tablename__ = "sepsis_bundles"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    patient_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("patients.id"), nullable=False, index=True
    )
    # Nullable link to the alert/score that triggered the bundle.
    alert_id: Mapped[Optional[str]] = mapped_column(UUIDType, nullable=True)
    protocol_version: Mapped[str] = mapped_column(String(48), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="open", server_default="open", nullable=False
    )
    started_by: Mapped[Optional[str]] = mapped_column(UUIDType, nullable=True)
    risk_level_at_start: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True
    )
    # Denormalised metric: minutes from start to antibiotics (the headline KPI).
    time_to_antibiotics_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    compliance_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow,
        server_default="now()" if not _is_sqlite else None,
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    tasks: Mapped[list["BundleTask"]] = relationship(
        back_populates="bundle",
        cascade="all, delete-orphan",
        order_by="BundleTask.order_index",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'completed', 'expired', 'cancelled')",
            name="ck_bundle_status",
        ),
        Index("ix_bundle_patient_status", "patient_id", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<SepsisBundle id={self.id!r} patient={self.patient_id!r} status={self.status!r}>"


class BundleTask(Base):
    """A single task within a bundle, with its completion state."""

    __tablename__ = "bundle_tasks"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    bundle_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("sepsis_bundles.id"), nullable=False, index=True
    )
    task_key: Mapped[str] = mapped_column(String(48), nullable=False)
    order_index: Mapped[int] = mapped_column(default=0)
    target_minutes: Mapped[int] = mapped_column(default=60)
    critical: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true"
    )
    completed: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    completed_by: Mapped[Optional[str]] = mapped_column(UUIDType, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Seconds from bundle start to task completion (for audit / KPI dashboards).
    minutes_from_start: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    bundle: Mapped[SepsisBundle] = relationship(back_populates="tasks")

    __table_args__ = (
        Index("ix_bundletask_bundle_key", "bundle_id", "task_key", unique=True),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<BundleTask key={self.task_key!r} completed={self.completed!r}>"
