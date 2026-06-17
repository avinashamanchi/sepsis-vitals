"""
SQLAlchemy ORM models for the sepsis-vitals database.

Maps to the schema defined in docker/postgres/init.sql.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Generator, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID as PG_UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

# ---------------------------------------------------------------------------
# Database URL configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sepsis_vitals.db")

# Convert postgres+asyncpg to regular postgresql for sync usage
if DATABASE_URL.startswith("postgresql+asyncpg"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg", "postgresql")

_is_sqlite = DATABASE_URL.startswith("sqlite")

# Choose column types that work for both PostgreSQL and SQLite.
# SQLite has no native UUID, INET, or JSONB, so we fall back to String/Text.
UUIDType = String(36) if _is_sqlite else PG_UUID(as_uuid=True)
InetType = String(45) if _is_sqlite else INET
JsonType = Text if _is_sqlite else JSONB


def _uuid_default() -> str:
    """Return a new UUID4 string (used as column default for SQLite)."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Engine & session
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    echo=False,
    **({"connect_args": {"check_same_thread": False}} if _is_sqlite else {}),
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class User(Base):
    """Maps to the ``users`` table."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    site_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    totp_secret: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    mfa_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    failed_attempts: Mapped[int] = mapped_column(
        SmallInteger, default=0, server_default="0"
    )
    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    audit_logs: Mapped[list[AuditLog]] = relationship(
        back_populates="user", lazy="dynamic"
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('nurse', 'researcher', 'system_admin')",
            name="ck_users_role",
        ),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id!r} email={self.email!r} role={self.role!r}>"


class Patient(Base):
    """Maps to the ``patients`` table."""

    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    external_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False
    )
    site_id: Mapped[str] = mapped_column(String(32), nullable=False)
    age_years: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )

    # Relationships
    vitals: Mapped[list[VitalReading]] = relationship(
        back_populates="patient", lazy="dynamic"
    )
    alerts: Mapped[list[Alert]] = relationship(
        back_populates="patient", lazy="dynamic"
    )

    __table_args__ = (
        CheckConstraint("sex IN ('M', 'F', 'U')", name="ck_patients_sex"),
    )

    def __repr__(self) -> str:
        return (
            f"<Patient id={self.id!r} external_id={self.external_id!r}>"
        )


class VitalReading(Base):
    """Maps to the ``vitals`` table (vital sign observations)."""

    __tablename__ = "vitals"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    patient_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("patients.id"), nullable=False
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    heart_rate: Mapped[Optional[int]] = mapped_column(
        SmallInteger, nullable=True
    )
    resp_rate: Mapped[Optional[int]] = mapped_column(
        SmallInteger, nullable=True
    )
    sbp: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    spo2: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    gcs: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )

    # Relationships
    patient: Mapped[Patient] = relationship(back_populates="vitals")
    scores: Mapped[list[Score]] = relationship(
        back_populates="vital_reading", lazy="dynamic"
    )

    __table_args__ = (
        CheckConstraint("gcs BETWEEN 3 AND 15", name="ck_vitals_gcs"),
        Index("idx_vitals_patient_time", "patient_id", recorded_at.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"<VitalReading id={self.id!r} patient_id={self.patient_id!r} "
            f"recorded_at={self.recorded_at!r}>"
        )


class Score(Base):
    """Maps to the ``scores`` table (computed sepsis risk scores)."""

    __tablename__ = "scores"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    vital_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("vitals.id"), nullable=False
    )
    qsofa: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    sirs_count: Mapped[Optional[int]] = mapped_column(
        SmallInteger, nullable=True
    )
    shock_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    news2_style: Mapped[Optional[int]] = mapped_column(
        SmallInteger, nullable=True
    )
    uva_style: Mapped[Optional[int]] = mapped_column(
        SmallInteger, nullable=True
    )
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    alert_flag: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    component_flags: Mapped[Optional[Any]] = mapped_column(
        JsonType, nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )

    # Relationships
    vital_reading: Mapped[VitalReading] = relationship(back_populates="scores")
    alerts: Mapped[list[Alert]] = relationship(
        back_populates="score", lazy="dynamic"
    )

    __table_args__ = (
        Index("idx_scores_risk", "risk_level", "alert_flag"),
    )

    def __repr__(self) -> str:
        return (
            f"<Score id={self.id!r} risk_level={self.risk_level!r} "
            f"alert_flag={self.alert_flag!r}>"
        )


class Alert(Base):
    """Maps to the ``alerts`` table."""

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    score_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("scores.id"), nullable=False
    )
    patient_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey("patients.id"), nullable=False
    )
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="active", server_default="active"
    )
    action_by: Mapped[Optional[str]] = mapped_column(UUIDType, nullable=True)
    action_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    time_to_action_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )
    actioned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    score: Mapped[Score] = relationship(back_populates="alerts")
    patient: Mapped[Patient] = relationship(back_populates="alerts")

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'acknowledged', 'dismissed', 'escalated')",
            name="ck_alerts_status",
        ),
        Index("idx_alerts_status", "status", created_at.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"<Alert id={self.id!r} risk_level={self.risk_level!r} "
            f"status={self.status!r}>"
        )


class AuditLog(Base):
    """Maps to the ``audit_log`` table."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        UUIDType, ForeignKey("users.id"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    resource_id: Mapped[Optional[str]] = mapped_column(UUIDType, nullable=True)
    details: Mapped[Optional[Any]] = mapped_column(JsonType, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(InetType, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )

    # Relationships
    user: Mapped[Optional[User]] = relationship(back_populates="audit_logs")

    __table_args__ = (
        Index("idx_audit_user_time", "user_id", created_at.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id!r} action={self.action!r} "
            f"user_id={self.user_id!r}>"
        )


# ---------------------------------------------------------------------------
# Dependency injection helper (FastAPI compatible)
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, ensuring it is closed after use.

    Usage with FastAPI::

        @app.get("/patients")
        def list_patients(db: Session = Depends(get_db)):
            return db.query(Patient).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Table creation helper
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create all tables defined by the ORM models.

    Safe to call multiple times; existing tables are not modified.
    """
    Base.metadata.create_all(bind=engine)
