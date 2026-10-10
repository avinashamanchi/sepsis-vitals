"""
SQLAlchemy ORM models for the sepsis-vitals database.

The production schema is owned by the Alembic migrations in ``alembic/``;
``init_db`` creates tables directly only for development databases and for
feature tables that are not migrated yet (see ``init_db``).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Generator, Optional


def _utcnow() -> datetime:
    """Return the current UTC datetime (used as Python-side column default)."""
    return datetime.now(timezone.utc)

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
    UniqueConstraint,
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
import sqlalchemy.types as sa_types


# ---------------------------------------------------------------------------
# Encrypted string column type (AES-256-GCM via FieldEncryptor)
# ---------------------------------------------------------------------------


class EncryptedString(sa_types.TypeDecorator):
    """SQLAlchemy column type that encrypts values at rest using AES-256-GCM.

    Transparently encrypts on INSERT/UPDATE and decrypts on SELECT.
    Falls through to plaintext when SEPSIS_PII_KEY is not configured
    (dev mode) or when reading legacy unencrypted data (no ``enc:`` prefix).
    """

    impl = sa_types.Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        from sepsis_vitals.security import FieldEncryptor
        return FieldEncryptor.get().encrypt(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        from sepsis_vitals.security import FieldEncryptor
        return FieldEncryptor.get().decrypt(value)

# ---------------------------------------------------------------------------
# Database URL configuration
# ---------------------------------------------------------------------------

def sync_database_url(url: str) -> str:
    """Normalise a database URL to the synchronous psycopg (v3) driver.

    Accepts ``postgres://``, ``postgresql://`` and legacy
    ``postgresql+asyncpg://`` URLs; the app and Alembic both use synchronous
    SQLAlchemy, and psycopg is the driver installed by the ``api`` extra.
    """
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg2://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


DATABASE_URL = sync_database_url(os.getenv("DATABASE_URL", "sqlite:///./sepsis_vitals.db"))

_is_sqlite = DATABASE_URL.startswith("sqlite")

# Choose column types that work for both PostgreSQL and SQLite.
# SQLite has no native UUID, INET, or JSONB, so we fall back to String/Text.
NIL_UUID = "00000000-0000-0000-0000-000000000000"


class GUID(sa_types.TypeDecorator):
    """UUID column that always reads and binds canonical *strings*.

    PostgreSQL stores a native UUID; SQLite stores CHAR(36). The application
    treats every identifier as ``str`` (JWT ``sub``, Pydantic models, dict
    keys); ``PG_UUID(as_uuid=True)`` returned ``uuid.UUID`` objects on
    Postgres, which broke token creation at login. Values that are not UUIDs
    (an MRN, a free-form patient label, a stray path segment) bind to the nil
    UUID, so lookups find nothing instead of raising a database error.
    """

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=False))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        try:
            return str(uuid.UUID(str(value)))
        except ValueError:
            return NIL_UUID

    def process_result_value(self, value, dialect):
        return None if value is None else str(value)


def is_uuid(value: object) -> bool:
    """True when *value* is a well-formed UUID (string or UUID object)."""
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    return True


UUIDType = GUID()
InetType = String(45) if _is_sqlite else INET
JsonType = Text if _is_sqlite else JSONB


def _uuid_default() -> str:
    """Return a new UUID4 string (used as column default for SQLite)."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Engine & session
# ---------------------------------------------------------------------------

_engine_kwargs: dict = {"echo": False}

if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    _engine_kwargs["pool_size"] = pool_size
    _engine_kwargs["max_overflow"] = max_overflow

    # Enforce SSL for PostgreSQL in production (HIPAA transit encryption)
    db_ssl = os.getenv("DB_SSL", "disable")
    if db_ssl == "require":
        _engine_kwargs["connect_args"] = {"sslmode": "require"}
    elif os.getenv("SEPSIS_ENV") == "production" and db_ssl != "require":
        import logging as _log
        _log.getLogger(__name__).warning(
            "DB_SSL is not set to 'require' in production. "
            "Database connections may transmit PHI in plaintext."
        )

engine = create_engine(DATABASE_URL, **_engine_kwargs)

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
    email: Mapped[str] = mapped_column(EncryptedString, nullable=False)
    email_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=""
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    site_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    totp_secret: Mapped[Optional[str]] = mapped_column(
        EncryptedString, nullable=True
    )
    # JSON list of keyed hashes of unused single-use recovery codes.
    mfa_recovery_hashes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
        DateTime(timezone=True), default=_utcnow,
        server_default="now()" if not _is_sqlite else None
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
        EncryptedString, nullable=False
    )
    # Identity is (site_id, external_id_hash): MRNs are only unique per site.
    external_id_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, default=""
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
        UniqueConstraint("site_id", "external_id_hash", name="uq_patients_site_mrn"),
        Index("idx_patients_mrn_hash", "external_id_hash"),
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
    dbp: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    map_pressure: Mapped[Optional[float]] = mapped_column(
        "map", Float, nullable=True
    )
    lactate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wbc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    procalcitonin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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


class PredictionRecord(Base):
    """Immutable ledger of every ML prediction for clinical audit trail.

    Redis handles ephemeral rolling-window math (24h TTL), but every
    prediction must be permanently recorded here so that risk-management
    and legal teams can reconstruct the algorithmic decision history
    for any patient at any point in time.
    """

    __tablename__ = "prediction_records"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_uuid_default
    )
    patient_id: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(UUIDType, nullable=True)
    risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    alert_fired: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    input_vitals: Mapped[Optional[Any]] = mapped_column(
        JsonType, nullable=True
    )
    output_scores: Mapped[Optional[Any]] = mapped_column(
        JsonType, nullable=True
    )
    top_risk_factors: Mapped[Optional[Any]] = mapped_column(
        JsonType, nullable=True
    )
    confidence_lower: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    confidence_upper: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    recommendation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(InetType, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default="now()" if not _is_sqlite else None
    )

    __table_args__ = (
        Index("idx_predictions_patient_time", "patient_id", created_at.desc()),
        Index("idx_predictions_risk", "risk_level", "alert_fired"),
    )

    def __repr__(self) -> str:
        return (
            f"<PredictionRecord id={self.id!r} patient={self.patient_id!r} "
            f"risk={self.risk_level!r} prob={self.risk_probability:.2f}>"
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


class SchemaMismatchError(RuntimeError):
    """An existing table lacks columns the ORM needs (the schema is out of date)."""


def _is_concurrent_create(exc: BaseException) -> bool:
    """True only for the error another worker's identical CREATE produces.

    SQLite: "table users already exists" / "index ... already exists".
    PostgreSQL: DuplicateTable ('relation "users" already exists') or, when
    two CREATE TABLE statements race in the catalog, a unique violation on
    ``pg_type_typname_nsp_index``. Permission, connection and other unique
    errors do not match and propagate.
    """
    message = str(getattr(exc, "orig", exc)).lower()
    return "already exists" in message or ("duplicate key" in message and "pg_type" in message)


def schema_drift(bind: Any = None) -> list[str]:
    """``table.column`` names the ORM defines but existing tables lack.

    Tables that do not exist yet are not reported (``create_all`` adds them).
    Only names are returned, never data.
    """
    from sqlalchemy import inspect

    inspector = inspect(bind if bind is not None else engine)
    missing: list[str] = []
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        present = {col["name"] for col in inspector.get_columns(table.name)}
        missing += [f"{table.name}.{col.name}" for col in table.columns if col.name not in present]
    return missing


def init_db(production: Optional[bool] = None, attempts: int = 5) -> None:
    """Create missing ORM tables, within the limits of who owns the schema.

    * **Production** (``SEPSIS_ENV=production``): Alembic owns the schema. If
      the database is not Alembic-managed yet, nothing is created: creating
      the tables here would make a later ``alembic upgrade head`` fail on
      "already exists" (for example API replicas starting before a separate
      migration task). ``/ready`` reports the database as not ready until it
      is migrated. On a migrated database only tables that have no migration
      yet (frozen billing and bundle features, when enabled) are created.
    * **Development**: all missing tables are created.

    Several uvicorn workers run this at the same time. When another worker
    creates the same table first, the "already exists" error is retried, at
    most ``attempts`` times; every other error propagates. Afterwards the
    existing tables must have every ORM column, otherwise
    :class:`SchemaMismatchError` stops startup instead of failing requests
    later.
    """
    import logging
    import time as _time

    from sqlalchemy import inspect
    from sqlalchemy.exc import DBAPIError

    log = logging.getLogger(__name__)
    if production is None:
        production = os.getenv("SEPSIS_ENV", "development") == "production"

    if production:
        with engine.connect() as conn:
            managed = inspect(conn).has_table("alembic_version")
        if not managed:
            log.error(
                "Database is not managed by Alembic; not creating tables in production. "
                "Run `alembic upgrade head` (docker/entrypoint.sh does this by default)."
            )
            return

    for attempt in range(1, attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)
            break
        except DBAPIError as exc:
            if attempt == attempts or not _is_concurrent_create(exc):
                raise
            # Another worker is creating the same schema; its tables are
            # visible on the next pass, which then has nothing left to create.
            _time.sleep(0.1 * attempt)

    missing = schema_drift()
    if missing:
        raise SchemaMismatchError(
            "Database schema is out of date; missing columns: "
            + ", ".join(missing)
            + ". Run `alembic upgrade head` (or recreate a development database)."
        )
