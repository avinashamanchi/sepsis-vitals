"""Initial schema — all core tables for sepsis-vitals.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-06-22 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- users --
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("site_id", sa.String(32), nullable=True),
        sa.Column("totp_secret", sa.String(64), nullable=True),
        sa.Column("mfa_enabled", sa.Boolean(), server_default="false"),
        sa.Column("failed_attempts", sa.SmallInteger(), server_default="0"),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "role IN ('nurse', 'researcher', 'system_admin')",
            name="ck_users_role",
        ),
    )

    # -- patients --
    op.create_table(
        "patients",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("external_id", sa.String(64), unique=True, nullable=False),
        sa.Column("site_id", sa.String(32), nullable=False),
        sa.Column("age_years", sa.SmallInteger(), nullable=True),
        sa.Column("sex", sa.String(1), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("sex IN ('M', 'F', 'U')", name="ck_patients_sex"),
    )

    # -- vitals --
    op.create_table(
        "vitals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id"),
            nullable=False,
        ),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("heart_rate", sa.SmallInteger(), nullable=True),
        sa.Column("resp_rate", sa.SmallInteger(), nullable=True),
        sa.Column("sbp", sa.SmallInteger(), nullable=True),
        sa.Column("spo2", sa.SmallInteger(), nullable=True),
        sa.Column("gcs", sa.SmallInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("gcs BETWEEN 3 AND 15", name="ck_vitals_gcs"),
    )
    op.create_index(
        "idx_vitals_patient_time",
        "vitals",
        ["patient_id", sa.text("recorded_at DESC")],
    )

    # -- scores --
    op.create_table(
        "scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "vital_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("vitals.id"),
            nullable=False,
        ),
        sa.Column("qsofa", sa.SmallInteger(), nullable=True),
        sa.Column("sirs_count", sa.SmallInteger(), nullable=True),
        sa.Column("shock_index", sa.Float(), nullable=True),
        sa.Column("news2_style", sa.SmallInteger(), nullable=True),
        sa.Column("uva_style", sa.SmallInteger(), nullable=True),
        sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column("alert_flag", sa.Boolean(), server_default="false"),
        sa.Column("component_flags", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("idx_scores_risk", "scores", ["risk_level", "alert_flag"])

    # -- alerts --
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "score_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("scores.id"),
            nullable=False,
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id"),
            nullable=False,
        ),
        sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column(
            "status", sa.String(16), server_default="active"
        ),
        sa.Column("action_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action_reason", sa.Text(), nullable=True),
        sa.Column("time_to_action_s", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("actioned_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('active', 'acknowledged', 'dismissed', 'escalated')",
            name="ck_alerts_status",
        ),
    )
    op.create_index(
        "idx_alerts_status",
        "alerts",
        ["status", sa.text("created_at DESC")],
    )

    # -- audit_log --
    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("resource_type", sa.String(32), nullable=True),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("ip_address", postgresql.INET(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_audit_user_time",
        "audit_log",
        ["user_id", sa.text("created_at DESC")],
    )

    # -- prediction_records --
    op.create_table(
        "prediction_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("patient_id", sa.String(100), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("risk_probability", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column("alert_fired", sa.Boolean(), server_default="false"),
        sa.Column("input_vitals", postgresql.JSONB(), nullable=True),
        sa.Column("output_scores", postgresql.JSONB(), nullable=True),
        sa.Column("top_risk_factors", postgresql.JSONB(), nullable=True),
        sa.Column("confidence_lower", sa.Float(), nullable=True),
        sa.Column("confidence_upper", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(32), nullable=True),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("ip_address", postgresql.INET(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_predictions_patient_time",
        "prediction_records",
        ["patient_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_predictions_risk",
        "prediction_records",
        ["risk_level", "alert_fired"],
    )


def downgrade() -> None:
    op.drop_table("prediction_records")
    op.drop_table("audit_log")
    op.drop_table("alerts")
    op.drop_table("scores")
    op.drop_table("vitals")
    op.drop_table("patients")
    op.drop_table("users")
