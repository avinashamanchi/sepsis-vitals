"""Add Hour-1 sepsis bundle tables.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sepsis_bundles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id"),
            nullable=False,
        ),
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("protocol_version", sa.String(48), nullable=False),
        sa.Column("status", sa.String(16), server_default="open", nullable=False),
        sa.Column("started_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("risk_level_at_start", sa.String(16), nullable=True),
        sa.Column("time_to_antibiotics_s", sa.Float(), nullable=True),
        sa.Column("compliance_pct", sa.Float(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('open', 'completed', 'expired', 'cancelled')",
            name="ck_bundle_status",
        ),
    )
    op.create_index(
        "ix_bundle_patient_status", "sepsis_bundles", ["patient_id", "status"]
    )

    op.create_table(
        "bundle_tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "bundle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sepsis_bundles.id"),
            nullable=False,
        ),
        sa.Column("task_key", sa.String(48), nullable=False),
        sa.Column("order_index", sa.Integer(), server_default="0"),
        sa.Column("target_minutes", sa.Integer(), server_default="60"),
        sa.Column("critical", sa.Boolean(), server_default="true"),
        sa.Column("completed", sa.Boolean(), server_default="false"),
        sa.Column("completed_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("minutes_from_start", sa.Float(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_bundletask_bundle_key",
        "bundle_tasks",
        ["bundle_id", "task_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_bundletask_bundle_key", table_name="bundle_tasks")
    op.drop_table("bundle_tasks")
    op.drop_index("ix_bundle_patient_status", table_name="sepsis_bundles")
    op.drop_table("sepsis_bundles")
