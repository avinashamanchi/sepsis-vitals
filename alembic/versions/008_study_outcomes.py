"""Adjudicated outcomes for a local silent-mode study.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-10-23 00:00:00.000000

New table study_outcomes (append-only by convention: corrections are new
rows). Additive; earlier application versions ignore it.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b8c9d0e1f2a3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("patient_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("site_id", sa.String(32), nullable=False),
        sa.Column("sepsis", sa.Boolean(), nullable=False),
        sa.Column("onset_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("died_in_hospital", sa.Boolean(), nullable=True),
        sa.Column("definition", sa.String(64), nullable=False),
        sa.Column("recorded_by", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_study_outcomes_patient_time", "study_outcomes", ["patient_id", sa.text("created_at DESC")])


def downgrade() -> None:
    op.drop_index("idx_study_outcomes_patient_time", table_name="study_outcomes")
    op.drop_table("study_outcomes")
