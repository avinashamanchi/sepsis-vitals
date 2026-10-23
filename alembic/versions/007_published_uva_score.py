"""Store the published UVA score with each scored reading.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-10-22 00:00:00.000000

Adds scores.uva_score and scores.uva_risk_band (nullable). Existing rows keep
NULL: they were scored before the published UVA was implemented, and are not
rewritten. scores.uva_style (the legacy project score) is unchanged.
Additive; the previous application version ignores the columns.
"""

from alembic import op
import sqlalchemy as sa

revision = "a7b8c9d0e1f2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("scores", sa.Column("uva_score", sa.SmallInteger(), nullable=True))
    op.add_column("scores", sa.Column("uva_risk_band", sa.String(8), nullable=True))


def downgrade() -> None:
    op.drop_column("scores", "uva_risk_band")
    op.drop_column("scores", "uva_score")
