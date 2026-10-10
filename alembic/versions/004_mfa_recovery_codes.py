"""Store keyed hashes of single-use MFA recovery codes.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-10-09 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("mfa_recovery_hashes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "mfa_recovery_hashes")
