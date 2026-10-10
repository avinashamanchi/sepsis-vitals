"""Record the last accepted TOTP time step per user (replay protection).

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-10-10 00:00:00.000000

Additive and nullable: existing rows get NULL ("no step used yet"), so every
enrolled user's next valid code is accepted. Older application versions
ignore the column, so the migration can run before the new code is
deployed. Replay protection is only complete once no old replica serves
logins (see docs/mfa_replay_protection.md).
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("totp_last_step", sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "totp_last_step")
