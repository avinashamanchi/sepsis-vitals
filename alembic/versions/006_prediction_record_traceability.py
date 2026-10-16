"""Traceable, append-only prediction records.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-10-15 00:00:00.000000

Adds five nullable columns to ``prediction_records`` (request_id, source,
threshold_policy, input_sha256, record_sha256) and makes the table
append-only: UPDATE, DELETE and TRUNCATE are refused by triggers.

Existing rows are not rewritten. They keep NULL in the new columns and are
reported as "unsealed" by ``sepsis_vitals.audit_records.verify_record``.

Deployment order: run this migration before the new application version
starts. The new version refuses to start against a table without these
columns (``init_db`` schema check). The old version keeps working after
the migration because it never updates or deletes prediction records.

Downgrade drops the triggers and the five columns, which removes their
values (request IDs, digests). Take a backup first if they must be kept.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("request_id", 64),
    ("source", 32),
    ("threshold_policy", 64),
    ("input_sha256", 64),
    ("record_sha256", 64),
)

# Kept literally here (not imported from the application) so the migration
# never changes when application code does.
_UPGRADE_SQL = (
    "CREATE OR REPLACE FUNCTION prediction_records_append_only() RETURNS trigger AS $$ "
    "BEGIN RAISE EXCEPTION USING MESSAGE = 'prediction_records is append-only: ' || TG_OP || ' refused'; END; "
    "$$ LANGUAGE plpgsql",
    "CREATE TRIGGER prediction_records_no_change BEFORE UPDATE OR DELETE ON prediction_records "
    "FOR EACH ROW EXECUTE FUNCTION prediction_records_append_only()",
    "CREATE TRIGGER prediction_records_no_truncate BEFORE TRUNCATE ON prediction_records "
    "FOR EACH STATEMENT EXECUTE FUNCTION prediction_records_append_only()",
)
_DOWNGRADE_SQL = (
    "DROP TRIGGER IF EXISTS prediction_records_no_truncate ON prediction_records",
    "DROP TRIGGER IF EXISTS prediction_records_no_change ON prediction_records",
    "DROP FUNCTION IF EXISTS prediction_records_append_only()",
)


def upgrade() -> None:
    for name, length in _COLUMNS:
        op.add_column("prediction_records", sa.Column(name, sa.String(length), nullable=True))
    for statement in _UPGRADE_SQL:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE_SQL:
        op.execute(statement)
    for name, _ in reversed(_COLUMNS):
        op.drop_column("prediction_records", name)
