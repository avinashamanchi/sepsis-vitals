"""Align the schema with the ORM; make patient identity per site.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-10-06 00:00:00.000000

Fixes drift between migrations 001/002 and ``sepsis_vitals.db``:

* ``users.email``, ``users.totp_secret`` and ``patients.external_id`` hold
  AES-GCM ciphertext (``EncryptedString``), which overflows VARCHAR(64/255)
  for realistic values. Widen them to TEXT.
* Add the blind-index lookup columns the application queries on every login
  and patient lookup (``users.email_hash``, ``patients.external_id_hash``)
  and backfill them from the decrypted values.
* Patient identity becomes ``(site_id, external_id_hash)``: two hospitals may
  legitimately use the same MRN.
* Add the vitals columns the ORM writes (dbp, map, lactate, wbc,
  procalcitonin).

The backfill decrypts existing rows, so run it with the same
``SEPSIS_PII_KEY`` the application uses. It is skipped in offline (--sql)
mode; generate offline SQL only for empty databases.
"""

from alembic import context, op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None

_VITALS_COLUMNS = (
    ("dbp", sa.SmallInteger()),
    ("map", sa.Float()),
    ("lactate", sa.Float()),
    ("wbc", sa.Float()),
    ("procalcitonin", sa.Float()),
)


def _backfill_blind_index(table: str, source: str, target: str) -> None:
    """Populate *target* with the blind index of the decrypted *source*."""
    if context.is_offline_mode():
        op.execute(f"-- {table}.{target}: backfill requires online mode (empty DB assumed)")
        return
    bind = op.get_bind()
    rows = bind.execute(sa.text(f"SELECT id, {source} FROM {table}")).fetchall()  # nosec B608
    if not rows:
        return
    from sepsis_vitals.security import FieldEncryptionError, FieldEncryptor, compute_blind_index

    encryptor = FieldEncryptor.get()
    failures = 0
    for row_id, value in rows:
        try:
            plaintext = encryptor.decrypt(value) if value is not None else ""
        except FieldEncryptionError:
            failures += 1
            continue
        bind.execute(
            sa.text(f"UPDATE {table} SET {target} = :h WHERE id = :id"),  # nosec B608
            {"h": compute_blind_index(plaintext), "id": row_id},
        )
    if failures:
        # Abort (the transaction rolls back); never print the values.
        raise RuntimeError(
            f"{failures} of {len(rows)} {table}.{source} values could not be decrypted. "
            "Run the migration with the same SEPSIS_PII_KEY as the application."
        )


def _require_unique(columns: str, table: str, what: str) -> None:
    """Fail with a clear message (no values) before a unique index would."""
    if context.is_offline_mode():
        return
    duplicates = op.get_bind().execute(
        sa.text(f"SELECT COUNT(*) FROM (SELECT 1 FROM {table} GROUP BY {columns} HAVING COUNT(*) > 1) d")  # nosec B608
    ).scalar()
    if duplicates:
        raise RuntimeError(
            f"{duplicates} duplicate {what} groups in {table}; merge or remove the duplicate "
            "records before upgrading (values are not shown)."
        )


def upgrade() -> None:
    # Encrypted columns need unbounded storage.
    op.alter_column("users", "email", type_=sa.Text(), existing_nullable=False)
    op.alter_column("users", "totp_secret", type_=sa.Text(), existing_nullable=True)
    op.alter_column("patients", "external_id", type_=sa.Text(), existing_nullable=False)

    # Blind-index lookup columns.
    op.add_column("users", sa.Column("email_hash", sa.String(64), nullable=True))
    op.add_column("patients", sa.Column("external_id_hash", sa.String(64), nullable=True))
    _backfill_blind_index("users", "email", "email_hash")
    _backfill_blind_index("patients", "external_id", "external_id_hash")
    op.alter_column("users", "email_hash", existing_type=sa.String(64), nullable=False)
    op.alter_column("patients", "external_id_hash", existing_type=sa.String(64), nullable=False)
    _require_unique("email_hash", "users", "account email")
    _require_unique("site_id, external_id_hash", "patients", "per-site MRN")
    op.create_index("uq_users_email_hash", "users", ["email_hash"], unique=True)
    op.create_unique_constraint(
        "uq_patients_site_mrn", "patients", ["site_id", "external_id_hash"]
    )
    op.create_index("idx_patients_mrn_hash", "patients", ["external_id_hash"])

    for name, type_ in _VITALS_COLUMNS:
        op.add_column("vitals", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_VITALS_COLUMNS):
        op.drop_column("vitals", name)
    op.drop_index("idx_patients_mrn_hash", table_name="patients")
    op.drop_constraint("uq_patients_site_mrn", "patients", type_="unique")
    op.drop_index("uq_users_email_hash", table_name="users")
    op.drop_column("patients", "external_id_hash")
    op.drop_column("users", "email_hash")
    # Narrowing back can fail if ciphertext exceeds the old limits.
    op.alter_column("patients", "external_id", type_=sa.String(64), existing_nullable=False)
    op.alter_column("users", "totp_secret", type_=sa.String(64), existing_nullable=True)
    op.alter_column("users", "email", type_=sa.String(255), existing_nullable=False)
