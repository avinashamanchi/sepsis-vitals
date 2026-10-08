"""
tests/test_migrations.py — Alembic migrations must produce the ORM schema.

Regression for drift where migrations lacked users.email_hash,
patients.external_id_hash and five vitals columns, so a database built with
``alembic upgrade head`` failed at login and patient lookup.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

alembic = pytest.importorskip("alembic")

ROOT = Path(__file__).resolve().parents[1]

# Billing is frozen: its tables are created on demand with
# SEPSIS_ENABLE_BILLING=true and are intentionally not migrated yet.
NOT_MIGRATED = {"organizations", "subscriptions", "invoices"}


def _offline_postgres_sql(monkeypatch) -> str:
    from alembic import command
    from alembic.config import Config

    monkeypatch.delenv("DATABASE_URL", raising=False)
    buf = io.StringIO()
    cfg = Config(str(ROOT / "alembic.ini"), output_buffer=buf)
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", "postgresql://user:pass@localhost/schema_check")
    command.upgrade(cfg, "head", sql=True)
    return buf.getvalue()


def _migrated_columns(sql: str) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for m in re.finditer(r"CREATE TABLE (\w+) \((.*?)\n\);", sql, re.S):
        cols = set(re.findall(r"^\s+(\w+) ", m.group(2), re.M))
        tables[m.group(1)] = cols - {"PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK"}
    for m in re.finditer(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", sql):
        tables.setdefault(m.group(1), set()).add(m.group(2))
    for m in re.finditer(r"ALTER TABLE (\w+) DROP COLUMN (\w+)", sql):
        tables.get(m.group(1), set()).discard(m.group(2))
    tables.pop("alembic_version", None)
    return tables


def _orm_columns() -> dict[str, set[str]]:
    import sepsis_vitals.billing.models  # noqa: F401  (register on Base)
    import sepsis_vitals.bundles.models  # noqa: F401
    from sepsis_vitals.db import Base

    return {t.name: {c.name for c in t.columns} for t in Base.metadata.sorted_tables}


def test_migrations_create_every_orm_table_and_column(monkeypatch):
    migrated = _migrated_columns(_offline_postgres_sql(monkeypatch))
    orm = _orm_columns()

    missing_tables = set(orm) - set(migrated) - NOT_MIGRATED
    assert not missing_tables, f"tables never migrated: {sorted(missing_tables)}"
    assert not set(migrated) - set(orm), "migrations create tables the ORM does not define"

    drift = {
        table: {"orm_only": sorted(orm[table] - migrated[table]),
                "migration_only": sorted(migrated[table] - orm[table])}
        for table in set(orm) & set(migrated)
        if orm[table] != migrated[table]
    }
    assert not drift, f"column drift between ORM and migrations: {drift}"


def test_encrypted_columns_are_not_length_limited(monkeypatch):
    """AES-GCM ciphertext overflows VARCHAR(64): a 15-char MRN encrypts to 64+ chars."""
    sql = _offline_postgres_sql(monkeypatch)
    for table, column in (("users", "email"), ("users", "totp_secret"), ("patients", "external_id")):
        assert re.search(
            rf"ALTER TABLE {table} ALTER COLUMN {column} TYPE TEXT", sql
        ), f"{table}.{column} must be TEXT"


def test_patient_identity_is_unique_per_site(monkeypatch):
    sql = _offline_postgres_sql(monkeypatch)
    assert re.search(r"uq_patients_site_mrn UNIQUE \(site_id, external_id_hash\)", sql)


def test_backfill_computes_blind_index_from_decrypted_values(monkeypatch):
    """Existing encrypted rows get the same blind index the app computes at login."""
    import base64
    import importlib.util
    import os

    import sqlalchemy as sa
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    from sepsis_vitals.security import FieldEncryptor, compute_blind_index

    monkeypatch.setenv("SEPSIS_PII_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setattr(FieldEncryptor, "_instance", None, raising=False)
    encryptor = FieldEncryptor.get()

    spec = importlib.util.spec_from_file_location(
        "mig003", ROOT / "alembic" / "versions" / "003_align_schema_with_orm.py"
    )
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)

    class _Online:
        @staticmethod
        def is_offline_mode() -> bool:
            return False

    monkeypatch.setattr(mig, "context", _Online)

    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT, email_hash TEXT)"))
        conn.execute(
            sa.text("INSERT INTO users (id, email) VALUES ('u1', :e)"),
            {"e": encryptor.encrypt("Nurse.A@Example.org")},
        )
        with Operations.context(MigrationContext.configure(conn)):
            mig._backfill_blind_index("users", "email", "email_hash")
        stored = conn.execute(sa.text("SELECT email_hash FROM users WHERE id='u1'")).scalar()

    assert stored == compute_blind_index("nurse.a@example.org")


@pytest.mark.parametrize("url,expected", [
    ("postgresql+asyncpg://u:p@db:5432/x", "postgresql+psycopg://u:p@db:5432/x"),
    ("postgres://u:p@h/x", "postgresql+psycopg://u:p@h/x"),
    ("postgresql://u:p@h/x", "postgresql+psycopg://u:p@h/x"),
    ("sqlite:///./a.db", "sqlite:///./a.db"),
])
def test_database_urls_use_installed_sync_driver(url, expected):
    """Regression: the api extra installed no driver for postgresql:// URLs."""
    from sepsis_vitals.db import sync_database_url

    assert sync_database_url(url) == expected
