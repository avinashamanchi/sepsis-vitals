#!/usr/bin/env python3
"""Migration and application smoke test against a real PostgreSQL database.

Run by the ``postgres-migrations`` CI job (see .github/workflows/ci.yml).
Requires DATABASE_URL (an empty Postgres database with the extensions from
docker/postgres/init.sql), SEPSIS_PII_KEY and SEPSIS_JWT_SECRET.

Checks:
1. ``alembic upgrade head`` / ``downgrade`` / ``upgrade`` on an empty database.
2. Migration 003 backfills blind indexes for rows written by the old schema.
3. The application can register, log in, store a long (encrypted) MRN, and
   register the same MRN at two sites, but not twice at one site.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_REVISION = "b2c3d4e5f6a7"


def _alembic(*args: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    getattr(command, args[0])(cfg, *args[1:])


def main() -> int:
    for var in ("DATABASE_URL", "SEPSIS_PII_KEY", "SEPSIS_JWT_SECRET"):
        if not os.getenv(var):
            print(f"{var} must be set", file=sys.stderr)
            return 2

    import sqlalchemy as sa

    from sepsis_vitals.db import sync_database_url
    from sepsis_vitals.security import FieldEncryptor, compute_blind_index

    engine = sa.create_engine(sync_database_url(os.environ["DATABASE_URL"]))

    # 1. Round-trip the migrations on an empty database.
    _alembic("upgrade", "head")
    _alembic("downgrade", LEGACY_REVISION)

    # 2. Seed rows as the pre-003 schema stored them, then upgrade.
    enc = FieldEncryptor.get()
    legacy_email = "Legacy.Nurse@Example.org"
    legacy_mrn = "MRN-LEGACY-1"
    user_id, patient_id = str(uuid.uuid4()), str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, email, password_hash, role, site_id) "
                "VALUES (:id, :email, 'x', 'nurse', 'SITE-A')"
            ),
            {"id": user_id, "email": enc.encrypt(legacy_email)},
        )
        conn.execute(
            sa.text("INSERT INTO patients (id, external_id, site_id) VALUES (:id, :mrn, 'SITE-A')"),
            {"id": patient_id, "mrn": enc.encrypt(legacy_mrn)},
        )
    _alembic("upgrade", "head")
    with engine.connect() as conn:
        email_hash = conn.execute(
            sa.text("SELECT email_hash FROM users WHERE id = :id"), {"id": user_id}
        ).scalar()
        mrn_hash = conn.execute(
            sa.text("SELECT external_id_hash FROM patients WHERE id = :id"), {"id": patient_id}
        ).scalar()
    assert email_hash == compute_blind_index(legacy_email), "email_hash not backfilled"
    assert mrn_hash == compute_blind_index(legacy_mrn), "external_id_hash not backfilled"
    print("backfill: ok")

    # 3. Application smoke on the migrated schema.
    from sepsis_vitals.auth.service import login_user, register_user
    from sepsis_vitals.db import SessionLocal
    from sepsis_vitals.patients import service

    db = SessionLocal()
    try:
        email = f"ci-{uuid.uuid4().hex[:8]}@example.org"
        register_user(email, "Correct-Horse-Battery-9!", "nurse", "SITE-A", db)
        assert login_user(email, "Correct-Horse-Battery-9!", db)["access_token"]
        long_mrn = "MRN-" + "7" * 28  # ciphertext > 64 chars: overflowed VARCHAR(64)
        service.create_patient(long_mrn, "SITE-A", 50, "F", db)
        service.create_patient(long_mrn, "SITE-B", 61, "M", db)  # same MRN, other site
        try:
            service.create_patient(long_mrn, "SITE-A", 50, "F", db)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate MRN accepted within one site")
    finally:
        db.close()
    print("application smoke: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
