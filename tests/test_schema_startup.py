"""
tests/test_schema_startup.py — who creates the schema at startup, and what
happens when several workers start at once or the schema is wrong.

The full "API replicas start before the migration task" sequence needs
PostgreSQL (the migrations use INET/JSONB) and runs in the CI
``postgres-migrations`` job; here the decision logic is tested on SQLite.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import sqlalchemy as sa
import sqlalchemy.exc as sa_exc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def scratch_engine(monkeypatch, tmp_path):
    """Point sepsis_vitals.db at an empty SQLite file for one test."""
    from sepsis_vitals import db

    engine = sa.create_engine(f"sqlite:///{tmp_path}/startup.db")
    monkeypatch.setattr(db, "engine", engine)
    yield engine
    engine.dispose()


def _tables(engine) -> set:
    return set(sa.inspect(engine).get_table_names())


# -- the race -----------------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("table users already exists", True),                                   # SQLite
    ("index ix_vitals_patient already exists", True),                       # SQLite
    ('relation "users" already exists', True),                              # PostgreSQL
    ('duplicate key value violates unique constraint "pg_type_typname_nsp_index"', True),
    ('duplicate key value violates unique constraint "users_email_hash_key"', False),
    ("permission denied for schema public", False),
    ("could not connect to server: Connection refused", False),
    ("disk I/O error", False),
])
def test_only_the_concurrent_create_error_is_retried(message, expected):
    from sepsis_vitals.db import _is_concurrent_create

    exc = sa_exc.OperationalError("CREATE TABLE users", {}, Exception(message))
    assert _is_concurrent_create(exc) is expected


def test_retries_are_bounded(scratch_engine, monkeypatch):
    from sepsis_vitals import db

    calls = {"n": 0}

    def always_racing(*args, **kwargs):
        calls["n"] += 1
        raise sa_exc.OperationalError("CREATE TABLE users", {}, Exception("table users already exists"))

    monkeypatch.setattr(db.Base.metadata, "create_all", always_racing)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(sa_exc.OperationalError):
        db.init_db(production=False, attempts=3)
    assert calls["n"] == 3


def test_concurrent_startup_on_a_fresh_database(tmp_path):
    """Six processes released together into init_db, three times, each round
    on an empty database. Without the retry most workers fail every round
    (``scripts/ci_concurrent_startup.py --attempts 1``)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci_concurrent_startup.py"),
         "--database-url", f"sqlite:///{tmp_path}/race.db", "--workers", "6", "--rounds", "3"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert "3/3 rounds clean" in result.stdout


# -- schema ownership -------------------------------------------------------------

def test_production_never_creates_an_unmanaged_schema(scratch_engine, caplog):
    """Creating tables before Alembic would make `alembic upgrade head` fail later."""
    from sepsis_vitals import db

    with caplog.at_level(logging.ERROR, logger="sepsis_vitals.db"):
        db.init_db(production=True)
    assert _tables(scratch_engine) == set()
    assert "not managed by Alembic" in caplog.text


def test_production_adds_only_missing_tables_to_a_migrated_database(scratch_engine):
    from sepsis_vitals import db

    with scratch_engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
    db.init_db(production=True)
    assert {t.name for t in db.Base.metadata.sorted_tables} <= _tables(scratch_engine)


def test_partially_created_schema_is_completed(scratch_engine):
    from sepsis_vitals import db

    db.Base.metadata.tables["users"].create(scratch_engine)
    db.init_db(production=False)
    assert {t.name for t in db.Base.metadata.sorted_tables} <= _tables(scratch_engine)
    assert db.schema_drift(scratch_engine) == []


def test_out_of_date_table_stops_startup_without_revealing_data(scratch_engine):
    """create_all never alters existing tables; a stale table used to surface
    as a 500 at the first login (N5). Now startup stops and names the columns."""
    from sepsis_vitals import db

    with scratch_engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id VARCHAR(36) PRIMARY KEY, email TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, email) VALUES ('u1', 'nurse.a@example.org')"))
    with pytest.raises(db.SchemaMismatchError) as err:
        db.init_db(production=False)
    message = str(err.value)
    assert "users.email_hash" in message and "alembic upgrade head" in message
    assert "nurse.a@example.org" not in message


# -- logging in the server process ------------------------------------------------

def test_audit_lines_reach_stdout_when_the_server_configures_no_logging(tmp_path):
    """Under uvicorn nothing configured the root logger, so INFO records such
    as HIPAA_AUDIT were dropped by Python's last-resort handler."""
    import subprocess
    import sys

    code = (
        "import logging\n"
        "from sepsis_vitals.logging_config import configure_logging\n"
        "logging.getLogger('sepsis_vitals.api').info('HIPAA_AUDIT dropped')\n"
        "assert configure_logging() is True\n"
        "assert configure_logging() is False  # idempotent\n"
        "logging.getLogger('sepsis_vitals.api').info('HIPAA_AUDIT kept')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PATH": "", "DATABASE_URL": f"sqlite:///{tmp_path}/x.db"})
    assert out.returncode == 0, out.stderr
    assert "HIPAA_AUDIT kept" in out.stdout and out.stdout.count("HIPAA_AUDIT") == 1


def test_logging_is_left_alone_when_the_host_configured_it():
    import logging

    from sepsis_vitals.logging_config import configure_logging

    assert logging.getLogger().handlers, "pytest installs root handlers"
    assert configure_logging() is False
