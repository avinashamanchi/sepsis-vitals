"""Alembic migration environment for Sepsis Vitals."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

# Arbitrary constant identifying the sepsis-vitals migration lock.
_MIGRATION_LOCK_KEY = 72_904_311

config = context.config

if config.config_file_name is not None:
    # Keep loggers created before migrations run (in-process alembic calls
    # would otherwise silence the application's own loggers).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Override sqlalchemy.url from environment variable if set
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # Same synchronous driver as the application (psycopg v3)
    from sepsis_vitals.db import sync_database_url

    config.set_main_option("sqlalchemy.url", sync_database_url(database_url))

try:
    from sepsis_vitals.db import Base
    target_metadata = Base.metadata
except ImportError:
    target_metadata = None


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        is_postgres = connection.dialect.name == "postgresql"
        if is_postgres:
            # Serialise concurrent upgrades (several API containers starting at
            # once): the session-level lock is held until released below, and
            # later runners find the schema already at head.
            connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_LOCK_KEY})
            connection.commit()
        try:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
        finally:
            if is_postgres:
                connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _MIGRATION_LOCK_KEY})
                connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
