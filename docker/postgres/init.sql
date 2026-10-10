-- Sepsis Vitals — database bootstrap (first docker-compose up only)
--
-- Extensions only. The schema is owned by Alembic and applied by the API
-- container on start (docker/entrypoint.sh -> alembic upgrade head).
-- Creating tables here previously produced a stale schema without the
-- blind-index columns, which broke login against the compose stack.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
