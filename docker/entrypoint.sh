#!/bin/sh
# Apply database migrations, then start the given command (uvicorn by default).
# Set SEPSIS_RUN_MIGRATIONS=false when migrations run as a separate one-off task
# (e.g. multiple ECS replicas).
set -e
if [ "${SEPSIS_RUN_MIGRATIONS:-true}" = "true" ]; then
    python -m alembic upgrade head
fi
exec "$@"
