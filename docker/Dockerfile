# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for cryptography / psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libffi-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies come from the hashed lock for [api,ml] (requirements/deploy.txt,
# see docs/dependencies.md): every package and version is pinned and verified.
COPY requirements/deploy.txt requirements/deploy.txt
RUN pip install --no-cache-dir --require-hashes -r requirements/deploy.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip wheel --no-deps --wheel-dir /wheels . && \
    pip check


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Sepsis Vitals API"
LABEL org.opencontainers.image.description="Vitals-only sepsis prediction for LMIC hospitals"
LABEL org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN groupadd -r sepsis && useradd -r -g sepsis -d /app -s /bin/false sepsis

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /wheels /wheels

COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/sepsis-entrypoint

# Install the package wheel (fail the build if it does not install)
RUN pip install --no-cache-dir --no-index --no-deps --find-links=/wheels sepsis_vitals && pip check

# Model artifacts are mounted read-only at /models (never baked into the
# image); mutable patient state lives in /app/state (mount an encrypted volume).
RUN mkdir -p /app/state /models && chown -R sepsis:sepsis /app && chmod 0700 /app/state
USER sepsis

# Health check — FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

# Security: read-only filesystem where possible
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SEPSIS_ENV=production \
    SEPSIS_MODEL_DIR=/models \
    SEPSIS_STATE_DIR=/app/state

ENTRYPOINT ["/usr/local/bin/sepsis-entrypoint"]
CMD ["uvicorn", "sepsis_vitals.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--no-access-log"]
