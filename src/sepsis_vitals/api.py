"""
sepsis_vitals.api — Production FastAPI application.

Wires together all security, ML, real-time, and monitoring subsystems:
- JWT token authentication with RBAC
- Token-bucket rate limiting per IP
- Pydantic request/response validation
- WebSocket real-time alert streaming
- Anthropic AI clinical copilot
- Prometheus-compatible metrics
- Structured error handling
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from sepsis_vitals import __version__
from sepsis_vitals.scores import compute_scores
from sepsis_vitals.security import RateLimiter, RateLimitExceeded, SecurityAlertTracker, sanitise_string

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

_is_production = os.getenv("SEPSIS_ENV", "development") == "production"

# ---------------------------------------------------------------------------
# Trusted proxy configuration for X-Forwarded-For validation
# ---------------------------------------------------------------------------

_TRUSTED_PROXIES: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
_raw_trusted = os.getenv("TRUSTED_PROXIES", "")
if _raw_trusted:
    for cidr in _raw_trusted.split(","):
        cidr = cidr.strip()
        if cidr:
            try:
                _TRUSTED_PROXIES.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Lifespan — database init and router wiring at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Startup/shutdown lifecycle for the FastAPI application."""
    _init_database()
    _include_routers()

    # Start PSI drift monitor background task
    from sepsis_vitals.monitoring.drift_monitor import get_drift_monitor
    drift_monitor = get_drift_monitor()
    await drift_monitor.start()

    yield

    await drift_monitor.stop()


app = FastAPI(
    title="Sepsis Vitals API",
    version=__version__,
    description=(
        "Investigational sepsis-model research API for retrospective and "
        "prospective silent-mode evaluation; not for patient care"
    ),
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
    lifespan=_lifespan,
)

_raw_origins = os.getenv(
    "SEPSIS_ALLOWED_ORIGINS",
    "http://localhost:8000,http://localhost:3000,https://avinashamanchi.github.io",
)
# Parse, strip whitespace, drop empty strings, reject wildcard with credentials
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.replace(" ", ",").split(",") if o.strip()]
if "*" in ALLOWED_ORIGINS:
    logger.warning("CORS wildcard '*' with allow_credentials is insecure — removing wildcard")
    ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# 10 req/s burst 20 for general API, 2 req/s burst 5 for expensive ML predict
_api_limiter = RateLimiter(rate=10.0, burst=20)
_ml_limiter = RateLimiter(rate=2.0, burst=5)
_auth_limiter = RateLimiter(rate=3.0, burst=10)   # Auth: 3/s burst 10 (brute-force protection)
_copilot_limiter = RateLimiter(rate=0.5, burst=3)
_billing_limiter = RateLimiter(rate=1.0, burst=3)  # Stripe mutations: 1/s
_webhook_limiter = RateLimiter(rate=5.0, burst=10)  # Stripe webhooks: 5/s


def _client_ip(request: Request) -> str:
    """Extract the real client IP, only trusting X-Forwarded-For when the
    immediate client is in ``TRUSTED_PROXIES``."""
    direct_ip = request.client.host if request.client else "unknown"
    if direct_ip == "unknown":
        return direct_ip

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and _TRUSTED_PROXIES:
        try:
            addr = ipaddress.ip_address(direct_ip)
            if any(addr in net for net in _TRUSTED_PROXIES):
                return forwarded.split(",")[0].strip()
        except ValueError:
            pass
    elif forwarded and not _TRUSTED_PROXIES:
        # No trusted proxies configured — fall back to direct IP
        return direct_ip

    return direct_ip


async def check_rate_limit(request: Request) -> None:
    """General API rate limit — dependency for most endpoints."""
    ip = _client_ip(request)
    if not _api_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again shortly.",
        )


async def check_ml_rate_limit(request: Request) -> None:
    """ML prediction rate limit — more restrictive."""
    ip = _client_ip(request)
    if not _ml_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="ML prediction rate limit exceeded. Max 2 requests/second.",
        )


async def check_auth_rate_limit(request: Request) -> None:
    """Auth endpoint rate limit — brute-force protection."""
    ip = _client_ip(request)
    if not _auth_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="Too many auth requests. Try again shortly.",
        )


# ---------------------------------------------------------------------------
# Authentication (JWT with short-lived access tokens + RBAC)
# ---------------------------------------------------------------------------

_auth_enabled = os.getenv("SEPSIS_AUTH_ENABLED", "true").lower() == "true"
if _is_production and not _auth_enabled:
    logger.warning(
        "SEPSIS_AUTH_ENABLED=false is ignored in production — forcing auth on"
    )
    _auth_enabled = True


def _anonymous_user() -> Dict[str, Any]:
    """Return a synthetic admin user dict when auth is disabled (dev only)."""
    return {"id": "anonymous", "email": "dev@localhost", "role": "system_admin", "org_id": None}


async def verify_auth(request: Request) -> Dict[str, Any]:
    """Verify JWT access token from Authorization header.

    Uses the real JWT middleware (short-lived HS256 tokens issued by
    /auth/login) when auth is enabled.  Falls back to an anonymous
    system_admin identity when SEPSIS_AUTH_ENABLED=false (development only).
    """
    if not _auth_enabled:
        return _anonymous_user()

    try:
        from sepsis_vitals.auth.middleware import get_current_user
        from sepsis_vitals.db import get_db

        # Resolve the DB session dependency manually since we're not in
        # a standard Depends() chain for this legacy shim.
        db_gen = get_db()
        db = next(db_gen)
        try:
            return get_current_user(request, db)
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    except ImportError:
        if _is_production:
            logger.critical("Auth middleware not available in production — rejecting request")
            raise HTTPException(
                status_code=500,
                detail="Authentication service unavailable",
            )
        logger.warning("Auth middleware not available — falling back to anonymous (dev only)")
        return _anonymous_user()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auth verification failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_patient_org(patient_id: str, user: Dict[str, Any], db) -> None:
    """Verify that the patient belongs to the requesting user's org.

    Delegates to :mod:`sepsis_vitals.auth.scope`: only ``system_admin``
    (including the anonymous dev identity when auth is disabled) is
    unscoped. Every other user must have a site assignment that matches the
    patient's ``site_id``; otherwise HTTP 404 is raised so existence at
    another site is not disclosed.
    """
    from sepsis_vitals.auth.scope import is_unscoped, load_patient_for_user

    if is_unscoped(user):
        return  # system_admin, incl. the auth-disabled dev identity
    load_patient_for_user(patient_id, user, db)


def require_role_dep(*roles: str):
    """Dependency factory that ensures the current user has one of the given roles."""
    allowed = set(roles)

    async def _check(user: Dict = Depends(verify_auth)) -> Dict[str, Any]:
        if user.get("role") not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {', '.join(sorted(allowed))}",
            )
        return user

    return _check


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class VitalsInput(BaseModel):
    temperature: Optional[float] = Field(None, ge=25.0, le=45.0, description="Body temperature in °C")
    heart_rate: Optional[float] = Field(None, ge=0, le=350, description="Heart rate in bpm")
    resp_rate: Optional[float] = Field(None, ge=0, le=80, description="Respiratory rate /min")
    sbp: Optional[float] = Field(None, ge=30, le=300, description="Systolic blood pressure mmHg")
    dbp: Optional[float] = Field(None, ge=20, le=200, description="Diastolic blood pressure mmHg")
    spo2: Optional[float] = Field(None, ge=0, le=100, description="Oxygen saturation %")
    gcs: Optional[float] = Field(None, ge=3, le=15, description="Glasgow Coma Scale")
    map: Optional[float] = Field(None, ge=20, le=200, description="Mean arterial pressure mmHg")
    lactate: Optional[float] = Field(None, ge=0, le=30, description="Serum lactate mmol/L")
    wbc: Optional[float] = Field(None, ge=0, le=100, description="White blood cell count x10^9/L")
    procalcitonin: Optional[float] = Field(None, ge=0, le=200, description="Procalcitonin ng/mL")
    on_supplemental_o2: Optional[bool] = Field(
        None, description="Receiving supplemental oxygen (NEWS2 adds 2 points)"
    )
    spo2_scale2: Optional[bool] = Field(
        None, description="Use NEWS2 SpO2 Scale 2 (prescribed 88-92% target only)"
    )


_NEWS2_FLAGS = ("on_supplemental_o2", "spo2_scale2")


def _count_measurements(vitals: Dict[str, Any]) -> int:
    """Number of measured values, excluding NEWS2 context flags."""
    return sum(1 for k in vitals if k not in _NEWS2_FLAGS)


class ComorbidityInput(BaseModel):
    has_hypertension: int = Field(0, ge=0, le=1)
    has_diabetes: int = Field(0, ge=0, le=1)
    has_ckd: int = Field(0, ge=0, le=1)
    has_copd: int = Field(0, ge=0, le=1)
    has_heart_failure: int = Field(0, ge=0, le=1)


class PredictRequest(BaseModel):
    vitals: VitalsInput
    patient_id: str = Field("unknown", max_length=100)
    age_years: Optional[int] = Field(None, ge=0, le=120)
    comorbidities: Optional[ComorbidityInput] = None


class BatchPredictRequest(BaseModel):
    patients: List[PredictRequest] = Field(..., max_length=10)


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: str
    risk_probability: float
    risk_level: str
    confidence_interval: ConfidenceInterval
    alert: bool
    clinical_scores: Dict[str, Any]
    top_risk_factors: List[Dict[str, Any]]
    recommendation: str
    model: Dict[str, str]
    research_only: bool = True
    validation_status: str = "Synthetic development baseline; no clinical validation"
    intended_use: str = "Retrospective research and prospective silent-mode evaluation"


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    model_loaded: bool
    model_name: Optional[str]
    auth_enabled: bool
    websocket_connections: int


class ScoreResponse(BaseModel):
    qsofa: int
    sirs_count: int
    news2_style: int
    shock_index: Optional[float]
    uva: int
    risk_level: str
    alert_flag: bool
    explanations: List[str]


class CopilotRequest(BaseModel):
    vitals: VitalsInput
    patient_id: str = Field("unknown", max_length=100)
    age_years: Optional[int] = Field(None, ge=0, le=120)
    comorbidities: Optional[ComorbidityInput] = None
    question: Optional[str] = Field(None, max_length=500, description="Optional clinical question")


class CopilotResponse(BaseModel):
    analysis: str
    risk_level: str
    key_concerns: List[str]
    suggested_actions: List[str]
    disclaimer: str


# Monitor / simulator request models
class MonitorRegisterRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=100, description="Patient identifier")
    demographics: Optional[Dict[str, Any]] = Field(None, description="Patient demographics")
    comorbidities: Optional[Dict[str, Any]] = Field(None, description="Patient comorbidities")


class SimulatorWardRequest(BaseModel):
    n_patients: int = Field(8, ge=1, le=50, description="Number of patients")
    speed: int = Field(360, ge=1, le=3600, description="Simulation speed multiplier")
    sepsis_count: int = Field(2, ge=0, le=50, description="Number of sepsis patients")
    seed: int = Field(42, ge=0, description="Random seed")


class SimulatorReplayRequest(BaseModel):
    subject_id: Optional[str] = Field(None, max_length=100, description="MIMIC subject ID or 'random'")
    speed: int = Field(720, ge=1, le=3600, description="Replay speed multiplier")
    sepsis_only: bool = Field(False, description="Only select sepsis cases")


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from sepsis_vitals.ml.predictor import SepsisPredictor
        _predictor = SepsisPredictor()
        try:
            _predictor.load()
        except FileNotFoundError:
            _predictor = None
            return None
    return _predictor


# ---------------------------------------------------------------------------
# Monitor singleton (lazy initialization)
# ---------------------------------------------------------------------------

_monitor_registry = None
_monitor_tracker = None
_monitor_ingester = None


def _get_monitor_components():
    """Lazy-initialize the monitoring components."""
    global _monitor_registry, _monitor_tracker, _monitor_ingester

    if _monitor_registry is None:
        from sepsis_vitals.ml.monitor import (
            PatientRegistry,
            DeteriorationTracker,
            VitalsIngester,
        )

        _monitor_registry = PatientRegistry()
        _monitor_tracker = DeteriorationTracker()

        predictor = _get_predictor()
        if predictor is not None:
            _monitor_ingester = VitalsIngester(
                predictor=predictor,
                registry=_monitor_registry,
                tracker=_monitor_tracker,
                ws_manager=ws_manager,
            )

    return _monitor_registry, _monitor_tracker, _monitor_ingester


# ---------------------------------------------------------------------------
# Simulator (gated behind ENABLE_SIMULATOR=true)
# ---------------------------------------------------------------------------

_simulator_enabled = os.getenv("ENABLE_SIMULATOR", "false").lower() == "true"
_simulation_manager = None


def _get_simulation_manager():
    """Lazy-initialize the SimulationManager."""
    global _simulation_manager
    if _simulation_manager is None:
        from sepsis_vitals.ml.simulator import SimulationManager
        _simulation_manager = SimulationManager()
    return _simulation_manager


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

_metrics: Dict[str, Any] = {
    "requests_total": 0,
    "predictions_total": 0,
    "alerts_total": 0,
    "errors_total": 0,
    "copilot_calls_total": 0,
    "rate_limited_total": 0,
    "avg_prediction_ms": 0.0,
    "_prediction_times": [],
}


def _track_prediction(duration_ms: float, alert: bool):
    _metrics["predictions_total"] += 1
    if alert:
        _metrics["alerts_total"] += 1
    times = _metrics["_prediction_times"]
    times.append(duration_ms)
    if len(times) > 100:
        times.pop(0)
    _metrics["avg_prediction_ms"] = sum(times) / len(times)


def _persist_prediction(
    patient_id: str,
    user_id: Optional[str],
    result: Dict[str, Any],
    input_vitals: Dict[str, Any],
    ip_address: str,
    model_version: Optional[str] = None,
) -> None:
    """Write prediction to Postgres as immutable audit record.

    Runs synchronously in a background-safe manner.  Failures are logged
    but never block the API response — the prediction is the priority.
    """
    try:
        from sepsis_vitals.db import SessionLocal, PredictionRecord
        db = SessionLocal()
        try:
            ci = result.get("confidence_interval", {})
            record = PredictionRecord(
                patient_id=patient_id,
                user_id=user_id if user_id != "anonymous" else None,
                risk_probability=result["risk_probability"],
                risk_level=result["risk_level"],
                alert_fired=result.get("alert", False),
                input_vitals=json.dumps(input_vitals),
                output_scores=json.dumps(result.get("clinical_scores", {})),
                top_risk_factors=json.dumps(result.get("top_risk_factors", [])),
                confidence_lower=ci.get("lower"),
                confidence_upper=ci.get("upper"),
                model_version=model_version,
                recommendation=result.get("recommendation"),
                ip_address=ip_address,
            )
            db.add(record)
            db.commit()
        except Exception:
            db.rollback()
            logger.warning("Failed to persist prediction record", exc_info=True)
        finally:
            db.close()
    except ImportError:
        pass  # DB module not available (e.g., minimal install)


# ---------------------------------------------------------------------------
# WebSocket manager (real-time alerts)
# ---------------------------------------------------------------------------

from sepsis_vitals.realtime.websocket import manager as ws_manager


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    _metrics["rate_limited_total"] += 1
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)},
        headers={"Retry-After": "5"},
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    _metrics["errors_total"] += 1
    # Don't leak internal errors
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Contact support."},
    )


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self'; frame-ancestors 'none'; "
        "base-uri 'self'; form-action 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# ---------------------------------------------------------------------------
# Request counting middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def count_requests(request: Request, call_next):
    _metrics["requests_total"] += 1
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# HIPAA Audit Controls — 45 CFR § 164.312(b)
# ---------------------------------------------------------------------------
# Structured, append-only audit log for every access to PHI.
# Answers: "Who accessed what patient data, when, from where?"

# Path patterns that access PHI and require audit logging
_PHI_AUDIT_PATTERNS = {
    "/predict": "ml_prediction",
    "/predict/batch": "ml_prediction_batch",
    "/patient/": "view_patient_data",
    "/copilot": "copilot_query",
    "/score": "score_calculation",
}


def _emit_audit_event(
    action: str,
    user_id: Optional[str],
    ip_address: str,
    path: str,
    patient_id: Optional[str] = None,
    status_code: int = 200,
) -> None:
    """Write a HIPAA audit event to both structured log and database.

    Events are emitted as structured JSON to stdout (for log aggregators
    like Datadog, Splunk, or CloudWatch) AND persisted to the audit_log
    table in Postgres for regulatory queries.
    """
    event = {
        "audit": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "action": action,
        "resource_type": "patient_phi",
        "resource_id": patient_id,
        "path": path,
        "ip_address": ip_address,
        "status_code": status_code,
    }

    # Structured log output (picked up by log aggregators)
    logger.info("HIPAA_AUDIT %s", json.dumps(event))

    # Persist to database (best-effort, never blocks)
    try:
        from sepsis_vitals.db import SessionLocal, AuditLog
        db = SessionLocal()
        try:
            record = AuditLog(
                user_id=user_id if user_id and user_id != "anonymous" else None,
                action=action,
                resource_type="patient_phi",
                resource_id=patient_id,
                details=json.dumps({"path": path, "status_code": status_code}),
                ip_address=ip_address,
            )
            db.add(record)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
    except ImportError:
        pass


@app.middleware("http")
async def hipaa_audit_middleware(request: Request, call_next):
    """Log all PHI access for HIPAA § 164.312(b) compliance.

    Also feeds the security alert tracker for anomaly detection:
    - Failed auth attempts (401s on auth endpoints)
    - Bulk PHI access per user
    - Error bursts per IP
    """
    response = await call_next(request)

    path = request.url.path
    ip = _client_ip(request)
    tracker = SecurityAlertTracker.get()

    # Security alerting: track errors and failed auth
    if response.status_code >= 400:
        tracker.record_error_burst(ip, response.status_code)
    if response.status_code == 401 and path.startswith("/auth/"):
        tracker.record_failed_auth(ip)

    # PHI audit logging
    audit_action = None
    for pattern, action in _PHI_AUDIT_PATTERNS.items():
        if path.startswith(pattern) or (pattern.endswith("/") and pattern[:-1] in path):
            audit_action = action
            break

    if audit_action:
        # Extract user_id from request state if available
        user_id = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from sepsis_vitals.auth.tokens import decode_token
                payload = decode_token(auth_header[7:])
                user_id = payload.get("sub")
            except Exception:
                user_id = "unauthenticated"

        # Security alerting: track bulk PHI access per user
        if user_id and user_id != "unauthenticated":
            tracker.record_phi_access(user_id, ip)

        # Extract patient_id from path if present
        patient_id = None
        if "/patient/" in path:
            parts = path.split("/patient/")
            if len(parts) > 1:
                patient_id = parts[1].split("/")[0]

        await asyncio.to_thread(
            _emit_audit_event,
            action=audit_action,
            user_id=user_id,
            ip_address=ip,
            path=path,
            patient_id=patient_id,
            status_code=response.status_code,
        )

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Minimal health check. Sensitive details hidden in production."""
    predictor = _get_predictor()
    if _is_production:
        return {"status": "ok", "version": __version__, "timestamp": time.time()}
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=time.time(),
        model_loaded=predictor is not None,
        model_name=predictor.metadata["model_name"] if predictor and predictor.metadata else None,
        auth_enabled=_auth_enabled,
        websocket_connections=ws_manager.active_connections,
    )


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(check_rate_limit)])
async def score_vitals(vitals: VitalsInput, user: Dict = Depends(verify_auth)):
    """Compute clinical sepsis scores (qSOFA, SIRS, NEWS2, Shock Index, UVA)."""
    vitals_dict = {k: v for k, v in vitals.model_dump().items() if v is not None}
    if _count_measurements(vitals_dict) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 vital signs.")
    result = compute_scores(vitals_dict)
    flag_explanations = {
        "qsofa_rr": "Respiratory rate meets the qSOFA criterion.",
        "qsofa_gcs": "Glasgow Coma Scale meets the qSOFA criterion.",
        "qsofa_sbp": "Systolic blood pressure meets the qSOFA criterion.",
        "sirs_temp": "Temperature meets a SIRS criterion.",
        "sirs_hr": "Heart rate meets a SIRS criterion.",
        "sirs_rr": "Respiratory rate meets a SIRS criterion.",
    }
    explanations = [
        flag_explanations[name]
        for name, fired in result.component_flags.items()
        if fired and name in flag_explanations
    ]
    return ScoreResponse(
        qsofa=result.qsofa,
        sirs_count=result.sirs_count,
        news2_style=result.news2_style,
        shock_index=result.shock_index,
        uva=result.uva_style,
        risk_level=result.risk_level,
        alert_flag=result.alert_flag,
        explanations=explanations,
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def predict_sepsis(body: PredictRequest, request: Request, user: Dict = Depends(verify_auth)):
    """ML-powered sepsis risk prediction with SHAP explanations."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run 'python -m sepsis_vitals.train' first.")

    vitals_dict = {k: v for k, v in body.vitals.model_dump().items() if v is not None}
    if _count_measurements(vitals_dict) < 3:
        raise HTTPException(status_code=422, detail="Provide at least 3 vital signs for ML prediction.")

    comorbidities = body.comorbidities.model_dump() if body.comorbidities else None
    await _ensure_not_foreign_patient_async(sanitise_string(body.patient_id), user)
    history = await _recorded_history_async(sanitise_string(body.patient_id))

    start = time.monotonic()
    prediction = predictor.predict(
        vitals=vitals_dict,
        patient_id=sanitise_string(body.patient_id),
        age_years=body.age_years,
        comorbidities=comorbidities,
        history=history,
    )
    elapsed_ms = (time.monotonic() - start) * 1000

    result = prediction.to_dict()
    _track_prediction(elapsed_ms, result.get("alert", False))

    # ── Drift monitoring — record vitals into rolling PSI buffer ──────
    from sepsis_vitals.monitoring.drift_monitor import get_drift_monitor
    get_drift_monitor().record_prediction(vitals_dict)

    # ── Permanent audit trail (Postgres) ─────────────────────────────
    # Redis handles ephemeral rolling windows; Postgres is the immutable
    # ledger.  Every prediction is recorded so legal/risk-management can
    # reconstruct the decision history for any patient at any time.
    await asyncio.to_thread(
        _persist_prediction,
        patient_id=body.patient_id,
        user_id=user.get("id"),
        result=result,
        input_vitals=vitals_dict,
        ip_address=_client_ip(request),
        model_version=predictor.metadata.get("version") if predictor.metadata else None,
    )

    # Broadcast alert via WebSocket if high/critical
    if result.get("alert"):
        await ws_manager.broadcast({
            "type": "sepsis_alert",
            "patient_id": body.patient_id,
            "risk_level": result["risk_level"],
            "risk_probability": result["risk_probability"],
            "recommendation": result["recommendation"],
            "timestamp": result["timestamp"],
        })

    return result


@app.post("/predict/batch", dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def predict_batch(body: BatchPredictRequest, request: Request, user: Dict = Depends(verify_auth)):
    """Batch prediction for multiple patients (max 10)."""
    # The check_ml_rate_limit dependency already consumed 1 token.
    # Consume additional tokens proportional to batch size so that a batch
    # of N patients costs N tokens, preventing rate-limit bypass via batching.
    ip = _client_ip(request)
    extra_tokens_needed = len(body.patients) - 1
    for _ in range(extra_tokens_needed):
        if not _ml_limiter.allow(ip):
            raise HTTPException(
                status_code=429,
                detail="ML prediction rate limit exceeded. Reduce batch size or try again shortly.",
            )

    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    errors = []
    model_version = predictor.metadata.get("version") if predictor.metadata else None

    for patient in body.patients:
        await _ensure_not_foreign_patient_async(sanitise_string(patient.patient_id), user)

    for i, patient in enumerate(body.patients):
        try:
            vitals_dict = {k: v for k, v in patient.vitals.model_dump().items() if v is not None}
            comorbidities = patient.comorbidities.model_dump() if patient.comorbidities else None
            prediction = predictor.predict(
                vitals=vitals_dict,
                patient_id=sanitise_string(patient.patient_id),
                age_years=patient.age_years,
                comorbidities=comorbidities,
                history=await _recorded_history_async(sanitise_string(patient.patient_id)),
            )
            result = prediction.to_dict()
            results.append(result)

            # Persist each prediction for audit trail (HIPAA compliance)
            await asyncio.to_thread(
                _persist_prediction,
                patient_id=patient.patient_id,
                user_id=user.get("id"),
                result=result,
                input_vitals=vitals_dict,
                ip_address=ip,
                model_version=model_version,
            )
        except Exception as exc:
            logger.warning("Batch predict failed for patient %d (%s): %s", i, patient.patient_id, exc)
            errors.append({"index": i, "patient_id": patient.patient_id, "error": str(exc)})

    return {"predictions": results, "count": len(results), "errors": errors}


async def _ensure_not_foreign_patient_async(patient_id: str, user: Dict[str, Any]) -> None:
    """Block predictions written against another site's registered patient."""
    def _check():
        from sepsis_vitals.auth.scope import ensure_not_foreign_patient
        from sepsis_vitals.db import SessionLocal
        db = SessionLocal()
        try:
            ensure_not_foreign_patient(patient_id, user, db)
        finally:
            db.close()
    await asyncio.to_thread(_check)


async def _recorded_history_async(patient_id: str, limit: int = 4) -> list:
    """Recent recorded vitals for a registered patient, oldest first.

    Gives /predict the same deltas, rolling statistics and observation gap the
    model saw in training. Unregistered IDs have no history.
    """
    def _load() -> list:
        from sepsis_vitals.db import SessionLocal, VitalReading
        db = SessionLocal()
        try:
            rows = (
                db.query(VitalReading)
                .filter(VitalReading.patient_id == patient_id)
                .order_by(VitalReading.recorded_at.desc())
                .limit(limit)
                .all()
            )
            names = ("temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2",
                     "gcs", "lactate", "wbc", "procalcitonin")
            history = [
                {"timestamp": r.recorded_at, "map": r.map_pressure,
                 **{n: getattr(r, n) for n in names}}
                for r in rows
            ]
            return list(reversed(history))
        finally:
            db.close()
    return await asyncio.to_thread(_load)


async def _verify_patient_org_async(patient_id: str, user: Dict[str, Any]) -> None:
    """Run :func:`verify_patient_org` in a worker thread with its own session."""
    def _check_org():
        from sepsis_vitals.db import SessionLocal
        db = SessionLocal()
        try:
            verify_patient_org(patient_id, user, db)
        finally:
            db.close()
    await asyncio.to_thread(_check_org)


@app.get("/patient/{patient_id}/trend", dependencies=[Depends(check_rate_limit)])
async def patient_trend(patient_id: str, request: Request, user: Dict = Depends(verify_auth)):
    """Get risk trend for a monitored patient."""
    # Org-level authorization: verify patient belongs to user's org (fail closed)
    await _verify_patient_org_async(patient_id, user)

    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    trend = predictor.get_patient_trend(sanitise_string(patient_id))
    if trend is None:
        raise HTTPException(status_code=404, detail=f"No data for patient {patient_id}")
    return trend


# ---------------------------------------------------------------------------
# Monitor endpoints (continuous patient monitoring)
# ---------------------------------------------------------------------------


@app.post("/monitor/register", dependencies=[Depends(check_rate_limit)])
async def monitor_register(body: MonitorRegisterRequest, user: Dict = Depends(verify_auth)):
    """Register a patient for continuous monitoring."""
    patient_id = sanitise_string(body.patient_id)
    await _verify_patient_org_async(patient_id, user)

    registry, tracker, ingester = _get_monitor_components()
    registry.register(
        patient_id,
        demographics=body.demographics,
        comorbidities=body.comorbidities,
    )

    return {"status": "registered", "patient_id": patient_id}


@app.delete("/monitor/{patient_id}", dependencies=[Depends(check_rate_limit)])
async def monitor_unregister(patient_id: str, user: Dict = Depends(verify_auth)):
    """Remove a patient from continuous monitoring."""
    # Org-level authorization: verify patient belongs to user's org (fail closed)
    await _verify_patient_org_async(patient_id, user)

    registry, tracker, ingester = _get_monitor_components()
    registry.unregister(sanitise_string(patient_id))
    tracker.remove_patient(sanitise_string(patient_id))

    return {"status": "unregistered", "patient_id": patient_id}


@app.get("/monitor/status", dependencies=[Depends(check_rate_limit)])
async def monitor_status(user: Dict = Depends(verify_auth)):
    """List all monitored patients with current risk and trend."""
    registry, tracker, ingester = _get_monitor_components()
    patients = registry.list_patients()

    from sepsis_vitals.auth.scope import require_site
    site = require_site(user)
    if site is not None:
        def _site_patient_ids() -> set:
            from sepsis_vitals.db import Patient, SessionLocal
            db = SessionLocal()
            try:
                return {row[0] for row in db.query(Patient.id).filter(Patient.site_id == site)}
            finally:
                db.close()
        allowed = await asyncio.to_thread(_site_patient_ids)
        patients = [p for p in patients if p["patient_id"] in allowed]

    # Enrich with deterioration data
    for p in patients:
        pid = p["patient_id"]
        det = tracker.evaluate(pid)
        p["alert_state"] = det.get("alert_state", "normal")
        p["deterioration_rate"] = det.get("deterioration_rate_per_hour", 0.0)
        p["window_hours"] = det.get("window_hours", 0.0)

    return {"patients": patients, "count": len(patients)}


# ---------------------------------------------------------------------------
# Simulator endpoints (gated behind ENABLE_SIMULATOR=true)
# ---------------------------------------------------------------------------


@app.post("/simulator/ward", dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def simulator_start_ward(body: SimulatorWardRequest, user: Dict = Depends(verify_auth)):
    """Start a synthetic ward simulation."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    _, _, ingester = _get_monitor_components()
    if ingester is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    manager = _get_simulation_manager()
    session_id = manager.start_ward(
        ingester=ingester,
        n_patients=body.n_patients,
        speed=body.speed,
        sepsis_count=body.sepsis_count,
        seed=body.seed,
    )

    return {"session_id": session_id, "status": "started"}


@app.post("/simulator/replay", dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def simulator_start_replay(body: SimulatorReplayRequest, user: Dict = Depends(verify_auth)):
    """Start a MIMIC-IV case replay."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    _, _, ingester = _get_monitor_components()
    if ingester is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    from sepsis_vitals.ml.case_library import CaseLibrary
    lib = CaseLibrary()

    if body.subject_id == "random" or body.subject_id is None:
        case_meta = lib.get_random_case(sepsis=body.sepsis_only if body.sepsis_only else None)
    else:
        case_meta = lib.get_case(subject_id=int(body.subject_id))

    if case_meta is None:
        raise HTTPException(status_code=404, detail="Case not found")

    # Load vitals timeline for this case
    from sepsis_vitals.ml.mimic_loader import MIMICLoader
    loader = MIMICLoader.from_demo()
    vitals = loader.load_vitals(stay_ids={case_meta["stay_id"]})

    manager = _get_simulation_manager()
    session_id = manager.start_replay(
        case_meta=case_meta,
        timeline=vitals,
        ingester=ingester,
        speed=body.speed,
    )

    return {"session_id": session_id, "subject_id": case_meta["subject_id"], "status": "started"}


@app.delete("/simulator/{session_id}", dependencies=[Depends(check_rate_limit)])
async def simulator_stop(session_id: str, user: Dict = Depends(verify_auth)):
    """Stop a simulation session."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    manager = _get_simulation_manager()
    stopped = manager.stop_session(sanitise_string(session_id))

    if not stopped:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "status": "stopped"}


@app.get("/simulator/sessions", dependencies=[Depends(check_rate_limit)])
async def simulator_sessions(user: Dict = Depends(verify_auth)):
    """List active simulation sessions."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    manager = _get_simulation_manager()
    return {"sessions": manager.list_sessions()}


@app.get("/simulator/cases", dependencies=[Depends(check_rate_limit)])
async def simulator_cases(user: Dict = Depends(verify_auth)):
    """List available MIMIC-IV cases for replay."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    from sepsis_vitals.ml.case_library import CaseLibrary
    lib = CaseLibrary()

    try:
        cases = lib.list_cases()
    except Exception:
        cases = []

    return {"cases": cases, "count": len(cases)}


@app.get("/model/info", dependencies=[Depends(check_rate_limit)])
async def model_info(user: Dict = Depends(verify_auth)):
    """Model metadata, performance metrics, and top features."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "model_name": predictor.metadata["model_name"],
        "version": predictor.metadata["version"],
        "is_calibrated": predictor.metadata.get("is_calibrated", False),
        "feature_count": len(predictor.feature_names),
        "training_data": predictor.metadata.get("model_card", {}).get("training_data", "Unknown"),
        "metrics": predictor.metadata.get("metrics", {}),
        "feature_importance": dict(list(
            predictor.metadata.get("feature_importance", {}).items()
        )[:15]),
    }


# ---------------------------------------------------------------------------
# AI Clinical Copilot (Anthropic-powered)
# ---------------------------------------------------------------------------

# The copilot is frozen by default until clinical validation and a human-factors
# review establish that it adds value without unsafe automation bias.
_copilot_enabled = os.getenv("SEPSIS_ENABLE_COPILOT", "false").lower() == "true"
# Enterprise LLM feature gate — separate opt-in, requires signed BAA.
_enterprise_llm_enabled = os.getenv("SEPSIS_ENTERPRISE_LLM", "false").lower() == "true"


def _deidentify_vitals(vitals: dict) -> dict:
    """Strip any patient-identifying information before sending to external LLM.

    Only numeric clinical measurements are sent. No names, MRNs, DOBs, or
    free-text fields cross the boundary.
    """
    safe_keys = {
        "temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2",
        "gcs", "map", "lactate", "wbc", "procalcitonin",
    }
    return {k: v for k, v in vitals.items() if k in safe_keys}


@app.post("/copilot", response_model=CopilotResponse, dependencies=[Depends(check_rate_limit)])
async def clinical_copilot(body: CopilotRequest, user: Dict = Depends(verify_auth)):
    """Research-only observation summary.

    Disabled by default. Enabling it requires an explicit feature flag; enabling
    external LLM processing additionally requires a signed BAA and separate flag.
    """
    if not _copilot_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Copilot is frozen for this investigational release pending "
                "clinical validation and human-factors review."
            ),
        )

    copilot_key = f"copilot:{user.get('user', user.get('email', 'anon'))}"
    if not _copilot_limiter.allow(copilot_key):
        raise HTTPException(status_code=429, detail="Copilot rate limit exceeded. Max 1 request per 2 seconds.")

    _metrics["copilot_calls_total"] += 1

    vitals_dict = {k: v for k, v in body.vitals.model_dump().items() if v is not None}
    scores = compute_scores(vitals_dict)
    scores_dict = scores.as_dict()

    # Get ML prediction if model loaded
    ml_risk = None
    predictor = _get_predictor()
    if predictor:
        comorbidities = body.comorbidities.model_dump() if body.comorbidities else None
        pred = predictor.predict(
            vitals=vitals_dict,
            patient_id=body.patient_id,
            age_years=body.age_years,
            comorbidities=comorbidities,
        )
        ml_risk = pred.to_dict()

    # LLM copilot: ONLY available under enterprise flag with BAA
    if _enterprise_llm_enabled:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                # Sanitise and check for prompt injection before LLM call
                safe_question = None
                if body.question:
                    from sepsis_vitals.security import check_prompt_injection, PromptInjectionError
                    try:
                        check_prompt_injection(body.question)
                    except PromptInjectionError:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid input detected in clinical question.",
                        )
                    safe_question = sanitise_string(body.question, max_length=500)
                safe_vitals = _deidentify_vitals(vitals_dict)
                analysis = await _anthropic_copilot(
                    safe_vitals, scores_dict, ml_risk, body.age_years, safe_question
                )
                return analysis
            except Exception:
                logger.warning("LLM copilot failed, falling back to rule-based", exc_info=True)

    # Default: deterministic rule-based analysis (legally safe, no hallucination risk)
    return _rule_based_copilot(vitals_dict, scores_dict, ml_risk, body.age_years)


async def _anthropic_copilot(
    vitals: dict, scores: dict, ml_risk: Optional[dict],
    age: Optional[int], question: Optional[str],
) -> CopilotResponse:
    """Call Anthropic Claude for clinical analysis."""
    import anthropic

    client = anthropic.Anthropic()

    risk_info = ""
    if ml_risk:
        risk_info = f"""
ML Model Prediction:
- Risk probability: {ml_risk['risk_probability']:.1%}
- Risk level: {ml_risk['risk_level']}
- Top risk factors: {json.dumps(ml_risk.get('top_risk_factors', [])[:3])}
"""

    prompt = f"""You summarize observations for an investigational sepsis-model validation study. Do not diagnose, prescribe, recommend treatment, or claim clinical benefit. Identify only the supplied score criteria, unusual measurements, missing data, and questions for a designated study reviewer.

Patient vitals: {json.dumps(vitals)}
Age: {age if age else 'Unknown'}
Clinical scores: qSOFA={scores.get('qsofa',0)}/3, SIRS={scores.get('sirs_count',0)}/3, NEWS2={scores.get('news2_style',0)}, Shock Index={scores.get('shock_index','N/A')}
Risk level: {scores.get('risk_level', 'unknown')}
{risk_info}
{f'Clinical question: {question}' if question else ''}

Respond in this exact JSON format:
{{
  "analysis": "2-3 sentence research observation summary",
  "risk_level": "low|moderate|high|critical",
  "key_concerns": ["concern1", "concern2"],
  "suggested_actions": ["data verification step", "study review step"]
}}

Be concise and precise. Suggested actions must be limited to data verification,
documentation, or review under the study protocol."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = getattr(message.content[0], "text", "")
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Enterprise LLM returned no text response")
    response_text = response_text.strip()
    # Extract JSON from response
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    parsed = json.loads(response_text)

    return CopilotResponse(
        analysis=parsed.get("analysis", "Analysis unavailable."),
        risk_level=parsed.get("risk_level", scores.get("risk_level", "unknown")),
        key_concerns=parsed.get("key_concerns", []),
        suggested_actions=parsed.get("suggested_actions", []),
        disclaimer=(
            "Investigational research summary. Not for diagnosis or treatment; "
            "review only under the approved study protocol."
        ),
    )


def _rule_based_copilot(
    vitals: dict, scores: dict, ml_risk: Optional[dict], age: Optional[int],
) -> CopilotResponse:
    """Produce a non-treatment research summary when external LLM use is off."""
    concerns: List[str] = []
    risk_level = scores.get("risk_level", "low")

    temp = vitals.get("temperature")
    if temp and (temp > 38.3 or temp < 36.0):
        concerns.append(f"Temperature ({temp}°C) meets an encoded score criterion.")

    hr = vitals.get("heart_rate")
    if hr and hr > 100:
        concerns.append(f"Heart rate ({hr} bpm) is above the encoded reference range.")
    elif hr and hr < 50:
        concerns.append(f"Heart rate ({hr} bpm) is below the encoded reference range.")

    rr = vitals.get("resp_rate")
    if rr and rr > 22:
        concerns.append(f"Respiratory rate ({rr}/min) meets the qSOFA criterion.")

    sbp = vitals.get("sbp")
    if sbp and sbp <= 100:
        concerns.append(f"Systolic blood pressure ({sbp} mmHg) meets the qSOFA criterion.")

    spo2 = vitals.get("spo2")
    if spo2 and spo2 < 94:
        concerns.append(f"SpO2 ({spo2}%) is below the encoded reference range.")

    gcs = vitals.get("gcs")
    if gcs and gcs < 15:
        concerns.append(f"GCS ({gcs}/15) meets the qSOFA criterion.")

    lactate = vitals.get("lactate")
    if lactate is not None and lactate >= 2.0:
        concerns.append(f"Lactate ({lactate} mmol/L) meets an encoded risk criterion.")

    qsofa = scores.get("qsofa", 0)
    sirs = scores.get("sirs_count", 0)
    ml_prob = ml_risk["risk_probability"] if ml_risk else None
    if ml_prob is not None:
        concerns.append(
            f"The unvalidated development model produced a {ml_prob:.0%} output."
        )

    if not concerns:
        concerns.append("No encoded score criteria fired in the supplied observations.")

    analysis = (
        f"Research summary: qSOFA {qsofa}/3 and SIRS {sirs}/3. "
        f"The encoded risk category is {risk_level}; this is not a diagnosis."
    )

    return CopilotResponse(
        analysis=analysis,
        risk_level=risk_level,
        key_concerns=concerns[:5],
        suggested_actions=[
            "Verify observation values, timestamps, units, and data source.",
            "Record reviewer feedback under the approved validation protocol.",
        ],
        disclaimer=(
            "Investigational research summary. Not for diagnosis or treatment."
        ),
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint for real-time alerts
# ---------------------------------------------------------------------------

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Real-time sepsis alert stream via WebSocket.

    Clients receive JSON messages when any patient triggers a high/critical alert.
    Authentication uses the ``bearer.<JWT>`` WebSocket subprotocol so credentials
    never appear in URLs or access logs.
    """
    # Authenticate WebSocket handshake via JWT
    ws_org_id = None  # org_id for filtering broadcasts
    ws_expires_at: Optional[float] = None  # close when the access token expires
    selected_subprotocol = None
    if _auth_enabled:
        offered = [
            value.strip()
            for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if value.strip()
        ]
        token_protocol = next(
            (value for value in offered if value.startswith("bearer.")),
            None,
        )
        token = token_protocol.removeprefix("bearer.") if token_protocol else None
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        try:
            from sepsis_vitals.auth.tokens import decode_token
            payload = decode_token(token)
            if payload.get("type") != "access":
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            ws_org_id = payload.get("org_id")
            ws_expires_at = float(payload["exp"])
            if ws_org_id is None and payload.get("role") != "system_admin":
                # Fail closed: an org-less connection would receive every site's alerts.
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            selected_subprotocol = "sepsis-vitals" if "sepsis-vitals" in offered else None
        except Exception:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await ws_manager.connect(
        websocket,
        org_id=ws_org_id,
        subprotocol=selected_subprotocol,
    )
    try:
        while True:
            # Keep connection alive, receive any client messages. The session
            # must not outlive its access token: the client reconnects with a
            # fresh token after refreshing.
            if ws_expires_at is not None:
                remaining = ws_expires_at - time.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                data = await asyncio.wait_for(websocket.receive_text(), timeout=remaining)
            else:
                data = await websocket.receive_text()
            # Client can send vitals for immediate scoring
            try:
                vitals = json.loads(data)
                scores = compute_scores(vitals)
                await websocket.send_json({
                    "type": "score_result",
                    "scores": scores.as_dict(),
                })
            except Exception:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except asyncio.TimeoutError:
        ws_manager.disconnect(websocket)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="token expired")


# ---------------------------------------------------------------------------
# Prometheus-compatible metrics
# ---------------------------------------------------------------------------

@app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(check_rate_limit)])
async def prometheus_metrics(user: Dict = Depends(verify_auth)):
    """Prometheus-compatible metrics endpoint. Requires auth in production."""
    from sepsis_vitals.monitoring.drift_monitor import get_drift_monitor
    drift_status = get_drift_monitor().get_drift_status()

    # Build per-vital PSI lines
    drift_lines = []
    for vital, info in drift_status.get("per_vital", {}).items():
        psi_val = info.get("psi", 0.0)
        drift_lines += [
            f'sepsis_psi{{vital="{vital}"}} {psi_val:.6f}',
        ]

    lines = [
        "# HELP sepsis_requests_total Total API requests",
        "# TYPE sepsis_requests_total counter",
        f'sepsis_requests_total {_metrics["requests_total"]}',
        "",
        "# HELP sepsis_predictions_total Total ML predictions made",
        "# TYPE sepsis_predictions_total counter",
        f'sepsis_predictions_total {_metrics["predictions_total"]}',
        "",
        "# HELP sepsis_alerts_total Total sepsis alerts triggered",
        "# TYPE sepsis_alerts_total counter",
        f'sepsis_alerts_total {_metrics["alerts_total"]}',
        "",
        "# HELP sepsis_errors_total Total API errors",
        "# TYPE sepsis_errors_total counter",
        f'sepsis_errors_total {_metrics["errors_total"]}',
        "",
        "# HELP sepsis_copilot_calls_total Total AI copilot calls",
        "# TYPE sepsis_copilot_calls_total counter",
        f'sepsis_copilot_calls_total {_metrics["copilot_calls_total"]}',
        "",
        "# HELP sepsis_rate_limited_total Total rate-limited requests",
        "# TYPE sepsis_rate_limited_total counter",
        f'sepsis_rate_limited_total {_metrics["rate_limited_total"]}',
        "",
        "# HELP sepsis_prediction_latency_ms Average prediction latency",
        "# TYPE sepsis_prediction_latency_ms gauge",
        f'sepsis_prediction_latency_ms {_metrics["avg_prediction_ms"]:.1f}',
        "",
        "# HELP sepsis_websocket_connections Active WebSocket connections",
        "# TYPE sepsis_websocket_connections gauge",
        f"sepsis_websocket_connections {ws_manager.active_connections}",
        "",
        "# HELP sepsis_model_loaded Whether the ML model is loaded",
        "# TYPE sepsis_model_loaded gauge",
        f"sepsis_model_loaded {1 if _get_predictor() is not None else 0}",
        "",
        "# HELP sepsis_drift_overall Whether overall population drift is detected (PSI>0.2)",
        "# TYPE sepsis_drift_overall gauge",
        f"sepsis_drift_overall {1 if drift_status['overall_drift'] else 0}",
        "",
        "# HELP sepsis_psi Population Stability Index per vital sign",
        "# TYPE sepsis_psi gauge",
        *drift_lines,
        "",
        "# HELP sepsis_drift_buffer_size Number of recent predictions buffered for drift detection",
        "# TYPE sepsis_drift_buffer_size gauge",
        *(
            f'sepsis_drift_buffer_size{{vital="{v}"}} {n}'
            for v, n in drift_status.get("buffer_counts", {}).items()
        ),
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Sub-routers — auth, patients, billing, alerts, FHIR
# ---------------------------------------------------------------------------

def _init_database():
    """Initialize database tables on startup.

    Registers only the models enabled for this deployment, then creates the
    required tables. Safe to call multiple times.
    """
    if os.getenv("SEPSIS_ENABLE_BILLING", "false").lower() == "true":
        try:
            # Billing is outside the investigational product's critical path.
            import sepsis_vitals.billing.models  # noqa: F401
        except ImportError:
            logger.info("Billing models not available — skipping")
        except Exception as exc:
            logger.error("Failed to import billing models: %s", exc)

    if os.getenv("SEPSIS_ENABLE_TREATMENT_BUNDLES", "false").lower() == "true":
        try:
            # Register treatment workflow tables only for explicitly approved
            # deployments; this feature is frozen for investigational use.
            import sepsis_vitals.bundles.models  # noqa: F401
        except ImportError:
            logger.info("Bundle models not available — skipping")
        except Exception as exc:
            logger.error("Failed to import bundle models: %s", exc)

    from sepsis_vitals.db import init_db
    init_db()
    logger.info("Database tables initialized")


def _include_routers():
    """Include sub-routers with graceful handling if optional deps are missing."""
    routers = [
        ("sepsis_vitals.auth.router", "auth", [Depends(check_auth_rate_limit)]),
        ("sepsis_vitals.patients.router", "patients", [Depends(check_rate_limit)]),
        ("sepsis_vitals.alerts.router", "alerts", [Depends(check_rate_limit)]),
        ("sepsis_vitals.fhir.router", "fhir", [Depends(check_rate_limit)]),
    ]
    if os.getenv("SEPSIS_ENABLE_BILLING", "false").lower() == "true":
        routers.append(("sepsis_vitals.billing.router", "billing", []))
    if os.getenv("SEPSIS_ENABLE_TREATMENT_BUNDLES", "false").lower() == "true":
        routers.append(
            ("sepsis_vitals.bundles.router", "bundles", [Depends(check_rate_limit)])
        )
    for module_path, tag, deps in routers:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            app.include_router(mod.router, tags=[tag], dependencies=deps)
        except ImportError as exc:
            logger.info("Skipping %s router (missing dependency: %s)", tag, exc)
        except Exception as exc:
            logger.error("Failed to load %s router: %s", tag, exc, exc_info=True)

# Database init and router wiring now happen via the lifespan context
# manager (see _lifespan above) instead of at module import time.
