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
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sepsis_vitals import __version__
from sepsis_vitals.scores import compute_scores
from sepsis_vitals.security import RateLimitExceeded, SecurityAlertTracker, sanitise_string

# Shared dependencies and schemas live in their own modules so routers never
# import this one; the names are re-exported here for compatibility.
from sepsis_vitals import dependencies as _deps
from sepsis_vitals.dependencies import (  # noqa: F401
    _TRUSTED_PROXIES,
    _anonymous_user,
    _api_limiter,
    _auth_limiter,
    _billing_limiter,
    _client_ip,
    _copilot_limiter,
    _is_production,
    _ml_limiter,
    _verify_patient_org_async,
    _webhook_limiter,
    check_auth_rate_limit,
    check_ml_rate_limit,
    check_rate_limit,
    require_role_dep,
    verify_auth,
    verify_patient_org,
)
from sepsis_vitals.schemas import (  # noqa: F401
    NEWS2_LIMITATIONS,
    BatchPredictRequest,
    ComorbidityInput,
    ConfidenceInterval,
    CopilotRequest,
    CopilotResponse,
    HealthResponse,
    MonitorRegisterRequest,
    PredictionResponse,
    PredictRequest,
    ScoreResponse,
    SimulatorReplayRequest,
    SimulatorWardRequest,
    VitalsInput,
    _count_measurements,
)

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lifespan — database init and model warm-up at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Startup/shutdown lifecycle for the FastAPI application."""
    from sepsis_vitals.logging_config import configure_logging

    configure_logging()  # audit and INFO logs reach stdout under uvicorn
    _init_database()
    _include_routers()  # no-op: done at import; kept for apps built before that
    # Load and verify the model off the event loop (~1.7 s); never raises, the
    # outcome is reported by /model/status.
    await asyncio.to_thread(_get_predictor)

    # Start PSI drift monitor background task
    from sepsis_vitals.monitoring.drift_monitor import get_drift_monitor
    drift_monitor = get_drift_monitor()
    await drift_monitor.start()

    yield

    await drift_monitor.stop()


app: FastAPI = FastAPI(
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
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_predictor = None


_model_status: Dict[str, Any] = {"state": "absent", "reason": "not loaded yet"}


def _get_predictor():
    """Return the loaded predictor, or None (state recorded in _model_status).

    Absent, tampered, or incompatible artifacts never raise out of request
    handlers: /predict answers 503 and /model/status explains why.
    """
    global _predictor, _model_status
    if _predictor is None:
        from sepsis_vitals.ml.artifacts import ModelArtifactError
        from sepsis_vitals.ml.predictor import SepsisPredictor
        candidate = SepsisPredictor()
        try:
            candidate.load()
        except FileNotFoundError:
            _model_status = {"state": "absent", "reason": "No model artifact is installed"}
            return None
        except ModelArtifactError as exc:
            _model_status = {"state": exc.state, "reason": exc.reason}
            logger.error("Model artifacts rejected (%s): %s", exc.state, exc.reason)
            return None
        except Exception as exc:  # corrupt pickle, unreadable metadata, ...
            _model_status = {"state": "invalid", "reason": f"Model failed to load: {type(exc).__name__}"}
            logger.error("Model failed to load", exc_info=True)
            return None
        _predictor = candidate
        _model_status = candidate.artifact_status.as_dict()
    return _predictor


def _model_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "message": "Predictions are unavailable: no usable model is installed.",
            "model_state": _model_status.get("state"),
            "reason": _model_status.get("reason"),
        },
    )


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
                output_scores=json.dumps({
                    "clinical_scores": result.get("clinical_scores", {}),
                    "rule_risk_level": result.get("rule_risk_level"),
                    "model_risk_level": result.get("model_risk_level"),
                    "provenance": result.get("provenance", {}),
                }),
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

# Path segments that carry a patient identifier (internal id or MRN).
_AUDIT_ID_PREFIXES = ("/fhir/Patient/", "/fhir/RiskAssessment/", "/patient/")


def _audit_action(method: str, path: str) -> Optional[str]:
    """Audit action for a request path, or None when it touches no PHI."""
    if path.startswith("/fhir/"):
        if path == "/fhir/metadata":
            return None
        return "fhir_write" if method == "POST" else "fhir_read"
    for pattern, action in _PHI_AUDIT_PATTERNS.items():
        if path.startswith(pattern) or (pattern.endswith("/") and pattern[:-1] in path):
            return action
    return None


def _audit_target(path: str) -> tuple[str, Optional[str]]:
    """(path safe to log, patient id to record) for an audited request.

    An internal patient id (a UUID) is recorded as the resource id. Any other
    identifier in the path, such as an MRN used for a FHIR lookup, is
    replaced in the logged path by its keyed reference and not stored.
    """
    from sepsis_vitals.db import is_uuid
    from sepsis_vitals.security import log_ref

    for prefix in _AUDIT_ID_PREFIXES:
        head, found, rest = path.partition(prefix)
        if not found:
            continue
        ident, sep, tail = rest.partition("/")
        if not ident:
            break
        if is_uuid(ident):
            return path, ident
        return f"{head}{prefix}{log_ref(ident)}{sep}{tail}", None
    return path, None


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
    audit_action = _audit_action(request.method, path)

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

        # Patient id from the path, if present (MRNs are not logged)
        safe_path, patient_id = _audit_target(path)

        await asyncio.to_thread(
            _emit_audit_event,
            action=audit_action,
            user_id=user_id,
            ip_address=ip,
            path=safe_path,
            patient_id=patient_id,
            status_code=response.status_code,
        )

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness. Never loads the model (see /model/status and /ready)."""
    predictor = _predictor
    if _is_production:
        return {"status": "ok", "version": __version__, "timestamp": time.time()}
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=time.time(),
        model_loaded=predictor is not None,
        model_name=predictor.metadata["model_name"] if predictor and predictor.metadata else None,
        auth_enabled=_deps._auth_enabled,
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
    predictor = await asyncio.to_thread(_get_predictor)
    if predictor is None:
        raise _model_unavailable()

    vitals_dict = {k: v for k, v in body.vitals.model_dump().items() if v is not None}
    if _count_measurements(vitals_dict) < 3:
        raise HTTPException(status_code=422, detail="Provide at least 3 vital signs for ML prediction.")

    comorbidities = body.comorbidities.model_dump() if body.comorbidities else None
    await _ensure_not_foreign_patient_async(sanitise_string(body.patient_id), user)
    history = await _recorded_history_async(sanitise_string(body.patient_id))

    start = time.monotonic()
    prediction = await asyncio.to_thread(
        predictor.predict,
        vitals=vitals_dict,
        patient_id=sanitise_string(body.patient_id),
        age_years=body.age_years,
        comorbidities=comorbidities,
        history=history,
    )
    elapsed_ms = (time.monotonic() - start) * 1000

    result = prediction.to_dict()
    result["validation_status"] = result["provenance"].get("validation_status") or "unvalidated"
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

    predictor = await asyncio.to_thread(_get_predictor)
    if predictor is None:
        raise _model_unavailable()

    results = []
    errors = []
    model_version = predictor.metadata.get("version") if predictor.metadata else None

    for patient in body.patients:
        await _ensure_not_foreign_patient_async(sanitise_string(patient.patient_id), user)

    for i, patient in enumerate(body.patients):
        try:
            vitals_dict = {k: v for k, v in patient.vitals.model_dump().items() if v is not None}
            comorbidities = patient.comorbidities.model_dump() if patient.comorbidities else None
            prediction = await asyncio.to_thread(
                predictor.predict,
                vitals=vitals_dict,
                patient_id=sanitise_string(patient.patient_id),
                age_years=patient.age_years,
                comorbidities=comorbidities,
                history=await _recorded_history_async(sanitise_string(patient.patient_id)),
            )
            result = prediction.to_dict()
            result["validation_status"] = result["provenance"].get("validation_status") or "unvalidated"
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


@app.get("/patient/{patient_id}/trend", dependencies=[Depends(check_rate_limit)])
async def patient_trend(patient_id: str, request: Request, user: Dict = Depends(verify_auth)):
    """Get risk trend for a monitored patient."""
    # Org-level authorization: verify patient belongs to user's org (fail closed)
    await _verify_patient_org_async(patient_id, user)

    predictor = await asyncio.to_thread(_get_predictor)
    if predictor is None:
        raise _model_unavailable()

    trend = predictor.get_patient_trend(sanitise_string(patient_id))
    if trend is None:
        raise HTTPException(status_code=404, detail=f"No data for patient {patient_id}")
    return trend


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


_routers_included = False


def _include_routers():
    """Include every router once, in a fixed order.

    Called when this module is imported, so ``app.routes`` and the OpenAPI
    schema are complete without running the lifespan (e.g. a TestClient used
    without ``with``, or schema export). Routers never import this module;
    their shared dependencies come from :mod:`sepsis_vitals.dependencies`.
    The billing and bundle routers are included only when their feature flag
    is set in the environment at import time. Idempotent.
    """
    global _routers_included
    if _routers_included:
        return
    _routers_included = True

    # Endpoint modules split out of this file, after this file's own routes
    # (the order routes were registered in before the split).
    from sepsis_vitals.routes import copilot, metrics, monitor, realtime, simulator, status

    for module in (monitor, simulator, status, copilot, realtime, metrics):
        app.include_router(module.router)

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


_include_routers()


# ---------------------------------------------------------------------------
# Compatibility re-exports: endpoint functions that used to live here, so
# `from sepsis_vitals.api import X` keeps working.
# ---------------------------------------------------------------------------

from sepsis_vitals.routes.copilot import (  # noqa: E402,F401
    _anthropic_copilot,
    _copilot_enabled,
    _deidentify_vitals,
    _enterprise_llm_enabled,
    _rule_based_copilot,
    clinical_copilot,
)
from sepsis_vitals.routes.metrics import prometheus_metrics  # noqa: E402,F401
from sepsis_vitals.routes.monitor import (  # noqa: E402,F401
    monitor_register,
    monitor_status,
    monitor_unregister,
)
from sepsis_vitals.routes.realtime import websocket_alerts  # noqa: E402,F401
from sepsis_vitals.routes.simulator import (  # noqa: E402,F401
    simulator_cases,
    simulator_sessions,
    simulator_start_replay,
    simulator_start_ward,
    simulator_stop,
)
from sepsis_vitals.routes.status import (  # noqa: E402,F401
    _alembic_head,
    model_info,
    model_status,
    readiness,
)
