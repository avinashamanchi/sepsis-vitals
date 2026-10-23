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
import re
import time
import uuid
from collections import deque
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
from sepsis_vitals.ml.decision_policy import for_display
from sepsis_vitals.scores import UVA_REFERENCE, compute_scores, uva_score
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
    ScoreInput,
    ScoreResponse,
    SimulatorReplayRequest,
    SimulatorWardRequest,
    TimeseriesPredictRequest,
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
        try:
            from sepsis_vitals.ml.artifacts import ModelArtifactError
            from sepsis_vitals.ml.predictor import SepsisPredictor
        except ImportError as exc:
            # An API-only install (sepsis-vitals[api] without [ml]) still serves
            # scores and patient data; predictions report why they are unavailable.
            _model_status = {
                "state": "unavailable",
                "reason": f"ML runtime not installed (missing {exc.name}); install sepsis-vitals[ml]",
            }
            return None
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


def _monitor_audit_writer(*, patient_id: str, result: Dict[str, Any], payload: Dict[str, Any]) -> str:
    """Audit record for a continuous-monitor result (raises AuditWriteError)."""
    from sepsis_vitals.audit_records import AuditWriteError, record_prediction

    version = (result.get("versions") or {}).get("model_version")
    try:
        record_id = record_prediction(patient_id=patient_id, user_id=None, result=result,
                                      payload=payload, source="monitor",
                                      request_id=f"monitor-{uuid.uuid4().hex}")
    except AuditWriteError as exc:
        _note_inference(None, "audit_unavailable", error_class=str(exc), model_version=version)
        raise
    _note_inference(None, "scored", model_version=version)
    return record_id


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
                audit_writer=_monitor_audit_writer,
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
    "audit_log_failures_total": 0,
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


# Outcome of recent inference attempts, for /model/status. Holds no patient data.
_recent_inferences: "deque[Dict[str, Any]]" = deque(maxlen=200)

NO_RISK_MODEL_ERROR = (
    "No risk calculated: the model could not score this request. "
    "This is not a low-risk result."
)
NO_RISK_AUDIT_UNAVAILABLE = (
    "No risk calculated: the result could not be written to the audit trail, so it "
    "was withheld. This is not a low-risk result."
)


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if not rid:
        rid = uuid.uuid4().hex
        request.state.request_id = rid
    return rid


def _note_inference(request: Optional[Request], outcome: str, **fields: Any) -> None:
    """Record an inference outcome for the request log and /model/status.

    ``fields`` must never contain patient data: model version, latency and
    error class only.
    """
    entry = {"outcome": outcome, "at": time.time(), **fields}
    _recent_inferences.append(entry)
    if request is not None:
        notes = getattr(request.state, "inference", None) or []
        notes.append(entry)
        request.state.inference = notes


async def _score_and_record(
    *,
    predictor: Any,
    request: Request,
    user: Dict[str, Any],
    patient_id: str,
    vitals: Dict[str, Any],
    age_years: Optional[int],
    comorbidities: Optional[Dict[str, Any]],
    history: list,
    source: str,
    observed_at: Optional[Any] = None,
) -> Dict[str, Any]:
    """Score one observation and write its audit record; fail closed.

    The result is returned only after its audit record is committed. A
    model failure or an audit failure raises 503 "no risk calculated":
    never a default, stale or partial score.
    """
    from sepsis_vitals.audit_records import AuditWriteError, input_payload, record_prediction

    request_id = _request_id(request)
    start = time.monotonic()
    try:
        prediction = await asyncio.to_thread(
            predictor.predict,
            vitals=vitals,
            patient_id=patient_id,
            age_years=age_years,
            comorbidities=comorbidities,
            history=history,
            observed_at=observed_at,
            track=False,
        )
    except Exception as exc:
        _note_inference(request, "model_error", error_class=type(exc).__name__)
        logger.error("Inference failed (%s); no risk calculated [request %s]",
                     type(exc).__name__, request_id)
        raise HTTPException(status_code=503, detail=NO_RISK_MODEL_ERROR) from exc
    elapsed_ms = (time.monotonic() - start) * 1000

    result = prediction.to_dict()
    result["validation_status"] = result["provenance"].get("validation_status") or "unvalidated"
    payload = input_payload(vitals, history, age_years, comorbidities, result["observed_at"])
    try:
        record_id = await asyncio.to_thread(
            record_prediction,
            patient_id=patient_id,
            user_id=user.get("id"),
            result=result,
            payload=payload,
            source=source,
            request_id=request_id,
            ip_address=_client_ip(request),
        )
    except AuditWriteError as exc:
        _note_inference(request, "audit_unavailable", error_class=str(exc),
                        model_version=result["versions"]["model_version"])
        raise HTTPException(status_code=503, detail=NO_RISK_AUDIT_UNAVAILABLE) from exc

    predictor.track(prediction)
    result["request_id"] = request_id
    result["audit_record_id"] = record_id
    _track_prediction(elapsed_ms, result.get("alert", False))
    _note_inference(request, "scored", latency_ms=round(elapsed_ms, 1),
                    model_version=result["versions"]["model_version"])

    # Drift monitoring: rolling PSI buffer of the current observation
    from sepsis_vitals.monitoring.drift_monitor import get_drift_monitor
    get_drift_monitor().record_prediction(vitals)
    return result


async def _broadcast_alert(result: Dict[str, Any]) -> None:
    """Send a recorded alert decision to WebSocket clients."""
    if result.get("alert"):
        await ws_manager.broadcast({
            "type": "sepsis_alert",
            "patient_id": result["patient_id"],
            "risk_level": result["risk_level"],
            "risk_probability": result["risk_probability"],
            "recommendation": result["recommendation"],
            "timestamp": result["timestamp"],
            "audit_record_id": result["audit_record_id"],
        })


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

    # Persist to database. The response has already been produced, so this
    # cannot block it; a failure is logged (error class only) and counted,
    # and the structured log line above remains as the second copy.
    try:
        from sepsis_vitals.db import AuditLog, SessionLocal, inet_or_none
        db = SessionLocal()
        try:
            record = AuditLog(
                user_id=user_id if user_id and user_id != "anonymous" else None,
                action=action,
                resource_type="patient_phi",
                resource_id=patient_id,
                details=json.dumps({"path": path, "status_code": status_code}),
                ip_address=inet_or_none(ip_address),
            )
            db.add(record)
            db.commit()
        except Exception as exc:
            db.rollback()
            _metrics["audit_log_failures_total"] += 1
            logger.error("Access audit event not persisted (%s)", type(exc).__name__)
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
# Request IDs and structured request log (outermost middleware)
# ---------------------------------------------------------------------------
# One JSON line per request: request id, method, *route template* (never the
# raw path, which can carry patient identifiers), status, latency, and for
# inference requests the model version, outcome and error class. No bodies,
# query strings, patient identifiers or user identifiers.

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{8,64}$")
request_logger = logging.getLogger("sepsis_vitals.requests")


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    request.state.request_id = incoming if _REQUEST_ID_PATTERN.match(incoming) else uuid.uuid4().hex
    start = time.monotonic()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        route = request.scope.get("route")
        entry: Dict[str, Any] = {
            "request_id": request.state.request_id,
            "method": request.method,
            "route": getattr(route, "path", "unmatched"),
            "status": status,
            "latency_ms": round((time.monotonic() - start) * 1000, 1),
        }
        notes = getattr(request.state, "inference", None)
        if notes:
            entry["inference"] = [
                {k: v for k, v in n.items() if k in ("outcome", "model_version", "latency_ms", "error_class")}
                for n in notes
            ]
        request_logger.info("REQUEST %s", json.dumps(entry, sort_keys=True))
    response.headers["X-Request-ID"] = request.state.request_id
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
async def score_vitals(vitals: ScoreInput, user: Dict = Depends(verify_auth)):
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
    uva = result.uva or uva_score(vitals_dict, hiv_positive=vitals_dict.get("hiv_positive"))
    return ScoreResponse(
        qsofa=result.qsofa,
        sirs_count=result.sirs_count,
        news2_style=result.news2_style,
        shock_index=result.shock_index,
        uva=uva.score,
        uva_risk_band=uva.risk_band,
        uva_missing_inputs=uva.missing_inputs,
        uva_reference=UVA_REFERENCE,
        risk_level=result.risk_level,
        alert_flag=result.alert_flag,
        explanations=explanations,
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def predict_sepsis(body: PredictRequest, request: Request, user: Dict = Depends(verify_auth)):
    """Score one observation. Decision support for research evaluation only.

    History comes from the patient's recorded vitals (registered patients
    only). Returns only after the audit record is written; otherwise 503
    "no risk calculated".
    """
    predictor = await asyncio.to_thread(_get_predictor)
    if predictor is None:
        _note_inference(request, "model_unavailable")
        raise _model_unavailable()

    vitals_dict = {k: v for k, v in body.vitals.model_dump().items() if v is not None}
    if _count_measurements(vitals_dict) < 3:
        raise HTTPException(status_code=422, detail="Provide at least 3 vital signs for ML prediction.")

    patient_id = sanitise_string(body.patient_id)
    comorbidities = body.comorbidities.model_dump() if body.comorbidities else None
    await _ensure_not_foreign_patient_async(patient_id, user)
    history = await _recorded_history_async(patient_id)

    result = await _score_and_record(
        predictor=predictor, request=request, user=user, patient_id=patient_id,
        vitals=vitals_dict, age_years=body.age_years, comorbidities=comorbidities,
        history=history, source="api",
    )
    shown = for_display(result)
    await _broadcast_alert(shown)
    return shown


@app.post("/predict/timeseries", response_model=PredictionResponse, dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def predict_timeseries(body: TimeseriesPredictRequest, request: Request, user: Dict = Depends(verify_auth)):
    """Score the last observation of a validated time series.

    The payload is the complete input: the earlier observations are the
    history and no recorded vitals are added. Validation (units, order,
    duplicates, timestamps, patient consistency, bounds) happens in
    :class:`TimeseriesPredictRequest`; failures are 422 with the reason.
    """
    predictor = await asyncio.to_thread(_get_predictor)
    if predictor is None:
        _note_inference(request, "model_unavailable")
        raise _model_unavailable()

    patient_id = sanitise_string(body.patient_id)
    await _ensure_not_foreign_patient_async(patient_id, user)
    observations = [
        {k: v for k, v in obs.model_dump().items() if v is not None and k != "patient_id"}
        for obs in body.observations
    ]
    current = observations[-1]
    observed_at = current.pop("timestamp")
    comorbidities = body.comorbidities.model_dump() if body.comorbidities else None

    result = await _score_and_record(
        predictor=predictor, request=request, user=user, patient_id=patient_id,
        vitals=current, age_years=body.age_years, comorbidities=comorbidities,
        history=observations[:-1], observed_at=observed_at, source="api-timeseries",
    )
    shown = for_display(result)
    await _broadcast_alert(shown)
    return shown


@app.post("/predict/batch", dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def predict_batch(body: BatchPredictRequest, request: Request, user: Dict = Depends(verify_auth)):
    """Score up to 10 patients. Each result is returned only once recorded.

    A patient that cannot be scored or recorded appears in ``errors`` with
    "no risk calculated", never with a default score.
    """
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
        _note_inference(request, "model_unavailable")
        raise _model_unavailable()

    for patient in body.patients:
        await _ensure_not_foreign_patient_async(sanitise_string(patient.patient_id), user)

    results = []
    errors = []
    for i, patient in enumerate(body.patients):
        patient_id = sanitise_string(patient.patient_id)
        vitals_dict = {k: v for k, v in patient.vitals.model_dump().items() if v is not None}
        if _count_measurements(vitals_dict) < 3:
            errors.append({"index": i, "error": "Provide at least 3 vital signs for ML prediction."})
            continue
        try:
            result = await _score_and_record(
                predictor=predictor, request=request, user=user, patient_id=patient_id,
                vitals=vitals_dict, age_years=patient.age_years,
                comorbidities=patient.comorbidities.model_dump() if patient.comorbidities else None,
                history=await _recorded_history_async(patient_id), source="api-batch",
            )
        except HTTPException as exc:
            # Logged by _score_and_record (error class only, no identifiers).
            errors.append({"index": i, "error": exc.detail})
            continue
        shown = for_display(result)
        results.append(shown)
        await _broadcast_alert(shown)

    return {"predictions": results, "count": len(results), "errors": errors,
            "request_id": _request_id(request)}


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
