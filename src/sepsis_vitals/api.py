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

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from sepsis_vitals.security import RateLimiter, RateLimitExceeded, sanitise_string

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sepsis Vitals API",
    version=__version__,
    description="AI-powered vitals-only sepsis prediction for low-resource hospitals",
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = os.getenv(
    "SEPSIS_ALLOWED_ORIGINS",
    "http://localhost:8000,http://localhost:3000,https://avinashamanchi.github.io",
).split(",")

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
_copilot_limiter = RateLimiter(rate=0.5, burst=3)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


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


# ---------------------------------------------------------------------------
# Authentication (JWT-style token auth)
# ---------------------------------------------------------------------------

# Simple token-based auth. In production, replace with proper JWT RS256.
API_KEYS: Dict[str, Dict[str, str]] = {}
_auth_enabled = os.getenv("SEPSIS_AUTH_ENABLED", "false").lower() == "true"


def _load_api_keys():
    """Load API keys from environment."""
    keys_json = os.getenv("SEPSIS_API_KEYS", "")
    if keys_json:
        try:
            parsed = json.loads(keys_json)
            for key, info in parsed.items():
                API_KEYS[key] = info
        except (json.JSONDecodeError, TypeError):
            pass


_load_api_keys()


async def verify_auth(request: Request) -> Optional[Dict[str, str]]:
    """Verify authorization header. Returns user info or None if auth disabled."""
    if not _auth_enabled:
        return {"role": "system_admin", "user": "anonymous"}

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Use 'Bearer <api_key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]
    user_info = API_KEYS.get(token)
    if user_info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_info


def require_permission(permission: str):
    """Dependency factory that checks RBAC permissions."""
    async def _check(user: Dict = Depends(verify_auth)):
        from sepsis_vitals.auth.jwt import check_permission, AuthorizationError
        try:
            check_permission(user.get("role", ""), permission)
        except (AuthorizationError, Exception):
            if _auth_enabled:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {permission}",
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
    patients: List[PredictRequest] = Field(..., max_length=50)


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
# Metrics tracking
# ---------------------------------------------------------------------------

_metrics = {
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
# Request counting middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def count_requests(request: Request, call_next):
    _metrics["requests_total"] += 1
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    predictor = _get_predictor()
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
async def score_vitals(vitals: VitalsInput):
    """Compute clinical sepsis scores (qSOFA, SIRS, NEWS2, Shock Index, UVA)."""
    vitals_dict = {k: v for k, v in vitals.dict().items() if v is not None}
    if len(vitals_dict) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 vital signs.")
    result = compute_scores(vitals_dict)
    d = result.as_dict()
    return ScoreResponse(**d)


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(check_rate_limit), Depends(check_ml_rate_limit)])
async def predict_sepsis(body: PredictRequest, user: Dict = Depends(verify_auth)):
    """ML-powered sepsis risk prediction with SHAP explanations."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run 'python -m sepsis_vitals.train' first.")

    vitals_dict = {k: v for k, v in body.vitals.dict().items() if v is not None}
    if len(vitals_dict) < 3:
        raise HTTPException(status_code=422, detail="Provide at least 3 vital signs for ML prediction.")

    comorbidities = body.comorbidities.dict() if body.comorbidities else None

    start = time.monotonic()
    prediction = predictor.predict(
        vitals=vitals_dict,
        patient_id=sanitise_string(body.patient_id),
        age_years=body.age_years,
        comorbidities=comorbidities,
    )
    elapsed_ms = (time.monotonic() - start) * 1000

    result = prediction.to_dict()
    _track_prediction(elapsed_ms, result.get("alert", False))

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
async def predict_batch(body: BatchPredictRequest, user: Dict = Depends(verify_auth)):
    """Batch prediction for multiple patients (max 50)."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for patient in body.patients:
        vitals_dict = {k: v for k, v in patient.vitals.dict().items() if v is not None}
        comorbidities = patient.comorbidities.dict() if patient.comorbidities else None
        prediction = predictor.predict(
            vitals=vitals_dict,
            patient_id=sanitise_string(patient.patient_id),
            age_years=patient.age_years,
            comorbidities=comorbidities,
        )
        results.append(prediction.to_dict())

    return {"predictions": results, "count": len(results)}


@app.get("/patient/{patient_id}/trend", dependencies=[Depends(check_rate_limit)])
async def patient_trend(patient_id: str, user: Dict = Depends(verify_auth)):
    """Get risk trend for a monitored patient."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    trend = predictor.get_patient_trend(sanitise_string(patient_id))
    if trend is None:
        raise HTTPException(status_code=404, detail=f"No data for patient {patient_id}")
    return trend


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
        "training_data": "NHANES-calibrated synthetic (10K patients)",
        "metrics": predictor.metadata.get("metrics", {}),
        "feature_importance": dict(list(
            predictor.metadata.get("feature_importance", {}).items()
        )[:15]),
    }


# ---------------------------------------------------------------------------
# AI Clinical Copilot (Anthropic-powered)
# ---------------------------------------------------------------------------

@app.post("/copilot", response_model=CopilotResponse, dependencies=[Depends(check_rate_limit)])
async def clinical_copilot(body: CopilotRequest, user: Dict = Depends(verify_auth)):
    """AI clinical decision support using Claude.

    Analyzes vitals, scores, and ML prediction to provide natural-language
    clinical guidance. Falls back to rule-based analysis if API key unavailable.
    """
    ip = _client_ip(body) if hasattr(body, 'client') else "api"
    if not _copilot_limiter.allow("copilot_global"):
        raise HTTPException(status_code=429, detail="Copilot rate limit exceeded. Max 1 request per 2 seconds.")

    _metrics["copilot_calls_total"] += 1

    vitals_dict = {k: v for k, v in body.vitals.dict().items() if v is not None}
    scores = compute_scores(vitals_dict)
    scores_dict = scores.as_dict()

    # Get ML prediction if model loaded
    ml_risk = None
    predictor = _get_predictor()
    if predictor:
        comorbidities = body.comorbidities.dict() if body.comorbidities else None
        pred = predictor.predict(
            vitals=vitals_dict,
            patient_id=body.patient_id,
            age_years=body.age_years,
            comorbidities=comorbidities,
        )
        ml_risk = pred.to_dict()

    # Try Anthropic API, fall back to rule-based
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            analysis = await _anthropic_copilot(
                vitals_dict, scores_dict, ml_risk, body.age_years, body.question
            )
            return analysis
        except Exception:
            pass  # Fall back to rule-based

    # Rule-based fallback
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

    prompt = f"""You are a clinical decision support system for sepsis screening. Analyze the following patient data and provide a structured assessment.

Patient vitals: {json.dumps(vitals)}
Age: {age if age else 'Unknown'}
Clinical scores: qSOFA={scores.get('qsofa',0)}/3, SIRS={scores.get('sirs_count',0)}/3, NEWS2={scores.get('news2_style',0)}, Shock Index={scores.get('shock_index','N/A')}
Risk level: {scores.get('risk_level', 'unknown')}
{risk_info}
{f'Clinical question: {question}' if question else ''}

Respond in this exact JSON format:
{{
  "analysis": "2-3 sentence clinical assessment",
  "risk_level": "low|moderate|high|critical",
  "key_concerns": ["concern1", "concern2"],
  "suggested_actions": ["action1", "action2", "action3"]
}}

Be concise, clinically precise. Focus on actionable next steps. This is a decision SUPPORT tool - always recommend clinician assessment."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
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
        disclaimer="AI-generated clinical decision support. Not a substitute for clinical judgment. Always verify with qualified clinician.",
    )


def _rule_based_copilot(
    vitals: dict, scores: dict, ml_risk: Optional[dict], age: Optional[int],
) -> CopilotResponse:
    """Rule-based fallback when Anthropic API is unavailable."""
    concerns = []
    actions = []
    risk_level = scores.get("risk_level", "low")

    # Analyze vital signs
    temp = vitals.get("temperature")
    if temp and (temp > 38.3 or temp < 36.0):
        concerns.append(f"Abnormal temperature ({temp}°C) — possible infection or hypothermia")
        actions.append("Obtain blood cultures before antibiotic administration")

    hr = vitals.get("heart_rate")
    if hr and hr > 100:
        concerns.append(f"Tachycardia (HR {hr} bpm) — may indicate sepsis, hypovolemia, or pain")
    elif hr and hr < 50:
        concerns.append(f"Bradycardia (HR {hr} bpm) — assess medication effects and cardiac status")

    rr = vitals.get("resp_rate")
    if rr and rr > 22:
        concerns.append(f"Tachypnea (RR {rr}/min) — qSOFA criterion, assess respiratory status")
        actions.append("Monitor oxygen saturation continuously")

    sbp = vitals.get("sbp")
    if sbp and sbp <= 100:
        concerns.append(f"Hypotension (SBP {sbp} mmHg) — qSOFA criterion, assess perfusion")
        actions.append("Consider IV fluid resuscitation (30 mL/kg crystalloid)")

    spo2 = vitals.get("spo2")
    if spo2 and spo2 < 94:
        concerns.append(f"Hypoxemia (SpO2 {spo2}%) — assess airway and provide supplemental O2")
        actions.append("Apply supplemental oxygen, target SpO2 ≥94%")

    gcs = vitals.get("gcs")
    if gcs and gcs < 15:
        concerns.append(f"Altered consciousness (GCS {gcs}/15) — qSOFA criterion")

    qsofa = scores.get("qsofa", 0)
    sirs = scores.get("sirs_count", 0)

    # ML risk integration
    ml_prob = ml_risk["risk_probability"] if ml_risk else None
    if ml_prob and ml_prob > 0.5:
        concerns.append(f"ML model predicts {ml_prob:.0%} sepsis probability")

    # Risk-based actions
    if risk_level == "critical" or qsofa >= 2:
        risk_level = "critical"
        actions.insert(0, "IMMEDIATE clinical assessment — activate sepsis protocol")
        actions.append("Obtain serum lactate level")
        actions.append("Administer broad-spectrum antibiotics within 1 hour")
        actions.append("Initiate Surviving Sepsis Campaign hour-1 bundle")
    elif risk_level == "high" or qsofa >= 1:
        actions.insert(0, "Urgent clinical review within 30 minutes")
        actions.append("Check serum lactate and complete blood count")
    elif risk_level == "moderate" or sirs >= 2:
        actions.append("Reassess vitals in 1-2 hours")
        actions.append("Consider infection workup if clinical suspicion")
    else:
        actions.append("Continue routine monitoring per protocol")

    if not concerns:
        concerns.append("Vital signs within normal limits")

    # Build analysis text
    ml_text = f" ML model predicts {ml_prob:.0%} risk." if ml_prob else ""
    analysis = (
        f"Patient presents with qSOFA {qsofa}/3, SIRS {sirs}/3 criteria met.{ml_text} "
        f"Risk classification: {risk_level.upper()}. "
        f"{len(concerns)} clinical concern(s) identified."
    )

    return CopilotResponse(
        analysis=analysis,
        risk_level=risk_level,
        key_concerns=concerns[:5],
        suggested_actions=actions[:6],
        disclaimer="Rule-based clinical decision support (AI copilot offline). Not a substitute for clinical judgment.",
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint for real-time alerts
# ---------------------------------------------------------------------------

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Real-time sepsis alert stream via WebSocket.

    Clients receive JSON messages when any patient triggers a high/critical alert.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Client can send vitals for immediate scoring
            try:
                vitals = json.loads(data)
                scores = compute_scores(vitals)
                await websocket.send_json({
                    "type": "score_result",
                    "scores": scores.as_dict(),
                })
            except (json.JSONDecodeError, Exception):
                await websocket.send_json({"type": "error", "detail": "Invalid JSON"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Prometheus-compatible metrics
# ---------------------------------------------------------------------------

@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
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
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Sub-routers — auth, patients, billing, alerts, FHIR
# ---------------------------------------------------------------------------

def _include_routers():
    """Include sub-routers with graceful handling if optional deps are missing."""
    try:
        from sepsis_vitals.auth.router import router as auth_router
        app.include_router(auth_router, tags=["auth"])
    except Exception:
        pass

    try:
        from sepsis_vitals.patients.router import router as patients_router
        app.include_router(patients_router, tags=["patients"])
    except Exception:
        pass

    try:
        from sepsis_vitals.billing.router import router as billing_router
        app.include_router(billing_router, tags=["billing"])
    except Exception:
        pass

    try:
        from sepsis_vitals.alerts.router import router as alerts_router
        app.include_router(alerts_router, tags=["alerts"])
    except Exception:
        pass

    try:
        from sepsis_vitals.fhir.router import router as fhir_router
        app.include_router(fhir_router, tags=["fhir"])
    except Exception:
        pass

_include_routers()
