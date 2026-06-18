"""
sepsis_vitals.patients.router
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FastAPI router for patient data management, vitals persistence, alert
lifecycle, and site-level dashboard statistics.

All endpoints accept an auth/user dependency parameter for downstream
integration with whichever authentication middleware is active.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from sepsis_vitals.db import get_db
from sepsis_vitals.patients import service

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/patients", tags=["patients"])


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class PatientCreate(BaseModel):
    """Body for creating a new patient."""

    external_id: str = Field(
        ..., min_length=1, max_length=64, description="Site-assigned patient identifier"
    )
    site_id: str = Field(
        ..., min_length=1, max_length=32, description="Site/facility code"
    )
    age_years: Optional[int] = Field(None, ge=0, le=150, description="Patient age in years")
    sex: Optional[str] = Field(
        None, pattern=r"^[MFU]$", description="Biological sex: M, F, or U"
    )


class PatientUpdate(BaseModel):
    """Body for updating patient fields."""

    age_years: Optional[int] = Field(None, ge=0, le=150)
    sex: Optional[str] = Field(None, pattern=r"^[MFU]$")
    site_id: Optional[str] = Field(None, min_length=1, max_length=32)


class PatientOut(BaseModel):
    """Serialised patient record."""

    id: str
    external_id: str
    site_id: str
    age_years: Optional[int] = None
    sex: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class VitalsRecord(BaseModel):
    """Body for recording a new set of vital signs."""

    temperature: Optional[float] = Field(None, ge=25.0, le=45.0, description="Body temperature in celsius")
    heart_rate: Optional[int] = Field(None, ge=0, le=350, description="Heart rate in bpm")
    resp_rate: Optional[int] = Field(None, ge=0, le=80, description="Respiratory rate per minute")
    sbp: Optional[int] = Field(None, ge=30, le=300, description="Systolic blood pressure in mmHg")
    spo2: Optional[int] = Field(None, ge=0, le=100, description="Oxygen saturation percentage")
    gcs: Optional[int] = Field(None, ge=3, le=15, description="Glasgow Coma Scale")
    recorded_at: Optional[datetime] = Field(
        None, description="Observation timestamp (defaults to now UTC)"
    )


class VitalReadingOut(BaseModel):
    """Serialised vital-sign reading."""

    id: str
    patient_id: str
    recorded_at: Optional[datetime] = None
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    resp_rate: Optional[int] = None
    sbp: Optional[int] = None
    spo2: Optional[int] = None
    gcs: Optional[int] = None

    model_config = {"from_attributes": True}


class ScoreOut(BaseModel):
    """Serialised score computed from a vital reading."""

    id: str
    vital_id: str
    qsofa: Optional[int] = None
    sirs_count: Optional[int] = None
    shock_index: Optional[float] = None
    news2_style: Optional[int] = None
    uva_style: Optional[int] = None
    risk_level: str
    alert_flag: bool
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class VitalsResponse(BaseModel):
    """Response returned after recording vitals."""

    vital: VitalReadingOut
    score: ScoreOut


class AlertOut(BaseModel):
    """Serialised alert record."""

    id: str
    score_id: str
    patient_id: str
    risk_level: str
    status: str
    action_by: Optional[str] = None
    action_reason: Optional[str] = None
    time_to_action_s: Optional[float] = None
    created_at: Optional[datetime] = None
    actioned_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AlertAcknowledge(BaseModel):
    """Body for acknowledging an alert."""

    user_id: str = Field(..., min_length=1, description="ID of the user acknowledging")
    reason: str = Field(..., min_length=1, max_length=1000, description="Reason for acknowledgement")


class PatientHistoryOut(BaseModel):
    """Full clinical history for a patient."""

    vitals: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]


class DashboardStats(BaseModel):
    """Site-level dashboard statistics."""

    patient_count: int
    active_alerts: int
    recent_predictions: int


# ---------------------------------------------------------------------------
# Dependency: current user stub
# ---------------------------------------------------------------------------
# Accept an optional auth dependency.  The actual implementation is owned by
# the authentication module wired at application startup.  Here we define a
# lightweight no-op so the router can function standalone during development
# and testing.


async def _current_user_stub() -> Dict[str, str]:
    """Fallback auth dependency returning an anonymous admin user."""
    return {"role": "system_admin", "user": "anonymous"}


# ---------------------------------------------------------------------------
# Endpoints — Patient CRUD
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=PatientOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create patient",
)
def create_patient(
    body: PatientCreate,
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> PatientOut:
    """Register a new patient."""
    try:
        patient = service.create_patient(
            external_id=body.external_id,
            site_id=body.site_id,
            age_years=body.age_years,
            sex=body.sex,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return PatientOut.model_validate(patient)


@router.get(
    "",
    response_model=List[PatientOut],
    summary="List patients",
)
def list_patients(
    site_id: Optional[str] = Query(None, description="Filter by site"),
    skip: int = Query(0, ge=0, description="Records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> List[PatientOut]:
    """Return a paginated list of patients, optionally filtered by site."""
    patients = service.list_patients(site_id=site_id, db=db, skip=skip, limit=limit)
    return [PatientOut.model_validate(p) for p in patients]


@router.get(
    "/{patient_id}",
    response_model=PatientOut,
    summary="Get patient",
)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> PatientOut:
    """Retrieve a single patient by ID."""
    patient = service.get_patient(patient_id, db)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )
    return PatientOut.model_validate(patient)


@router.put(
    "/{patient_id}",
    response_model=PatientOut,
    summary="Update patient",
)
def update_patient(
    patient_id: str,
    body: PatientUpdate,
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> PatientOut:
    """Update mutable fields on a patient record."""
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields provided for update",
        )
    try:
        patient = service.update_patient(patient_id, updates, db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    return PatientOut.model_validate(patient)


# ---------------------------------------------------------------------------
# Endpoints — Vitals
# ---------------------------------------------------------------------------


@router.post(
    "/{patient_id}/vitals",
    response_model=VitalsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record vital signs",
)
def record_vitals(
    patient_id: str,
    body: VitalsRecord,
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> VitalsResponse:
    """Record a new set of vital signs.

    Automatically computes clinical scores (qSOFA, SIRS, NEWS2, Shock Index,
    UVA) and creates an alert if the risk level is high or critical.
    """
    vitals_dict = body.model_dump(exclude={"recorded_at"}, exclude_none=True)
    if not vitals_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one vital sign must be provided",
        )

    try:
        vital, score = service.record_vitals(
            patient_id=patient_id,
            vitals_dict=vitals_dict,
            recorded_at=body.recorded_at,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    return VitalsResponse(
        vital=VitalReadingOut.model_validate(vital),
        score=ScoreOut.model_validate(score),
    )


@router.get(
    "/{patient_id}/vitals",
    response_model=List[VitalReadingOut],
    summary="Get vitals history",
)
def get_patient_vitals(
    patient_id: str,
    hours_back: int = Query(24, ge=1, le=8760, description="Look-back window in hours"),
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> List[VitalReadingOut]:
    """Return vital-sign readings for a patient within the look-back window."""
    patient = service.get_patient(patient_id, db)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )
    readings = service.get_patient_vitals(patient_id, db, hours_back=hours_back)
    return [VitalReadingOut.model_validate(r) for r in readings]


# ---------------------------------------------------------------------------
# Endpoints — History
# ---------------------------------------------------------------------------


@router.get(
    "/{patient_id}/history",
    response_model=PatientHistoryOut,
    summary="Full clinical history",
)
def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> PatientHistoryOut:
    """Return the full clinical history (vitals, scores, alerts) for a patient."""
    patient = service.get_patient(patient_id, db)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found",
        )
    history = service.get_patient_history(patient_id, db)
    return PatientHistoryOut(**history)


# ---------------------------------------------------------------------------
# Endpoints — Alerts  (mounted at /patients/alerts/...)
# ---------------------------------------------------------------------------

alerts_router = APIRouter(prefix="/alerts", tags=["alerts"])


@alerts_router.get(
    "",
    response_model=List[AlertOut],
    summary="List active alerts",
)
def list_active_alerts(
    site_id: Optional[str] = Query(None, description="Filter by site"),
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> List[AlertOut]:
    """Return all currently active (unacknowledged) alerts."""
    alerts = service.get_active_alerts(site_id=site_id, db=db)
    return [AlertOut.model_validate(a) for a in alerts]


@alerts_router.put(
    "/{alert_id}/acknowledge",
    response_model=AlertOut,
    summary="Acknowledge alert",
)
def acknowledge_alert(
    alert_id: str,
    body: AlertAcknowledge,
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> AlertOut:
    """Acknowledge an active alert with a reason."""
    try:
        alert = service.acknowledge_alert(
            alert_id=alert_id,
            user_id=body.user_id,
            reason=body.reason,
            db=db,
        )
    except ValueError as exc:
        error_msg = str(exc)
        if "not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error_msg,
        )
    return AlertOut.model_validate(alert)


# ---------------------------------------------------------------------------
# Endpoints — Dashboard
# ---------------------------------------------------------------------------

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@dashboard_router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Site dashboard statistics",
)
def get_dashboard_stats(
    site_id: str = Query(..., min_length=1, description="Site to query"),
    db: Session = Depends(get_db),
    user: Dict[str, str] = Depends(_current_user_stub),
) -> DashboardStats:
    """Return aggregate statistics for the site dashboard."""
    stats = service.get_site_dashboard_stats(site_id=site_id, db=db)
    return DashboardStats(**stats)
