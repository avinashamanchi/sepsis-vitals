"""
sepsis_vitals.bundles.router
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FastAPI router exposing Hour-1 bundle tracking.

Endpoints (all under ``/bundles``):

    POST   /bundles/start                 open (or return existing) bundle
    GET    /bundles/patient/{patient_id}  current open bundle for a patient
    GET    /bundles/{bundle_id}           fetch a bundle by id
    PATCH  /bundles/{bundle_id}/task      mark a task complete / undo
    POST   /bundles/{bundle_id}/cancel    close without completing
    POST   /bundles/sweep                 expire stale bundles (admin/cron)

Mirrors the conventions in :mod:`sepsis_vitals.patients.router`:
``verify_auth`` gates every route and ``get_db`` supplies the session.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from sepsis_vitals.api import verify_auth
from sepsis_vitals.bundles import service
from sepsis_vitals.bundles.protocol import list_task_keys
from sepsis_vitals.db import get_db

router = APIRouter(prefix="/bundles", tags=["bundles"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BundleStartRequest(BaseModel):
    patient_id: str = Field(..., description="Patient UUID")
    alert_id: Optional[str] = Field(None, description="Triggering alert id")
    risk_level: Optional[str] = Field(None, description="Risk level at start")
    vitals: Optional[Dict[str, Any]] = Field(
        None,
        description="Latest vitals; used to select conditional tasks "
        "(lactate, map, sbp).",
    )


class TaskUpdateRequest(BaseModel):
    task_key: str = Field(..., description="Protocol task key")
    completed: bool = Field(True, description="Set false to undo a task")
    note: Optional[str] = Field(None, max_length=500)


class CancelRequest(BaseModel):
    reason: Optional[str] = Field(None, max_length=500)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _uid(user: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of a user id from the auth dependency."""
    if not isinstance(user, dict):
        return None
    return user.get("id") or user.get("user") or user.get("email")


@router.post("/start", status_code=status.HTTP_201_CREATED)
def start_bundle(
    body: BundleStartRequest,
    db: Session = Depends(get_db),
    user: Dict = Depends(verify_auth),
):
    """Open a Hour-1 bundle (idempotent per patient)."""
    try:
        bundle = service.start_bundle(
            body.patient_id,
            db,
            started_by=_uid(user),
            alert_id=body.alert_id,
            risk_level=body.risk_level,
            vitals=body.vitals,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.bundle_to_dict(bundle)


@router.get("/patient/{patient_id}")
def get_patient_bundle(
    patient_id: str,
    db: Session = Depends(get_db),
    user: Dict = Depends(verify_auth),
):
    """Return the patient's currently-open bundle, or 404 if none."""
    bundle = service.get_open_bundle(patient_id, db)
    if bundle is None:
        raise HTTPException(status_code=404, detail="No open bundle")
    return service.bundle_to_dict(bundle)


@router.get("/{bundle_id}")
def get_bundle(
    bundle_id: str,
    db: Session = Depends(get_db),
    user: Dict = Depends(verify_auth),
):
    from sepsis_vitals.bundles.models import SepsisBundle

    bundle = db.query(SepsisBundle).filter(SepsisBundle.id == bundle_id).first()
    if bundle is None:
        raise HTTPException(status_code=404, detail="Bundle not found")
    return service.bundle_to_dict(bundle)


@router.patch("/{bundle_id}/task")
def update_task(
    bundle_id: str,
    body: TaskUpdateRequest,
    db: Session = Depends(get_db),
    user: Dict = Depends(verify_auth),
):
    """Mark a bundle task complete (or undo it)."""
    if body.task_key not in list_task_keys():
        raise HTTPException(
            status_code=400, detail=f"Unknown task '{body.task_key}'"
        )
    try:
        bundle = service.complete_task(
            bundle_id,
            body.task_key,
            db,
            completed_by=_uid(user),
            note=body.note,
            completed=body.completed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.bundle_to_dict(bundle)


@router.post("/{bundle_id}/cancel")
def cancel_bundle(
    bundle_id: str,
    body: CancelRequest,
    db: Session = Depends(get_db),
    user: Dict = Depends(verify_auth),
):
    try:
        bundle = service.cancel_bundle(
            bundle_id, db, cancelled_by=_uid(user), reason=body.reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.bundle_to_dict(bundle)


@router.post("/sweep")
def sweep_bundles(
    db: Session = Depends(get_db),
    user: Dict = Depends(verify_auth),
):
    """Expire bundles whose window has lapsed. Safe to call from a cron job."""
    expired = service.expire_stale_bundles(db)
    return {"expired": expired}
