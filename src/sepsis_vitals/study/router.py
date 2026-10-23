"""
sepsis_vitals.study.router — record adjudicated outcomes for a local study.

POST /study/outcomes  record (or correct) one patient's adjudicated outcome
GET  /study/summary   counts only: patients scored, outcomes recorded

Researchers and system administrators only, within their own site. No free
text is accepted, so outcomes carry no PHI beyond the link to the patient.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from sepsis_vitals.auth.scope import load_patient_for_user, resolve_site_filter
from sepsis_vitals.db import get_db, is_uuid
from sepsis_vitals.dependencies import verify_auth

router = APIRouter(prefix="/study", tags=["study"])

STUDY_ROLES = frozenset({"researcher", "system_admin"})


def _require_study_role(user: Dict[str, Any]) -> None:
    if user.get("role") not in STUDY_ROLES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Recording study outcomes needs the researcher or system_admin role")


class OutcomeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(..., description="Internal patient id (UUID), not an MRN")
    sepsis: bool = Field(..., description="Outcome under the site's reference standard")
    onset_at: Optional[datetime] = Field(None, description="Adjudicated onset time, with UTC offset")
    died_in_hospital: Optional[bool] = None
    definition: str = Field(..., min_length=1, max_length=64,
                            description="Reference standard and version, e.g. 'Sepsis-3 site protocol v1'")

    @field_validator("onset_at")
    @classmethod
    def _onset_is_aware_and_past(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("onset_at must include a UTC offset")
        if value > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("onset_at is in the future")
        return value


@router.post("/outcomes", status_code=status.HTTP_201_CREATED)
def record_outcome(body: OutcomeIn, db: Session = Depends(get_db),
                   user: Dict[str, Any] = Depends(verify_auth)) -> Dict[str, Any]:
    """Append an adjudicated outcome; the latest row per patient is the one used."""
    from sepsis_vitals.db import StudyOutcome

    _require_study_role(user)
    if not is_uuid(body.patient_id):
        raise HTTPException(status_code=422, detail="patient_id must be the internal patient id (UUID)")
    if body.onset_at is not None and not body.sepsis:
        raise HTTPException(status_code=422, detail="onset_at is only meaningful when sepsis is true")
    patient = load_patient_for_user(body.patient_id, user, db)
    outcome = StudyOutcome(
        patient_id=patient.id, site_id=patient.site_id, sepsis=body.sepsis, onset_at=body.onset_at,
        died_in_hospital=body.died_in_hospital, definition=body.definition,
        recorded_by=user.get("id") if is_uuid(user.get("id")) else None,
    )
    db.add(outcome)
    db.commit()
    return {"id": outcome.id, "patient_id": patient.id}


@router.get("/summary")
def study_summary(site_id: Optional[str] = None, db: Session = Depends(get_db),
                  user: Dict[str, Any] = Depends(verify_auth)) -> Dict[str, Any]:
    """Counts only, for the caller's site."""
    from sepsis_vitals.study.evaluation import load_study_rows

    _require_study_role(user)
    site = resolve_site_filter(user, site_id)
    rows = load_study_rows(db, site)
    with_outcome = [r for r in rows if r["sepsis"] is not None]
    return {
        "site_id": site,
        "patients_with_predictions": len(rows),
        "patients_with_outcome": len(with_outcome),
        "sepsis_positive": sum(1 for r in with_outcome if r["sepsis"]),
    }
