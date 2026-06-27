"""
sepsis_vitals.fhir.router -- FastAPI router for HL7 FHIR R4 endpoints.

Provides a standard FHIR interface so EHR systems can send and receive
patient data, vital-sign observations, and sepsis risk assessments using
the ``application/fhir+json`` content type.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from sepsis_vitals.api import verify_auth
from sepsis_vitals.db import Patient, Score, VitalReading, get_db
from sepsis_vitals.fhir.loinc import INTERNAL_TO_ENTRY
from sepsis_vitals.security import compute_blind_index
from sepsis_vitals.fhir.resources import (
    FHIR_CONTENT_TYPE,
    FHIRBundle,
    FHIRObservation,
    FHIRPatient,
    operation_outcome,
    to_fhir_observation,
    to_fhir_patient,
    to_fhir_risk_assessment,
    vitals_from_observations,
)
from sepsis_vitals.scores import compute_scores

# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/fhir", tags=["FHIR R4"])

# FHIR media type for all responses
_FHIR_MEDIA = FHIR_CONTENT_TYPE


def _fhir_response(
    body: dict[str, Any],
    status_code: int = 200,
) -> JSONResponse:
    """Return a ``JSONResponse`` with the FHIR content type.

    If *body* contains the internal ``_http_status`` key (set by
    ``operation_outcome``), it is removed from the payload and used as the
    HTTP status code instead of *status_code*.
    """
    code = body.pop("_http_status", status_code)
    return JSONResponse(content=body, status_code=code, media_type=_FHIR_MEDIA)


def _error(
    status: int,
    code: str,
    message: str,
) -> JSONResponse:
    """Shorthand for returning a FHIR OperationOutcome error response."""
    oo = operation_outcome("error", code, message, http_status=status)
    return _fhir_response(oo)


# ---------------------------------------------------------------------------
# POST /fhir/Patient
# ---------------------------------------------------------------------------


@router.post("/Patient")
async def create_patient(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Receive a FHIR Patient resource and create or update an internal patient."""
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid", "Request body is not valid JSON.")

    try:
        fhir_patient = FHIRPatient.from_fhir(body)
    except ValueError as exc:
        return _error(400, "structure", str(exc))

    internal = fhir_patient.to_internal()

    # Upsert by external_id
    existing = (
        db.query(Patient)
        .filter(Patient.external_id_hash == compute_blind_index(internal["external_id"]))
        .first()
    )

    if existing is not None:
        existing.age_years = internal.get("age_years")
        existing.sex = internal.get("sex", "U")
        existing.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        patient_id = existing.id
        status_code = 200
    else:
        new_patient = Patient(**internal)
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        patient_id = new_patient.id
        status_code = 201

    result = to_fhir_patient({**internal, "id": patient_id})
    return _fhir_response(result, status_code=status_code)


# ---------------------------------------------------------------------------
# POST /fhir/Observation
# ---------------------------------------------------------------------------


@router.post("/Observation")
async def create_observation(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Receive a FHIR Observation resource and record the vital sign."""
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid", "Request body is not valid JSON.")

    try:
        obs = FHIRObservation.from_fhir(body)
    except ValueError as exc:
        return _error(400, "structure", str(exc))

    if obs is None:
        return _error(
            422,
            "not-supported",
            "Observation does not contain a recognised vital sign LOINC code.",
        )

    # Resolve patient
    patient = _resolve_patient(obs.patient_reference, db)
    if patient is None:
        return _error(
            404,
            "not-found",
            f"Patient referenced by '{obs.patient_reference}' not found.",
        )

    # Build a VitalReading row with just this observation
    recorded_at = _parse_datetime(obs.effective_datetime)
    reading_kwargs: dict[str, Any] = {
        "patient_id": patient.id,
        "recorded_at": recorded_at,
        obs.internal_name: obs.value,
    }
    reading = VitalReading(**reading_kwargs)
    db.add(reading)
    db.commit()

    # Feed into monitor if available
    try:
        from sepsis_vitals.api import _get_monitor_components
        _, _, ingester = _get_monitor_components()
        if ingester is not None:
            import asyncio
            asyncio.ensure_future(
                ingester.ingest_single(
                    str(patient.id),
                    {obs.internal_name: obs.value},
                )
            )
    except ImportError:
        pass  # Monitor not available

    fhir_obs = to_fhir_observation(
        vital_name=obs.internal_name,
        value=obs.value,
        patient_ref=str(patient.id),
        timestamp=recorded_at.isoformat(),
    )
    return _fhir_response(fhir_obs, status_code=201)


# ---------------------------------------------------------------------------
# POST /fhir/Bundle
# ---------------------------------------------------------------------------


@router.post("/Bundle")
async def create_bundle(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Receive a FHIR Bundle with Patient and Observation resources.

    Processes all patients first (upsert), then records all observations.
    Returns a FHIR Bundle of type ``transaction-response``.
    """
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid", "Request body is not valid JSON.")

    try:
        bundle = FHIRBundle.from_fhir(body)
    except ValueError as exc:
        return _error(400, "structure", str(exc))

    response_entries: list[dict[str, Any]] = []

    # -- patients ---------------------------------------------------------
    patient_id_map: dict[str, str] = {}  # FHIR resource id -> internal db id

    for fp in bundle.patients:
        internal = fp.to_internal()
        existing = (
            db.query(Patient)
            .filter(Patient.external_id_hash == compute_blind_index(internal["external_id"]))
            .first()
        )
        if existing is not None:
            existing.age_years = internal.get("age_years")
            existing.sex = internal.get("sex", "U")
            existing.updated_at = datetime.now(timezone.utc)
            db.flush()
            patient_id_map[fp.resource_id] = existing.id
            response_entries.append(
                _bundle_response_entry("200 OK", f"Patient/{existing.id}")
            )
        else:
            new_patient = Patient(**internal)
            db.add(new_patient)
            db.flush()
            patient_id_map[fp.resource_id] = new_patient.id
            response_entries.append(
                _bundle_response_entry("201 Created", f"Patient/{new_patient.id}")
            )

    # -- observations -----------------------------------------------------
    for obs in bundle.observations:
        # Try to resolve patient via map first, then DB
        patient_db_id: str | None = None
        if obs.patient_reference:
            patient_db_id = patient_id_map.get(obs.patient_reference)
            if patient_db_id is None:
                patient = _resolve_patient(obs.patient_reference, db)
                if patient is not None:
                    patient_db_id = patient.id

        if patient_db_id is None:
            response_entries.append(
                _bundle_response_entry(
                    "404 Not Found",
                    f"Observation/{obs.resource_id}",
                    outcome_text=(
                        f"Patient '{obs.patient_reference}' not found "
                        f"for Observation/{obs.resource_id}"
                    ),
                )
            )
            continue

        recorded_at = _parse_datetime(obs.effective_datetime)
        reading_kwargs: dict[str, Any] = {
            "patient_id": patient_db_id,
            "recorded_at": recorded_at,
            obs.internal_name: obs.value,
        }
        reading = VitalReading(**reading_kwargs)
        db.add(reading)
        db.flush()

        response_entries.append(
            _bundle_response_entry("201 Created", f"Observation/{reading.id}")
        )

    db.commit()

    response_bundle: dict[str, Any] = {
        "resourceType": "Bundle",
        "type": "transaction-response",
        "entry": response_entries,
    }
    return _fhir_response(response_bundle, status_code=200)


# ---------------------------------------------------------------------------
# GET /fhir/Patient/{id}
# ---------------------------------------------------------------------------


@router.get("/Patient/{patient_id}")
async def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Return a patient as a FHIR Patient resource."""
    patient = _find_patient(patient_id, db)
    if patient is None:
        return _error(404, "not-found", f"Patient '{patient_id}' not found.")

    internal = _patient_to_dict(patient)
    fhir = to_fhir_patient(internal)
    return _fhir_response(fhir)


# ---------------------------------------------------------------------------
# GET /fhir/Patient/{id}/observations
# ---------------------------------------------------------------------------


@router.get("/Patient/{patient_id}/observations")
async def get_observations(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Return the patient's vital-sign readings as a FHIR searchset Bundle."""
    patient = _find_patient(patient_id, db)
    if patient is None:
        return _error(404, "not-found", f"Patient '{patient_id}' not found.")

    readings = (
        db.query(VitalReading)
        .filter(VitalReading.patient_id == patient.id)
        .order_by(VitalReading.recorded_at.desc())
        .limit(100)
        .all()
    )

    entries: list[dict[str, Any]] = []
    for reading in readings:
        ts = reading.recorded_at.isoformat() if reading.recorded_at else None
        for vital_name, entry_meta in INTERNAL_TO_ENTRY.items():
            val = getattr(reading, vital_name, None)
            if val is not None:
                obs = to_fhir_observation(
                    vital_name=vital_name,
                    value=float(val),
                    patient_ref=str(patient.id),
                    timestamp=ts,
                )
                entries.append({"resource": obs})

    bundle: dict[str, Any] = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(entries),
        "entry": entries,
    }
    return _fhir_response(bundle)


# ---------------------------------------------------------------------------
# GET /fhir/RiskAssessment/{patient_id}
# ---------------------------------------------------------------------------


@router.get("/RiskAssessment/{patient_id}")
async def get_risk_assessment(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Return the latest sepsis risk as a FHIR RiskAssessment resource."""
    patient = _find_patient(patient_id, db)
    if patient is None:
        return _error(404, "not-found", f"Patient '{patient_id}' not found.")

    # Find the most recent score
    latest_score = (
        db.query(Score)
        .join(VitalReading, Score.vital_id == VitalReading.id)
        .filter(VitalReading.patient_id == patient.id)
        .order_by(Score.created_at.desc())
        .first()
    )

    if latest_score is None:
        # No scores yet -- compute from latest vitals
        latest_reading = (
            db.query(VitalReading)
            .filter(VitalReading.patient_id == patient.id)
            .order_by(VitalReading.recorded_at.desc())
            .first()
        )
        if latest_reading is None:
            return _error(
                404,
                "not-found",
                f"No vital readings found for patient '{patient_id}'.",
            )

        vitals_dict = _reading_to_vitals(latest_reading)
        scores = compute_scores(vitals_dict)
        prediction = scores.as_dict()
    else:
        prediction = {
            "qsofa": latest_score.qsofa,
            "sirs_count": latest_score.sirs_count,
            "news2_style": latest_score.news2_style,
            "shock_index": latest_score.shock_index,
            "uva_style": latest_score.uva_style,
            "risk_level": latest_score.risk_level,
            "alert_flag": latest_score.alert_flag,
        }

    fhir_ra = to_fhir_risk_assessment(prediction, str(patient.id))
    return _fhir_response(fhir_ra)


# ---------------------------------------------------------------------------
# POST /fhir/$process-vitals  (custom operation)
# ---------------------------------------------------------------------------


@router.post("/$process-vitals")
async def process_vitals(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Custom FHIR operation: receive a vitals Bundle, compute scores, and
    return a RiskAssessment.

    The inbound Bundle should contain at least one Patient and one or more
    Observation resources.  The response is a FHIR RiskAssessment for the
    first patient in the bundle.
    """
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid", "Request body is not valid JSON.")

    try:
        bundle = FHIRBundle.from_fhir(body)
    except ValueError as exc:
        return _error(400, "structure", str(exc))

    if not bundle.observations:
        return _error(
            422,
            "required",
            "Bundle must contain at least one Observation with a "
            "recognised vital sign LOINC code.",
        )

    # Build vitals dict from observations
    vitals_dict = vitals_from_observations(bundle.observations)

    if len(vitals_dict) < 2:
        return _error(
            422,
            "business-rule",
            "At least 2 distinct vital signs are required for scoring.",
        )

    # Compute scores
    scores = compute_scores(vitals_dict)
    prediction = scores.as_dict()

    # Determine patient reference
    patient_ref: str = "Patient/unknown"
    if bundle.patients:
        patient_ref = f"Patient/{bundle.patients[0].resource_id}"
    elif bundle.observations[0].patient_reference:
        patient_ref = f"Patient/{bundle.observations[0].patient_reference}"

    # Persist if patient exists in DB
    patient_db_id: str | None = None
    if bundle.patients:
        fp = bundle.patients[0]
        internal = fp.to_internal()
        existing = (
            db.query(Patient)
            .filter(Patient.external_id_hash == compute_blind_index(internal["external_id"]))
            .first()
        )
        if existing is not None:
            patient_db_id = existing.id
        else:
            new_patient = Patient(**internal)
            db.add(new_patient)
            db.flush()
            patient_db_id = new_patient.id

    if patient_db_id is not None:
        recorded_at = _parse_datetime(
            bundle.observations[0].effective_datetime
        )
        reading_kwargs: dict[str, Any] = {
            "patient_id": patient_db_id,
            "recorded_at": recorded_at,
        }
        for name, val in vitals_dict.items():
            reading_kwargs[name] = val

        reading = VitalReading(**reading_kwargs)
        db.add(reading)
        db.flush()

        score_record = Score(
            vital_id=reading.id,
            qsofa=scores.qsofa,
            sirs_count=scores.sirs_count,
            shock_index=scores.shock_index,
            news2_style=scores.news2_style,
            uva_style=scores.uva_style,
            risk_level=scores.risk_level,
            alert_flag=scores.alert_flag,
        )
        db.add(score_record)
        db.commit()

        patient_ref = f"Patient/{patient_db_id}"

    fhir_ra = to_fhir_risk_assessment(prediction, patient_ref)
    return _fhir_response(fhir_ra)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_patient(patient_id: str, db: Session) -> Patient | None:
    """Look up a patient by internal id or external_id."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        patient = (
            db.query(Patient)
            .filter(Patient.external_id_hash == compute_blind_index(patient_id))
            .first()
        )
    return patient


def _resolve_patient(ref: str | None, db: Session) -> Patient | None:
    """Resolve a FHIR subject reference to a ``Patient`` row."""
    if ref is None:
        return None
    # Strip "Patient/" prefix if present
    pid = ref.split("/")[-1] if "/" in ref else ref
    return _find_patient(pid, db)


def _patient_to_dict(patient: Patient) -> dict[str, Any]:
    """Convert a ``Patient`` ORM model to a plain dict."""
    return {
        "id": patient.id,
        "external_id": patient.external_id,
        "site_id": patient.site_id,
        "age_years": patient.age_years,
        "sex": patient.sex,
    }


def _reading_to_vitals(reading: VitalReading) -> dict[str, float]:
    """Extract non-null vital values from a ``VitalReading`` row."""
    vitals: dict[str, float] = {}
    for name in ("temperature", "heart_rate", "resp_rate", "sbp", "spo2", "gcs"):
        val = getattr(reading, name, None)
        if val is not None:
            vitals[name] = float(val)
    return vitals


def _parse_datetime(iso_str: str | None) -> datetime:
    """Parse an ISO-8601 string or return the current UTC time."""
    if iso_str is None:
        return datetime.now(timezone.utc)
    try:
        # Handle various ISO formats
        cleaned = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _bundle_response_entry(
    status: str,
    location: str,
    outcome_text: str | None = None,
) -> dict[str, Any]:
    """Build a single entry for a Bundle transaction-response."""
    entry: dict[str, Any] = {
        "response": {
            "status": status,
            "location": location,
        }
    }
    if outcome_text is not None:
        entry["response"]["outcome"] = {
            "resourceType": "OperationOutcome",
            "issue": [
                {
                    "severity": "error",
                    "code": "not-found",
                    "diagnostics": outcome_text,
                }
            ],
        }
    return entry
