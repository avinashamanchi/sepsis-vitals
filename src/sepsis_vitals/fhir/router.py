"""
sepsis_vitals.fhir.router -- FastAPI router for HL7 FHIR R4 endpoints.

Provides a standard FHIR interface so EHR systems can send and receive
patient data, vital-sign observations, and sepsis risk assessments using
the ``application/fhir+json`` content type.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sepsis_vitals.db import Patient, Score, VitalReading, get_db
from sepsis_vitals.dependencies import verify_auth
from sepsis_vitals.fhir.access import can_access, find_patient, ingest_site, resolve_patient
from sepsis_vitals.fhir.ingest import (
    TRANSIENT_MESSAGE,
    IngestError,
    Txn,
    bundle_work,
    check_values,
    observation_work,
    parse,
    patient_work,
    process_vitals_work,
    run_in_worker,
)
from sepsis_vitals.fhir.loinc import INTERNAL_TO_ENTRY
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

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Site-scoping helpers (now in fhir.access), under their former names.
_ingest_site = ingest_site
_can_access = can_access
_find_patient = find_patient
_resolve_patient = resolve_patient


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
# GET /fhir/metadata
# ---------------------------------------------------------------------------


@router.get("/metadata", summary="FHIR CapabilityStatement")
async def capability_statement() -> JSONResponse:
    """Return FHIR R4 CapabilityStatement describing this server's capabilities."""
    capability_statement_resource = {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": "2026-06-27",
        "kind": "instance",
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "rest": [{
            "mode": "server",
            "resource": [
                {
                    "type": "Patient",
                    "interaction": [
                        {"code": "read"},
                        {"code": "create"},
                    ],
                },
                {
                    "type": "Observation",
                    "interaction": [
                        {"code": "read"},
                        {"code": "create"},
                        {"code": "search-type"},
                    ],
                },
                {
                    "type": "RiskAssessment",
                    "interaction": [
                        {"code": "read"},
                    ],
                },
                {
                    "type": "Bundle",
                    "interaction": [
                        {"code": "create"},
                    ],
                },
            ],
            "operation": [
                {
                    "name": "process-vitals",
                    "definition": "OperationDefinition/process-vitals",
                },
            ],
        }],
    }
    return _fhir_response(capability_statement_resource)


# ---------------------------------------------------------------------------
# Write endpoints: execution model in sepsis_vitals.fhir.ingest
# ---------------------------------------------------------------------------


_monitor_tasks: set = set()


async def _read_resource(request: Request) -> Dict[str, Any]:
    """The request body as one FHIR resource (a JSON object)."""
    try:
        body = await request.json()
    except Exception:
        raise IngestError(400, "invalid", "Request body is not valid JSON.") from None
    if not isinstance(body, dict):
        raise IngestError(400, "structure", "Request body must be a single FHIR resource (a JSON object).")
    return body


async def _ingest(work: Callable[[Txn], T], what: str) -> T:
    """Run a unit of work in a worker thread with its own session."""
    try:
        return await asyncio.to_thread(run_in_worker, work)
    except (IngestError, HTTPException):
        raise
    except SQLAlchemyError as exc:
        logger.error("FHIR %s ingestion failed and was rolled back (%s)", what, type(exc).__name__)
        raise IngestError(503, "transient", TRANSIENT_MESSAGE) from exc


def _monitor_task_done(task: "asyncio.Task[Any]") -> None:
    _monitor_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Monitor ingestion failed (%s)", type(task.exception()).__name__)


async def _feed_monitor(patient_id: str, values: Dict[str, float]) -> None:
    """Pass a newly stored reading to the continuous monitor, if it runs."""
    try:
        from sepsis_vitals.api import _get_monitor_components

        # The first call may load the model: keep it off the event loop.
        _, _, ingester = await asyncio.to_thread(_get_monitor_components)
    except Exception as exc:
        logger.warning("Monitor unavailable (%s)", type(exc).__name__)
        return
    if ingester is None:
        return
    task = asyncio.ensure_future(ingester.ingest_single(str(patient_id), values))
    _monitor_tasks.add(task)  # keep a reference until it finishes
    task.add_done_callback(_monitor_task_done)


@router.post("/Patient")
async def create_patient(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Receive a FHIR Patient resource and create or update an internal patient.

    201 when created, 200 when the MRN already exists at the caller's site
    (including a concurrent duplicate create).
    """
    try:
        fhir_patient = parse(FHIRPatient, await _read_resource(request))
        result, created = await _ingest(patient_work(fhir_patient, current_user), "Patient")
    except IngestError as err:
        return _error(err.status, err.code, err.message)
    return _fhir_response(to_fhir_patient(result), status_code=201 if created else 200)


@router.post("/Observation")
async def create_observation(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Receive a FHIR Observation resource and record the vital sign.

    201 when stored; 200 when the same reading (patient, vital, effective
    time and value) is already stored, so retries do not double-count.
    """
    try:
        obs = parse(FHIRObservation, await _read_resource(request))
        if obs is None:
            return _error(
                422,
                "not-supported",
                "Observation does not contain a recognised vital sign LOINC code.",
            )
        result = await _ingest(observation_work(obs, current_user), "Observation")
    except IngestError as err:
        return _error(err.status, err.code, err.message)

    if result.created:
        await _feed_monitor(result.patient_id, {obs.internal_name: obs.value})

    fhir_obs = to_fhir_observation(
        vital_name=obs.internal_name,
        value=obs.value,
        patient_ref=str(result.patient_id),
        timestamp=result.recorded_at.isoformat(),
    )
    return _fhir_response(fhir_obs, status_code=201 if result.created else 200)


@router.post("/Bundle")
async def create_bundle(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Receive a FHIR Bundle with Patient and Observation resources.

    Processes all patients first (upsert), then records all observations, in
    one transaction. Returns a FHIR Bundle of type ``transaction-response``.
    """
    try:
        bundle = parse(FHIRBundle, await _read_resource(request))
        entries = await _ingest(bundle_work(bundle, current_user), "Bundle")
    except IngestError as err:
        return _error(err.status, err.code, err.message)
    return _fhir_response(
        {"resourceType": "Bundle", "type": "transaction-response", "entry": entries},
        status_code=200,
    )


# ---------------------------------------------------------------------------
# GET /fhir/Patient/{id}
# ---------------------------------------------------------------------------


@router.get("/Patient/{patient_id}")
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Return a patient as a FHIR Patient resource."""
    patient = _find_patient(patient_id, db, current_user)
    if patient is None:
        return _error(404, "not-found", f"Patient '{patient_id}' not found.")

    internal = _patient_to_dict(patient)
    fhir = to_fhir_patient(internal)
    return _fhir_response(fhir)


# ---------------------------------------------------------------------------
# GET /fhir/Patient/{id}/observations
# ---------------------------------------------------------------------------


@router.get("/Patient/{patient_id}/observations")
def get_observations(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Return the patient's vital-sign readings as a FHIR searchset Bundle."""
    patient = _find_patient(patient_id, db, current_user)
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
def get_risk_assessment(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Return the latest sepsis risk as a FHIR RiskAssessment resource."""
    patient = _find_patient(patient_id, db, current_user)
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
    current_user: Dict[str, Any] = Depends(verify_auth),
) -> JSONResponse:
    """Custom FHIR operation: receive a vitals Bundle, compute scores, and
    return a RiskAssessment.

    The inbound Bundle should contain at least one Patient and one or more
    Observation resources.  The response is a FHIR RiskAssessment for the
    first patient in the bundle; the combined reading and its scores are
    stored for that patient (a replayed bundle is not stored twice).
    """
    try:
        bundle = parse(FHIRBundle, await _read_resource(request))
    except IngestError as err:
        return _error(err.status, err.code, err.message)

    if not bundle.observations:
        return _error(
            422,
            "required",
            "Bundle must contain at least one Observation with a "
            "recognised vital sign LOINC code.",
        )

    vitals_dict = vitals_from_observations(bundle.observations)
    if len(vitals_dict) < 2:
        return _error(
            422,
            "business-rule",
            "At least 2 distinct vital signs are required for scoring.",
        )

    try:
        check_values(vitals_dict)
        scores = compute_scores(vitals_dict)
        result = await _ingest(process_vitals_work(bundle, current_user, scores), "process-vitals")
    except IngestError as err:
        return _error(err.status, err.code, err.message)

    if result.patient_id is not None:
        patient_ref = f"Patient/{result.patient_id}"
    elif bundle.observations[0].patient_reference:
        patient_ref = f"Patient/{bundle.observations[0].patient_reference}"
    else:
        patient_ref = "Patient/unknown"

    return _fhir_response(to_fhir_risk_assessment(scores.as_dict(), patient_ref))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


