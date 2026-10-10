"""
sepsis_vitals.fhir.ingest — database work for the FHIR write endpoints.

Execution model
---------------
The POST handlers in ``fhir.router`` read and parse the request body on the
event loop, then hand one *unit of work* to :func:`run_in_worker`:

* The unit of work runs in a worker thread with a session it opens itself.
  It commits at the end; any exception rolls the whole transaction back.
  The session is created, used, committed and closed in that one thread, and
  no blocking database call runs on the event loop.
* A unique-constraint conflict (two requests creating the same MRN at the
  same site at once) or a PostgreSQL deadlock/serialisation failure rolls
  back and re-runs the unit of work once. The second pass sees the other
  request's committed rows: a duplicate create becomes an update.
* Readings are de-duplicated. An observation with the same patient, vital,
  effective time and value as a stored reading is acknowledged (HTTP 200)
  without a second row, so client retries and replays do not double-count.
  Writers for one patient are serialised by a per-patient lock in this
  process and by ``SELECT ... FOR UPDATE`` on the patient row (PostgreSQL;
  SQLite ignores it, and SQLite is single-host development only). Locks are
  taken in sorted order, before any write.
* Observations without ``effectiveDateTime`` are timestamped on receipt and
  therefore cannot be recognised as replays.
* If the client disconnects, the worker thread still finishes: threads
  cannot be cancelled, and the transaction commits or rolls back as a whole.

Values are checked against the same plausibility bounds as ``/predict``
(``sepsis_vitals.schemas.VitalsInput``); these are input-validation limits,
not clinical thresholds.
"""

from __future__ import annotations

import logging
import math
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from sepsis_vitals.db import Patient, Score, SessionLocal, VitalReading
from sepsis_vitals.fhir.access import can_access, ingest_site, resolve_patient
from sepsis_vitals.fhir.resources import FHIRBundle, FHIRObservation, FHIRPatient, vitals_from_observations
from sepsis_vitals.schemas import VitalsInput
from sepsis_vitals.security import blind_index_candidates

logger = logging.getLogger(__name__)

T = TypeVar("T")

PATIENT_LOCK_TIMEOUT_S = 10.0
TRANSIENT_MESSAGE = "The data could not be stored and no changes were made. Retry the request."

_patient_locks: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
_patient_locks_guard = threading.Lock()


class IngestError(Exception):
    """A request-level failure, reported to the client as an OperationOutcome."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class Txn:
    """One ingestion transaction: its session and the patient locks it holds."""

    db: Session
    _held: List[Any] = field(default_factory=list)
    _locked: bool = False

    def lock_patients(self, patient_ids: Iterable[str]) -> None:
        """Serialise writers for these patients until this transaction ends.

        Call once per transaction, before writing (locks are taken in sorted
        order, so two transactions never wait on each other in a cycle).
        """
        if self._locked:
            raise RuntimeError("lock_patients may be called once per transaction")
        self._locked = True
        for pid in sorted(set(patient_ids)):
            with _patient_locks_guard:
                lock = _patient_locks.get(pid)
                if lock is None:
                    lock = threading.Lock()
                    _patient_locks[pid] = lock
            if not lock.acquire(timeout=PATIENT_LOCK_TIMEOUT_S):
                raise IngestError(503, "transient", TRANSIENT_MESSAGE)
            self._held.append(lock)
            # Other processes: hold the patient row lock until commit.
            self.db.query(Patient.id).filter(Patient.id == pid).with_for_update().first()

    def release(self) -> None:
        while self._held:
            self._held.pop().release()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, IntegrityError):
        return True
    code = getattr(getattr(exc, "orig", None), "sqlstate", None) or getattr(getattr(exc, "orig", None), "pgcode", None)
    return code in {"40P01", "40001"}  # deadlock detected, serialization failure


def run_in_worker(work: Callable[[Txn], T]) -> T:
    """Run *work* in its own session and transaction (call via ``asyncio.to_thread``)."""
    for attempt in (1, 2):
        db = SessionLocal()
        txn = Txn(db)
        try:
            result = work(txn)
            db.commit()
            return result
        except DBAPIError as exc:
            db.rollback()
            if attempt == 2 or not _is_retryable(exc):
                raise
        except BaseException:
            db.rollback()
            raise
        finally:
            txn.release()
            db.close()
    raise AssertionError("unreachable")


# -- validation ---------------------------------------------------------------------

def effective_time(value: Any) -> datetime:
    """UTC timestamp from effectiveDateTime; the receipt time when absent."""
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise IngestError(400, "value", "effectiveDateTime is not a valid ISO-8601 date-time.") from None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def check_values(values: Dict[str, float]) -> None:
    """Reject non-finite values and values outside the API's input bounds."""
    for name, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise IngestError(422, "value", f"{name} is not a finite number.")
        if name in VitalsInput.model_fields:
            try:
                VitalsInput.model_validate({name: value})
            except ValidationError:
                raise IngestError(422, "value", f"{name} is outside the range this API accepts.") from None


def parse(kind: Any, body: Dict[str, Any]) -> Any:
    """``kind.from_fhir(body)`` with every structural error reported as HTTP 400."""
    try:
        return kind.from_fhir(body)
    except ValueError as exc:
        raise IngestError(400, "structure", str(exc)) from None
    except (TypeError, AttributeError, KeyError):
        raise IngestError(400, "structure", f"The {kind.__name__[4:]} resource is malformed.") from None


# -- shared steps --------------------------------------------------------------------

def _lookup(db: Session, internal: Dict[str, Any]) -> Optional[Patient]:
    return (
        db.query(Patient)
        .filter(
            Patient.site_id == internal["site_id"],
            Patient.external_id_hash.in_(blind_index_candidates(internal["external_id"])),
        )
        .first()
    )


def _upsert(db: Session, internal: Dict[str, Any], existing: Optional[Patient]) -> Tuple[str, bool]:
    """Update *existing* or insert a new patient. Returns (patient id, created)."""
    if existing is not None:
        existing.age_years = internal.get("age_years")
        existing.sex = internal.get("sex", "U")
        existing.updated_at = datetime.now(timezone.utc)
        db.flush()
        return existing.id, False
    patient = Patient(**internal)
    db.add(patient)
    db.flush()  # a concurrent insert of the same MRN fails here; run_in_worker retries
    return patient.id, True


def _same_reading(db: Session, patient_id: str, recorded_at: datetime, values: Dict[str, float]) -> Optional[VitalReading]:
    query = db.query(VitalReading).filter(
        VitalReading.patient_id == patient_id, VitalReading.recorded_at == recorded_at
    )
    for name, value in values.items():
        query = query.filter(getattr(VitalReading, name) == value)
    return query.first()


def _record(db: Session, patient_id: str, recorded_at: datetime, values: Dict[str, float]) -> Tuple[str, bool]:
    """Insert a reading unless the same one is stored. Returns (reading id, created)."""
    existing = _same_reading(db, patient_id, recorded_at, values)
    if existing is not None:
        return existing.id, False
    reading = VitalReading(patient_id=patient_id, recorded_at=recorded_at, **values)
    db.add(reading)
    db.flush()
    return reading.id, True


# -- units of work -------------------------------------------------------------------

def patient_work(fhir_patient: FHIRPatient, user: Dict[str, Any]) -> Callable[[Txn], Tuple[Dict[str, Any], bool]]:
    internal = fhir_patient.to_internal(site_id=ingest_site(user))

    def work(txn: Txn) -> Tuple[Dict[str, Any], bool]:
        existing = _lookup(txn.db, internal)
        if existing is not None and not can_access(existing, user):
            raise IngestError(409, "conflict", "Patient could not be created.")
        patient_id, created = _upsert(txn.db, internal, existing)
        return {**internal, "id": patient_id}, created

    return work


@dataclass
class ObservationResult:
    patient_id: str
    reading_id: str
    recorded_at: datetime
    created: bool


def observation_work(obs: FHIRObservation, user: Dict[str, Any]) -> Callable[[Txn], ObservationResult]:
    recorded_at = effective_time(obs.effective_datetime)
    values = {obs.internal_name: obs.value}
    check_values(values)

    def work(txn: Txn) -> ObservationResult:
        patient = resolve_patient(obs.patient_reference, txn.db, user)
        if patient is None:
            raise IngestError(404, "not-found", f"Patient referenced by '{obs.patient_reference}' not found.")
        txn.lock_patients([patient.id])
        reading_id, created = _record(txn.db, patient.id, recorded_at, values)
        return ObservationResult(patient.id, reading_id, recorded_at, created)

    return work


def bundle_work(bundle: FHIRBundle, user: Dict[str, Any]) -> Callable[[Txn], List[Dict[str, Any]]]:
    """Patients first (upsert), then observations; one transaction for all.

    Observations whose patient is not found get a 404 entry and the rest are
    stored; a patient owned by another site aborts the whole bundle.
    """
    site = ingest_site(user)
    patients = [(fp, fp.to_internal(site_id=site)) for fp in bundle.patients]
    observations = [(obs, effective_time(obs.effective_datetime)) for obs in bundle.observations]
    for obs, _ in observations:
        check_values({obs.internal_name: obs.value})

    def work(txn: Txn) -> List[Dict[str, Any]]:
        db = txn.db
        # 1. Read: existing patients and referenced patients outside the bundle.
        existing = {fp.resource_id: _lookup(db, internal) for fp, internal in patients}
        for found in existing.values():
            if found is not None and not can_access(found, user):
                raise IngestError(409, "conflict", "Bundle references a patient that could not be created.")
        referenced: Dict[str, Optional[str]] = {}
        for obs, _ in observations:
            ref = obs.patient_reference
            if ref and ref not in existing and ref not in referenced:
                found = resolve_patient(ref, db, user)
                referenced[ref] = found.id if found is not None else None

        # 2. Lock every existing patient this bundle writes for, before writing.
        txn.lock_patients(
            [p.id for p in existing.values() if p is not None] + [pid for pid in referenced.values() if pid]
        )

        # 3. Write.
        entries: List[Dict[str, Any]] = []
        id_map: Dict[str, str] = {}
        for fp, internal in patients:
            new_id, created = _upsert(db, internal, existing[fp.resource_id])
            id_map[fp.resource_id] = new_id
            entries.append(bundle_entry("201 Created" if created else "200 OK", f"Patient/{new_id}"))
        for obs, recorded_at in observations:
            ref = obs.patient_reference
            patient_id: Optional[str] = id_map.get(ref) if ref else None
            if patient_id is None and ref:
                patient_id = referenced.get(ref)
            if patient_id is None:
                entries.append(bundle_entry(
                    "404 Not Found", f"Observation/{obs.resource_id}",
                    outcome_text=f"Patient '{ref}' not found for Observation/{obs.resource_id}",
                ))
                continue
            reading_id, created = _record(db, patient_id, recorded_at, {obs.internal_name: obs.value})
            entries.append(bundle_entry("201 Created" if created else "200 OK", f"Observation/{reading_id}"))
        return entries

    return work


@dataclass
class ProcessVitalsResult:
    patient_id: Optional[str]
    created_reading: bool


def process_vitals_work(bundle: FHIRBundle, user: Dict[str, Any], scores: Any) -> Callable[[Txn], ProcessVitalsResult]:
    """Persist the combined reading and its scores for the bundle's first patient."""
    vitals = vitals_from_observations(bundle.observations)
    check_values(vitals)
    recorded_at = effective_time(bundle.observations[0].effective_datetime)
    internal = bundle.patients[0].to_internal(site_id=ingest_site(user)) if bundle.patients else None

    def work(txn: Txn) -> ProcessVitalsResult:
        if internal is None:
            return ProcessVitalsResult(None, False)
        db = txn.db
        existing = _lookup(db, internal)
        if existing is not None and not can_access(existing, user):
            raise IngestError(409, "conflict", "Patient could not be created.")
        if existing is not None:
            txn.lock_patients([existing.id])
            patient_id = existing.id
        else:
            patient = Patient(**internal)
            db.add(patient)
            db.flush()
            patient_id = patient.id
        reading_id, created = _record(db, patient_id, recorded_at, vitals)
        if created or db.query(Score.id).filter(Score.vital_id == reading_id).first() is None:
            db.add(Score(
                vital_id=reading_id,
                qsofa=scores.qsofa,
                sirs_count=scores.sirs_count,
                shock_index=scores.shock_index,
                news2_style=scores.news2_style,
                uva_style=scores.uva_style,
                risk_level=scores.risk_level,
                alert_flag=scores.alert_flag,
            ))
            db.flush()
        return ProcessVitalsResult(patient_id, created)

    return work


def bundle_entry(status: str, location: str, outcome_text: Optional[str] = None) -> Dict[str, Any]:
    """One entry of a Bundle transaction-response."""
    entry: Dict[str, Any] = {"response": {"status": status, "location": location}}
    if outcome_text is not None:
        entry["response"]["outcome"] = {
            "resourceType": "OperationOutcome",
            "issue": [{"severity": "error", "code": "not-found", "diagnostics": outcome_text}],
        }
    return entry
