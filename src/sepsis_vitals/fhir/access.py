"""
sepsis_vitals.fhir.access — site scoping for FHIR reads and writes.

Shared by the FHIR read handlers (``fhir.router``) and ingestion
(``fhir.ingest``). Only ``system_admin`` is unscoped; everyone else reads and
writes patients of their own site, and patients at another site are
reported as not found.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy.orm import Session

from sepsis_vitals.auth.scope import is_unscoped, require_site
from sepsis_vitals.db import Patient
from sepsis_vitals.security import blind_index_candidates


def ingest_site(user: Dict[str, Any]) -> str:
    """Site that newly ingested patients are assigned to.

    Scoped users always ingest into their own site; administrators keep the
    legacy ``"fhir"`` staging site.
    """
    return require_site(user) or "fhir"


def can_access(patient: Patient, user: Dict[str, Any]) -> bool:
    """True when *user* may read or modify *patient* (same site or admin)."""
    if is_unscoped(user):
        return True
    return patient.site_id == require_site(user)


def find_patient(patient_id: str, db: Session, user: Dict[str, Any]) -> Patient | None:
    """Look up a patient by internal id or external_id within the user's site.

    Patients at another site are reported as not found.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        # MRNs are unique per site: scoped users only match their own site.
        query = db.query(Patient).filter(
            Patient.external_id_hash.in_(blind_index_candidates(patient_id))
        )
        site = require_site(user)
        if site is not None:
            query = query.filter(Patient.site_id == site)
        patient = query.first()
    if patient is not None and not can_access(patient, user):
        return None
    return patient


def resolve_patient(ref: str | None, db: Session, user: Dict[str, Any]) -> Patient | None:
    """Resolve a FHIR subject reference to a ``Patient`` row the user may access."""
    if ref is None:
        return None
    # Strip "Patient/" prefix if present
    pid = ref.split("/")[-1] if "/" in ref else ref
    return find_patient(pid, db, user)
