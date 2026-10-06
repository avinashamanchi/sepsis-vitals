"""
sepsis_vitals.auth.scope
~~~~~~~~~~~~~~~~~~~~~~~~
Tenant (site) scoping shared by every router that touches patient data.

Policy (fail closed):

* ``system_admin`` is the only unscoped role. It may read any site and may
  pass an explicit ``site_id`` filter. When auth is disabled in development
  the anonymous identity is a ``system_admin``, so local workflows still work.
* Every other role is restricted to ``user["org_id"]`` (the ``User.site_id``
  column). A caller-supplied ``site_id`` for another site is rejected.
* A scoped user with no site assignment gets no patient data at all; an
  administrator must assign a site first.

Cross-site lookups return 404 rather than 403 so that patient existence at
another site is not disclosed.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from fastapi import HTTPException, status

UNSCOPED_ROLES = frozenset({"system_admin"})

_NOT_FOUND = "Patient not found"


def is_unscoped(user: Mapping[str, Any]) -> bool:
    """Return True when *user* may access every site."""
    return user.get("role") in UNSCOPED_ROLES


def require_site(user: Mapping[str, Any]) -> Optional[str]:
    """Return the site *user* is restricted to, or None for unscoped users.

    Raises 403 for a scoped user without a site assignment.
    """
    if is_unscoped(user):
        return None
    site = user.get("org_id")
    if not site:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has no site assignment; contact an administrator",
        )
    return str(site)


def resolve_site_filter(user: Mapping[str, Any], requested: Optional[str]) -> Optional[str]:
    """Return the site filter to apply to a list or aggregate query.

    Unscoped users get *requested* unchanged (None means all sites). Scoped
    users always get their own site; requesting a different one is a 404.
    """
    site = require_site(user)
    if site is None:
        return requested
    if requested is not None and requested != site:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Site not found")
    return site


def ensure_patient_access(patient: Any, user: Mapping[str, Any]) -> Any:
    """Return *patient* if *user* may access it; otherwise raise 404."""
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    site = require_site(user)
    if site is not None and getattr(patient, "site_id", None) != site:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return patient


def ensure_site_write(user: Mapping[str, Any], target_site: Optional[str]) -> None:
    """Reject creating or moving a record into a site the user does not own."""
    site = require_site(user)
    if site is not None and target_site is not None and target_site != site:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot write records for another site",
        )


def load_patient_for_user(patient_id: str, user: Mapping[str, Any], db: Any) -> Any:
    """Load a patient by primary key and enforce site access."""
    from sepsis_vitals.db import Patient

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    return ensure_patient_access(patient, user)


def ensure_not_foreign_patient(patient_id: str, user: Mapping[str, Any], db: Any) -> None:
    """Reject writes that target a registered patient at another site.

    Free-form identifiers that are not registered patients (e.g. research
    scenarios on the Predict page) remain allowed.
    """
    from sepsis_vitals.db import Patient

    site = require_site(user)
    if site is None:
        return
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is not None and patient.site_id != site:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
