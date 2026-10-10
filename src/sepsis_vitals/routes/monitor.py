"""
sepsis_vitals.routes.monitor

Endpoints moved out of sepsis_vitals.api (behaviour unchanged). They
register on the shared ``app`` and reach shared state through ``core`` at
call time, so tests and callers that patch ``sepsis_vitals.api`` still work.
"""

from __future__ import annotations

import asyncio
from typing import Dict
from fastapi import (
    Depends,
)
from sepsis_vitals.security import sanitise_string

from sepsis_vitals import api as core

# ---------------------------------------------------------------------------
# Monitor endpoints (continuous patient monitoring)
# ---------------------------------------------------------------------------


@core.app.post("/monitor/register", dependencies=[Depends(core.check_rate_limit)])
async def monitor_register(body: core.MonitorRegisterRequest, user: Dict = Depends(core.verify_auth)):
    """Register a patient for continuous monitoring."""
    patient_id = sanitise_string(body.patient_id)
    await core._verify_patient_org_async(patient_id, user)

    registry, tracker, ingester = core._get_monitor_components()
    registry.register(
        patient_id,
        demographics=body.demographics,
        comorbidities=body.comorbidities,
    )

    return {"status": "registered", "patient_id": patient_id}


@core.app.delete("/monitor/{patient_id}", dependencies=[Depends(core.check_rate_limit)])
async def monitor_unregister(patient_id: str, user: Dict = Depends(core.verify_auth)):
    """Remove a patient from continuous monitoring."""
    # Org-level authorization: verify patient belongs to user's org (fail closed)
    await core._verify_patient_org_async(patient_id, user)

    registry, tracker, ingester = core._get_monitor_components()
    registry.unregister(sanitise_string(patient_id))
    tracker.remove_patient(sanitise_string(patient_id))

    return {"status": "unregistered", "patient_id": patient_id}


@core.app.get("/monitor/status", dependencies=[Depends(core.check_rate_limit)])
async def monitor_status(user: Dict = Depends(core.verify_auth)):
    """List all monitored patients with current risk and trend."""
    registry, tracker, ingester = core._get_monitor_components()
    patients = registry.list_patients()

    from sepsis_vitals.auth.scope import require_site
    site = require_site(user)
    if site is not None:
        def _site_patient_ids() -> set:
            from sepsis_vitals.db import Patient, SessionLocal
            db = SessionLocal()
            try:
                return {row[0] for row in db.query(Patient.id).filter(Patient.site_id == site)}
            finally:
                db.close()
        allowed = await asyncio.to_thread(_site_patient_ids)
        patients = [p for p in patients if p["patient_id"] in allowed]

    # Enrich with deterioration data
    for p in patients:
        pid = p["patient_id"]
        det = tracker.evaluate(pid)
        p["alert_state"] = det.get("alert_state", "normal")
        p["deterioration_rate"] = det.get("deterioration_rate_per_hour", 0.0)
        p["window_hours"] = det.get("window_hours", 0.0)

    return {"patients": patients, "count": len(patients)}


