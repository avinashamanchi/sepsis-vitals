"""
sepsis_vitals.routes.simulator

Endpoints moved out of sepsis_vitals.api (behaviour unchanged). They
register on the shared ``app`` and reach shared state through ``core`` at
call time, so tests and callers that patch ``sepsis_vitals.api`` still work.
"""

from __future__ import annotations

from typing import Dict
from fastapi import (
    Depends,
    HTTPException,
)
from sepsis_vitals.security import sanitise_string

from sepsis_vitals import api as core

# ---------------------------------------------------------------------------
# Simulator endpoints (gated behind ENABLE_SIMULATOR=true)
# ---------------------------------------------------------------------------


@core.app.post("/simulator/ward", dependencies=[Depends(core.check_rate_limit), Depends(core.check_ml_rate_limit)])
async def simulator_start_ward(body: core.SimulatorWardRequest, user: Dict = Depends(core.verify_auth)):
    """Start a synthetic ward simulation."""
    if not core._simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    _, _, ingester = core._get_monitor_components()
    if ingester is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    manager = core._get_simulation_manager()
    session_id = manager.start_ward(
        ingester=ingester,
        n_patients=body.n_patients,
        speed=body.speed,
        sepsis_count=body.sepsis_count,
        seed=body.seed,
    )

    return {"session_id": session_id, "status": "started"}


@core.app.post("/simulator/replay", dependencies=[Depends(core.check_rate_limit), Depends(core.check_ml_rate_limit)])
async def simulator_start_replay(body: core.SimulatorReplayRequest, user: Dict = Depends(core.verify_auth)):
    """Start a MIMIC-IV case replay."""
    if not core._simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    _, _, ingester = core._get_monitor_components()
    if ingester is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    from sepsis_vitals.ml.case_library import CaseLibrary
    lib = CaseLibrary()

    if body.subject_id == "random" or body.subject_id is None:
        case_meta = lib.get_random_case(sepsis=body.sepsis_only if body.sepsis_only else None)
    else:
        case_meta = lib.get_case(subject_id=int(body.subject_id))

    if case_meta is None:
        raise HTTPException(status_code=404, detail="Case not found")

    # Load vitals timeline for this case
    from sepsis_vitals.ml.mimic_loader import MIMICLoader
    loader = MIMICLoader.from_demo()
    vitals = loader.load_vitals(stay_ids={case_meta["stay_id"]})

    manager = core._get_simulation_manager()
    session_id = manager.start_replay(
        case_meta=case_meta,
        timeline=vitals,
        ingester=ingester,
        speed=body.speed,
    )

    return {"session_id": session_id, "subject_id": case_meta["subject_id"], "status": "started"}


@core.app.delete("/simulator/{session_id}", dependencies=[Depends(core.check_rate_limit)])
async def simulator_stop(session_id: str, user: Dict = Depends(core.verify_auth)):
    """Stop a simulation session."""
    if not core._simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    manager = core._get_simulation_manager()
    stopped = manager.stop_session(sanitise_string(session_id))

    if not stopped:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "status": "stopped"}


@core.app.get("/simulator/sessions", dependencies=[Depends(core.check_rate_limit)])
async def simulator_sessions(user: Dict = Depends(core.verify_auth)):
    """List active simulation sessions."""
    if not core._simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    manager = core._get_simulation_manager()
    return {"sessions": manager.list_sessions()}


@core.app.get("/simulator/cases", dependencies=[Depends(core.check_rate_limit)])
async def simulator_cases(user: Dict = Depends(core.verify_auth)):
    """List available MIMIC-IV cases for replay."""
    if not core._simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    from sepsis_vitals.ml.case_library import CaseLibrary
    lib = CaseLibrary()

    try:
        cases = lib.list_cases()
    except Exception:
        cases = []

    return {"cases": cases, "count": len(cases)}
