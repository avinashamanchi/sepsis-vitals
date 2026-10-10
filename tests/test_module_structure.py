"""
tests/test_module_structure.py — the api.py and FHIR listener splits keep
their public interfaces.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi missing")

EXPECTED_APP_PATHS = {
    "/health", "/score", "/predict", "/predict/batch", "/patient/{patient_id}/trend",
    "/monitor/register", "/monitor/{patient_id}", "/monitor/status",
    "/simulator/ward", "/simulator/replay", "/simulator/{session_id}", "/simulator/sessions",
    "/simulator/cases", "/ready", "/model/status", "/model/info", "/copilot", "/ws/alerts", "/metrics",
}


def test_every_endpoint_is_still_registered():
    import sepsis_vitals.api as api

    paths = {getattr(r, "path", None) for r in api.app.routes}
    assert EXPECTED_APP_PATHS <= paths


@pytest.mark.parametrize("module", [
    "sepsis_vitals.routes.status", "sepsis_vitals.routes.copilot", "sepsis_vitals.routes.realtime",
    "sepsis_vitals.routes.monitor", "sepsis_vitals.routes.simulator", "sepsis_vitals.routes.metrics",
])
def test_route_modules_can_be_imported_first(module):
    """A direct import must not hit the api <-> routes import cycle half-initialised."""
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]


def test_api_reexports_moved_names():
    import sepsis_vitals.api as api
    from sepsis_vitals.routes import copilot, realtime, status

    assert api._deidentify_vitals is copilot._deidentify_vitals
    assert api.websocket_alerts is realtime.websocket_alerts
    assert api._alembic_head is status._alembic_head


def test_listener_facade_reexports_the_split_modules():
    listener = importlib.import_module("sepsis_vitals.fhir.listener")
    owners = {
        "HL7Parser": "hl7", "FHIRObservationParser": "observation_parser",
        "VitalsIngestionQueue": "ingest_queue", "MLLPServer": "mllp",
        "FHIRWebhookHandler": "webhook", "FHIRWebhookServer": "webhook",
        "VitalsReading": "ingest_models", "LOINC_VITAL_MAP": "ingest_models",
    }
    for name, module in owners.items():
        real = getattr(importlib.import_module(f"sepsis_vitals.fhir.{module}"), name)
        assert getattr(listener, name) is real, name


def test_handlers_with_blocking_database_work_do_not_run_on_the_event_loop():
    """Old audit check 2.1: an ``async def`` handler that queries the database
    synchronously blocks every other request and WebSocket while it waits."""
    import inspect

    from sepsis_vitals.billing import router as billing
    from sepsis_vitals.fhir import router as fhir

    for fn in (fhir.get_patient, fhir.get_observations, fhir.get_risk_assessment,
               billing.create_checkout, billing.create_portal, billing.get_subscription,
               billing.update_beds):
        assert not inspect.iscoroutinefunction(fn), fn.__name__


def test_verify_auth_looks_up_the_user_in_a_worker_thread(monkeypatch):
    import asyncio

    from fastapi import HTTPException
    from starlette.requests import Request

    import sepsis_vitals.api as api

    used = []
    real = asyncio.to_thread

    async def spy(fn, *args, **kwargs):
        used.append(fn.__name__)
        return await real(fn, *args, **kwargs)

    monkeypatch.setattr(api, "_auth_enabled", True)
    monkeypatch.setattr(api.asyncio, "to_thread", spy)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""})
    with pytest.raises(HTTPException) as err:
        asyncio.run(api.verify_auth(request))
    assert err.value.status_code == 401  # no token
    assert used == ["_resolve"]
