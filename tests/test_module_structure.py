"""
tests/test_module_structure.py — the api.py and FHIR listener splits keep
their public interfaces.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
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


# Sub-router paths: included when sepsis_vitals.api is imported, not at startup.
EXPECTED_SUBROUTER_PATHS = {"/auth/login", "/auth/mfa/enroll", "/patients", "/alerts/history", "/fhir/Patient"}

ROUTE_MODULES = [
    "sepsis_vitals.routes.status", "sepsis_vitals.routes.copilot", "sepsis_vitals.routes.realtime",
    "sepsis_vitals.routes.monitor", "sepsis_vitals.routes.simulator", "sepsis_vitals.routes.metrics",
]
ROUTER_MODULES = [
    "sepsis_vitals.patients.router", "sepsis_vitals.fhir.router", "sepsis_vitals.alerts.router",
    "sepsis_vitals.auth.router", "sepsis_vitals.billing.router",
]


def _registered_paths(app) -> set:
    """HTTP paths from the OpenAPI schema plus WebSocket routes (not in OpenAPI).

    FastAPI 0.142 stores included routers lazily, so ``app.routes`` does not
    list their paths; the schema and ``url_path_for`` do.
    """
    app.openapi_schema = None
    paths = set(app.openapi()["paths"])
    paths.add(app.url_path_for("websocket_alerts"))
    return paths


def _run(code: str, tmp_path) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path}/import-probe.db"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_every_endpoint_is_registered_without_running_startup():
    import sepsis_vitals.api as api

    assert EXPECTED_APP_PATHS | EXPECTED_SUBROUTER_PATHS <= _registered_paths(api.app)


@pytest.mark.parametrize("module", ROUTE_MODULES + ROUTER_MODULES)
def test_router_modules_do_not_import_the_application(module, tmp_path):
    """Routers take shared dependencies from sepsis_vitals.dependencies; none
    imports sepsis_vitals.api, so there is no import cycle to order around."""
    result = _run(
        f"import sys, {module}\n"
        "assert 'sepsis_vitals.api' not in sys.modules, 'imported the app module'\n"
        "import sepsis_vitals.api as api\n"
        "app = api.app\n"
        "assert app.url_path_for('readiness') == '/ready'\n"
        "assert '/auth/login' in app.openapi()['paths']\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr[-800:]


@pytest.mark.parametrize("order", [ROUTE_MODULES, list(reversed(ROUTE_MODULES))])
def test_import_order_does_not_change_the_app(order, tmp_path):
    code = "".join(f"import {m}\n" for m in order) + (
        "import json, sepsis_vitals.api as api\n"
        "print(json.dumps(sorted(api.app.openapi()['paths'])))\n"
    )
    first = _run(code, tmp_path)
    api_first = _run(
        "import json, sepsis_vitals.api as api\n"
        + "".join(f"import {m}\n" for m in order)
        + "print(json.dumps(sorted(api.app.openapi()['paths'])))\n",
        tmp_path,
    )
    assert first.returncode == 0 and api_first.returncode == 0, first.stderr[-500:] + api_first.stderr[-500:]
    assert first.stdout == api_first.stdout


def test_importing_the_app_does_no_database_or_model_work(tmp_path):
    result = _run(
        "import sepsis_vitals.api as api\n"
        "assert api._predictor is None, 'model loaded at import'\n"
        "assert api._model_status['state'] == 'absent'\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert not (tmp_path / "import-probe.db").exists(), "import touched the database"


def test_startup_does_not_duplicate_routes():
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api

    before = (len(api.app.routes), sorted(_registered_paths(api.app)))
    for _ in range(2):
        with TestClient(api.app):
            pass
    assert (len(api.app.routes), sorted(_registered_paths(api.app))) == before


def test_shared_dependencies_are_the_same_objects_everywhere():
    """dependency_overrides keyed on api.check_rate_limit must reach every router."""
    import sepsis_vitals.api as api
    from sepsis_vitals import dependencies
    from sepsis_vitals.fhir import router as fhir
    from sepsis_vitals.patients import router as patients
    from sepsis_vitals.routes import monitor

    for name in ("verify_auth", "check_rate_limit", "check_ml_rate_limit", "check_auth_rate_limit"):
        assert getattr(api, name) is getattr(dependencies, name), name
    assert fhir.verify_auth is patients.verify_auth is monitor.verify_auth is api.verify_auth


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

    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", True)
    monkeypatch.setattr(api.asyncio, "to_thread", spy)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""})
    with pytest.raises(HTTPException) as err:
        asyncio.run(api.verify_auth(request))
    assert err.value.status_code == 401  # no token
    assert used == ["_resolve"]
