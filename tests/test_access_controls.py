"""
tests/test_access_controls.py — billing authorisation and disabled features.

Billing is frozen by default, but its endpoints must still refuse non-admins
(no user is linked to a billing Organization), and disabled features must be
unreachable or explicitly rejected.
"""

from __future__ import annotations

import hashlib
import importlib.util

import pytest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")

BILLING_CALLS = [
    ("post", "/billing/checkout", {"org_id": "00000000-0000-0000-0000-000000000001",
                                   "plan_tier": "clinical", "bed_count": 10,
                                   "success_url": "https://example.org/ok",
                                   "cancel_url": "https://example.org/cancel"}),
    ("post", "/billing/portal", {"org_id": "00000000-0000-0000-0000-000000000001",
                                 "return_url": "https://example.org/back"}),
    ("get", "/billing/subscription?org_id=00000000-0000-0000-0000-000000000001", None),
    ("put", "/billing/beds", {"org_id": "00000000-0000-0000-0000-000000000001", "bed_count": 20}),
]


@pytest.fixture()
def billing_app():
    """The billing router on its own app, with a switchable caller."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import sepsis_vitals.billing.models  # noqa: F401  (register tables)
    from sepsis_vitals.billing import router as billing
    from sepsis_vitals.db import init_db

    init_db()
    app = FastAPI()
    app.include_router(billing.router)
    caller = {"role": "nurse"}

    async def fake_auth(request=None):
        return {"id": "u-1", "email": "x@example.org", "org_id": "SITE-A", **caller}

    async def no_rate_limit(request=None):
        return None

    app.dependency_overrides[billing._require_auth] = fake_auth
    app.dependency_overrides[billing._check_billing_rate] = no_rate_limit
    with TestClient(app) as client:
        yield client, caller


@pytest.mark.parametrize("method,path,body", BILLING_CALLS)
def test_billing_rejects_non_admins(billing_app, method, path, body):
    client, caller = billing_app
    for role in ("nurse", "researcher"):
        caller["role"] = role
        resp = getattr(client, method)(path, **({"json": body} if body else {}))
        assert resp.status_code == 403, (role, path, resp.status_code, resp.text)


@pytest.mark.parametrize("method,path,body", BILLING_CALLS)
def test_billing_admin_passes_authorisation(billing_app, method, path, body):
    client, caller = billing_app
    caller["role"] = "system_admin"
    resp = getattr(client, method)(path, **({"json": body} if body else {}))
    assert resp.status_code == 404, resp.text  # authorised; the org simply does not exist
    assert "00000000-0000-0000-0000-000000000001" not in resp.text


def test_billing_plans_stay_public(billing_app):
    client, _ = billing_app
    assert client.get("/billing/plans").status_code == 200


def test_billing_not_mounted_when_disabled(monkeypatch):
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api

    monkeypatch.delenv("SEPSIS_ENABLE_BILLING", raising=False)
    with TestClient(api.app):
        paths = {getattr(r, "path", "") for r in api.app.routes}
    assert not any(p.startswith("/billing") for p in paths)


def test_routers_are_mounted_once_across_restarts():
    """Restarting the app in one process must not append duplicate routers."""
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api

    with TestClient(api.app):
        pass
    count = len(api.app.routes)
    with TestClient(api.app) as client:
        assert client.post("/auth/login", json={}).status_code != 404  # still mounted
    assert len(api.app.routes) == count


def test_break_glass_is_rejected_even_with_a_valid_token(monkeypatch):
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api

    token = "sealed-envelope-token-for-test"
    monkeypatch.setenv("BREAK_GLASS_TOKEN_HASH", hashlib.sha256(token.encode()).hexdigest())
    api.app.dependency_overrides[api.check_auth_rate_limit] = lambda: None
    try:
        with TestClient(api.app) as client:
            resp = client.post("/auth/break-glass", json={
                "emergency_token": token, "reason": "Network outage during resuscitation",
            })
    finally:
        api.app.dependency_overrides.clear()
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"]
    assert "access_token" not in resp.text
