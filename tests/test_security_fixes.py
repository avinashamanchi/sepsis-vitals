"""
tests/test_security_fixes.py — Regression tests for v1/v2 audit security fixes.

Covers: JWT key loading, WebSocket org scoping, FieldEncryptor production guard,
NEWS2 Scale 2, verify_patient_org (when fastapi available).
"""
import asyncio
import importlib.util
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if fastapi is available (not installed in all test envs)
HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


# ---------------------------------------------------------------------------
# M1: verify_patient_org — null site_id policy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestVerifyPatientOrg:
    """Verify org-level authorization handles edge cases."""

    def _make_patient(self, site_id):
        p = MagicMock()
        p.site_id = site_id
        return p

    def _call(self, patient_id, user, db):
        from sepsis_vitals.api import verify_patient_org
        verify_patient_org(patient_id, user, db)

    def test_system_admin_without_org_allowed(self):
        """system_admin (incl. the auth-disabled dev identity) is unscoped."""
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = self._make_patient("org-b")
        self._call("P1", {"role": "system_admin", "org_id": None}, db)

    def test_scoped_user_without_org_rejected(self):
        """Fail closed: a non-admin with no site assignment sees nothing."""
        from fastapi import HTTPException
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = self._make_patient("org-a")
        for role in ("nurse", "researcher", None):
            with pytest.raises(HTTPException) as exc_info:
                self._call("P1", {"role": role, "org_id": None}, db)
            assert exc_info.value.status_code == 403

    def test_matching_org_allowed(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = self._make_patient("org-a")
        self._call("P1", {"org_id": "org-a"}, db)

    def test_mismatched_org_rejected(self):
        from fastapi import HTTPException
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = self._make_patient("org-b")
        with pytest.raises(HTTPException) as exc_info:
            self._call("P1", {"org_id": "org-a"}, db)
        assert exc_info.value.status_code == 404

    def test_null_site_id_rejected_for_authed_user(self):
        """Patient with site_id=None must be rejected when user has an org."""
        from fastapi import HTTPException
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = self._make_patient(None)
        with pytest.raises(HTTPException) as exc_info:
            self._call("P1", {"org_id": "org-a"}, db)
        assert exc_info.value.status_code == 404

    def test_nonexistent_patient_rejected(self):
        from fastapi import HTTPException
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            self._call("P1", {"org_id": "org-a"}, db)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# M2: _load_keys() thread safety
# ---------------------------------------------------------------------------

class TestLoadKeysThreadSafety:
    """Verify _load_keys uses proper locking."""

    def test_concurrent_loads_no_crash(self):
        """Multiple threads calling _load_keys should not raise."""
        import sepsis_vitals.auth.tokens as tokens_mod

        # Reset state
        tokens_mod._KEYS_LOADED = False
        tokens_mod._SECRET_KEY = None
        tokens_mod._RSA_PRIVATE_KEY = None
        tokens_mod._RSA_PUBLIC_KEY = None

        errors = []

        with patch.dict(os.environ, {"SEPSIS_JWT_SECRET": "test-secret-key-1234"}):
            def load():
                try:
                    tokens_mod._load_keys()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=load) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(errors) == 0
        assert tokens_mod._KEYS_LOADED is True

        # Cleanup
        tokens_mod._KEYS_LOADED = False
        tokens_mod._SECRET_KEY = None

    def test_keys_loaded_only_after_assignment(self):
        """_KEYS_LOADED must be True only after key variables are set."""
        import sepsis_vitals.auth.tokens as tokens_mod

        tokens_mod._KEYS_LOADED = False
        tokens_mod._SECRET_KEY = None
        tokens_mod._RSA_PRIVATE_KEY = None
        tokens_mod._RSA_PUBLIC_KEY = None

        with patch.dict(os.environ, {"SEPSIS_JWT_SECRET": "my-secret"}):
            tokens_mod._load_keys()

        assert tokens_mod._KEYS_LOADED is True
        assert tokens_mod._SECRET_KEY == "my-secret"

        # Cleanup
        tokens_mod._KEYS_LOADED = False
        tokens_mod._SECRET_KEY = None


# ---------------------------------------------------------------------------
# H1: WebSocket org scoping — fail-closed
# ---------------------------------------------------------------------------

class TestWebSocketOrgScoping:
    """WebSocket broadcast must not leak to wrong orgs."""

    def test_unresolvable_patient_blocked_for_authed_connections(self):
        """When patient org can't be resolved, authed connections get nothing."""
        from sepsis_vitals.realtime.websocket import ConnectionManager

        mgr = ConnectionManager()
        ws_authed = AsyncMock()
        ws_demo = AsyncMock()

        mgr._connections = [(ws_authed, "org-a"), (ws_demo, None)]

        with patch.object(mgr, "_patient_org_id", return_value=None):
            asyncio.run(
                mgr.broadcast({"patient_id": "P999", "type": "alert"})
            )

        ws_authed.send_text.assert_not_called()
        ws_demo.send_text.assert_called_once()

    def test_matching_org_receives_message(self):
        from sepsis_vitals.realtime.websocket import ConnectionManager

        mgr = ConnectionManager()
        ws = AsyncMock()

        mgr._connections = [(ws, "org-a")]

        with patch.object(mgr, "_patient_org_id", return_value="org-a"):
            asyncio.run(
                mgr.broadcast({"patient_id": "P1", "type": "alert"})
            )

        ws.send_text.assert_called_once()

    def test_mismatched_org_blocked(self):
        from sepsis_vitals.realtime.websocket import ConnectionManager

        mgr = ConnectionManager()
        ws = AsyncMock()

        mgr._connections = [(ws, "org-b")]

        with patch.object(mgr, "_patient_org_id", return_value="org-a"):
            asyncio.run(
                mgr.broadcast({"patient_id": "P1", "type": "alert"})
            )

        ws.send_text.assert_not_called()


# ---------------------------------------------------------------------------
# M6: FieldEncryptor production guard
# ---------------------------------------------------------------------------

class TestFieldEncryptorProductionGuard:
    """FieldEncryptor must refuse to start without key in production."""

    def test_missing_key_in_production_raises(self):
        from sepsis_vitals.security import FieldEncryptor

        # Reset singleton
        FieldEncryptor._instance = None
        FieldEncryptor._key = None

        with patch.dict(os.environ, {"SEPSIS_ENV": "production", "SEPSIS_PII_KEY": ""}, clear=False):
            with pytest.raises(RuntimeError, match="production"):
                FieldEncryptor._load_key()

        # Cleanup
        FieldEncryptor._instance = None
        FieldEncryptor._key = None

    def test_missing_key_in_dev_warns(self):
        from sepsis_vitals.security import FieldEncryptor

        FieldEncryptor._instance = None
        FieldEncryptor._key = None

        with patch.dict(os.environ, {"SEPSIS_ENV": "development", "SEPSIS_PII_KEY": ""}, clear=False):
            FieldEncryptor._load_key()
            assert FieldEncryptor._key is None

        FieldEncryptor._instance = None
        FieldEncryptor._key = None


# ---------------------------------------------------------------------------
# M5: NEWS2 Scale 2
# ---------------------------------------------------------------------------

class TestNEWS2Scale2:
    """NEWS2 oxygen handling per the RCP 2017 chart.

    Supplemental oxygen adds 2 points and does not by itself select Scale 2;
    Scale 2 is reserved for patients with a prescribed 88-92% target.
    """

    def _n2(self, **v):
        from sepsis_vitals.scores import news2_style
        return news2_style(v)

    def test_scale1_default(self):
        assert self._n2(spo2=94) == 1
        assert self._n2(spo2=96) == 0

    def test_supplemental_oxygen_adds_two_points(self):
        assert self._n2(spo2=97, on_supplemental_o2=True) == 2

    def test_hypoxic_patient_on_oxygen_is_not_underscored(self):
        # Regression: previously "on oxygen" switched to Scale 2 and scored 0.
        assert self._n2(spo2=90, on_supplemental_o2=True) == 3 + 2

    def test_scale2_target_range_scores_zero(self):
        assert self._n2(spo2=90, spo2_scale2=True, on_supplemental_o2=True) == 0 + 2

    def test_scale2_high_saturation_scored_only_on_oxygen(self):
        assert self._n2(spo2=97, spo2_scale2=True, on_supplemental_o2=True) == 3 + 2
        assert self._n2(spo2=95, spo2_scale2=True, on_supplemental_o2=True) == 2 + 2
        assert self._n2(spo2=93, spo2_scale2=True, on_supplemental_o2=True) == 1 + 2
        assert self._n2(spo2=97, spo2_scale2=True) == 0  # on air

    def test_scale2_low_bands(self):
        assert self._n2(spo2=83, spo2_scale2=True) == 3
        assert self._n2(spo2=85, spo2_scale2=True) == 2
        assert self._n2(spo2=87, spo2_scale2=True) == 1
        assert self._n2(spo2=88, spo2_scale2=True) == 0
