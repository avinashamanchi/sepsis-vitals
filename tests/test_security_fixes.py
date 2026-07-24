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

    def test_demo_user_allowed(self):
        """Users with org_id=None (demo) can access any patient."""
        db = MagicMock()
        self._call("P1", {"org_id": None}, db)
        db.query.assert_not_called()

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
    """NEWS2 scoring must support Scale 2 for supplemental O2 patients."""

    def test_scale1_default(self):
        from sepsis_vitals.scores import news2_style
        # SpO2 94% under Scale 1 = score 1
        score = news2_style({"spo2": 94})
        assert score == 1

    def test_scale2_target_range(self):
        from sepsis_vitals.scores import news2_style
        # SpO2 90% under Scale 2 = score 0 (in target 88-92)
        score = news2_style({"spo2": 90, "on_supplemental_o2": True})
        assert score == 0

    def test_scale2_high_spo2_penalized(self):
        from sepsis_vitals.scores import news2_style
        # SpO2 97% under Scale 2 = score 3 (too high for hypercapnic patient)
        score = news2_style({"spo2": 97, "on_supplemental_o2": True})
        assert score == 3

    def test_scale2_low_spo2(self):
        from sepsis_vitals.scores import news2_style
        # SpO2 82% under Scale 2 = score 3
        score = news2_style({"spo2": 82, "on_supplemental_o2": True})
        assert score == 3

    def test_scale1_spo2_96_scores_0(self):
        from sepsis_vitals.scores import news2_style
        score = news2_style({"spo2": 96})
        assert score == 0

    def test_scale2_spo2_95_scores_2(self):
        from sepsis_vitals.scores import news2_style
        # Scale 2: 93-96 = score 2
        score = news2_style({"spo2": 95, "on_supplemental_o2": True})
        assert score == 2
