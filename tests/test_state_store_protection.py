"""
tests/test_state_store_protection.py — identifiers in the SQLite side stores.

The patient-state store keeps only keyed references; the escalation store
encrypts patient IDs, user IDs and audit text. Legacy plaintext stores are
never read or rewritten in place. All tests use temporary files.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECRET_MRN = "MRN-SECRET-998877"
SECRET_REASON = "patient seen by Dr Example at bed 4"


@pytest.fixture()
def pii_key(monkeypatch):
    from sepsis_vitals.security import FieldEncryptor

    monkeypatch.setenv("SEPSIS_PII_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setattr(FieldEncryptor, "_instance", None, raising=False)
    monkeypatch.setattr(FieldEncryptor, "_key", None, raising=False)
    assert FieldEncryptor.get().encrypt("x").startswith("enc:")


def _raw_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    wal = path.with_name(path.name + "-wal")
    return data + (wal.read_bytes() if wal.exists() else b"")


def test_patient_state_store_holds_no_identifier(tmp_path, pii_key):
    from sepsis_vitals.ml.state_store import PatientStateStore

    path = tmp_path / "state" / "patient_state.v1.db"
    store = PatientStateStore(str(path))
    store.add_prediction(SECRET_MRN, "2026-10-08T10:00:00", 0.42, "moderate")
    store.add_prediction(SECRET_MRN, "2026-10-08T11:00:00", 0.61, "high")
    trend = store.get_trend(SECRET_MRN)
    store.close()

    assert trend["patient_id"] == SECRET_MRN and trend["n_observations"] == 2
    assert SECRET_MRN.encode() not in _raw_bytes(path)
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_escalation_store_encrypts_and_round_trips(tmp_path, pii_key):
    from sepsis_vitals.alerts.escalation import AlertEscalationManager

    path = tmp_path / "state" / "alert_escalation.v1.db"
    first = AlertEscalationManager(db_path=str(path))
    first.register_alert("a-1", SECRET_MRN, "high")
    first.snooze_alert("a-1", user_id="user-42", snooze_minutes=5)
    first._add_audit(first._alerts["a-1"], "note", user_id="user-42", detail=SECRET_REASON)
    first._conn.close()

    assert SECRET_MRN.encode() not in _raw_bytes(path)
    assert SECRET_REASON.encode() not in _raw_bytes(path)

    reloaded = AlertEscalationManager(db_path=str(path))
    assert reloaded.get_alert_status("a-1")["patient_id"] == SECRET_MRN
    trail = reloaded.get_alert_lifecycle("a-1")
    assert any(e["detail"] == SECRET_REASON for e in trail)


def test_default_paths_use_the_state_dir(tmp_path, monkeypatch):
    from sepsis_vitals.alerts.escalation import default_escalation_path
    from sepsis_vitals.ml.state_store import default_store_path

    monkeypatch.setenv("SEPSIS_STATE_DIR", str(tmp_path))
    assert default_store_path() == tmp_path / "patient_state.v1.db"
    assert default_escalation_path() == tmp_path / "alert_escalation.v1.db"


def _legacy_stores(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(directory / "patient_state.db")
    conn.executescript("""
        CREATE TABLE patients (patient_id TEXT PRIMARY KEY, baseline_risk REAL, created_at REAL);
        CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL,
            timestamp TEXT NOT NULL, risk_probability REAL NOT NULL, risk_level TEXT NOT NULL,
            created_at REAL NOT NULL);
    """)
    conn.execute("INSERT INTO patients VALUES (?, 0.2, 1.0)", (SECRET_MRN,))
    conn.execute(
        "INSERT INTO predictions (patient_id, timestamp, risk_probability, risk_level, created_at) "
        "VALUES (?, '2026-10-08T09:00:00', 0.2, 'low', 1.0)", (SECRET_MRN,),
    )
    conn.commit()
    conn.close()
    conn = sqlite3.connect(directory / "alert_escalation.db")
    conn.executescript("""
        CREATE TABLE tracked_alerts (alert_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL,
            risk_level TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            current_tier INTEGER NOT NULL DEFAULT 0, snoozed_until TEXT);
        CREATE TABLE alert_audit_trail (id TEXT PRIMARY KEY, alert_id TEXT NOT NULL, action TEXT NOT NULL,
            user_id TEXT, detail TEXT, timestamp TEXT NOT NULL);
    """)
    conn.execute("INSERT INTO tracked_alerts VALUES ('a-9', ?, 'high', '2026-10-08T09:00:00+00:00', "
                 "'pending', 0, NULL)", (SECRET_MRN,))
    conn.execute("INSERT INTO alert_audit_trail VALUES ('x', 'a-9', 'registered', 'user-1', ?, "
                 "'2026-10-08T09:00:00+00:00')", (SECRET_REASON,))
    conn.commit()
    conn.close()


def test_legacy_store_is_left_untouched_and_not_read(tmp_path, pii_key, caplog):
    from sepsis_vitals.ml.state_store import PatientStateStore

    _legacy_stores(tmp_path)
    before = (tmp_path / "patient_state.db").read_bytes()
    store = PatientStateStore(str(tmp_path / "patient_state.v1.db"))
    assert store.get_predictions(SECRET_MRN) == []
    store.close()
    assert (tmp_path / "patient_state.db").read_bytes() == before
    assert "Legacy plaintext state store" in caplog.text


def test_migration_writes_protected_copies_and_never_touches_sources(tmp_path, pii_key):
    spec = importlib.util.spec_from_file_location(
        "migrate_state_stores", ROOT / "scripts" / "migrate_state_stores.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    source, output = tmp_path / "legacy", tmp_path / "state"
    _legacy_stores(source)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}

    assert mod.migrate(source, output, dry_run=True) == {
        "patient_state": {"patients": 1, "predictions": 1},
        "alert_escalation": {"alerts": 1, "audit_entries": 1},
        "sources_unchanged": True,
    }
    assert not output.exists()

    mod.migrate(source, output)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals
    for name in ("patient_state.v1.db", "alert_escalation.v1.db"):
        assert SECRET_MRN.encode() not in _raw_bytes(output / name)
    assert SECRET_REASON.encode() not in _raw_bytes(output / "alert_escalation.v1.db")

    from sepsis_vitals.alerts.escalation import AlertEscalationManager
    from sepsis_vitals.ml.state_store import PatientStateStore

    assert len(PatientStateStore(str(output / "patient_state.v1.db")).get_predictions(SECRET_MRN)) == 1
    manager = AlertEscalationManager(db_path=str(output / "alert_escalation.v1.db"))
    assert manager.get_alert_status("a-9")["patient_id"] == SECRET_MRN

    with pytest.raises(SystemExit):
        mod.migrate(source, source)  # in-place rewrites are refused



def _migration_module():
    spec = importlib.util.spec_from_file_location(
        "migrate_state_stores", ROOT / "scripts" / "migrate_state_stores.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_refuses_to_write_unprotected_copies(tmp_path, monkeypatch):
    from sepsis_vitals.security import FieldEncryptor

    mod = _migration_module()
    _legacy_stores(tmp_path / "legacy")
    monkeypatch.delenv("SEPSIS_PII_KEY", raising=False)
    for attr in ("_instance", "_key", "_keyring"):
        monkeypatch.setattr(FieldEncryptor, attr, None, raising=False)
    with pytest.raises(SystemExit, match="not be protected"):
        mod.migrate(tmp_path / "legacy", tmp_path / "state")
    assert not (tmp_path / "state").exists()


def test_interrupted_migration_leaves_no_final_file_and_resumes(tmp_path, pii_key, monkeypatch):
    mod = _migration_module()
    source, output = tmp_path / "legacy", tmp_path / "state"
    _legacy_stores(source)
    real_verify = mod._verify_escalation

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash while writing the escalation copy")

    monkeypatch.setattr(mod, "_verify_escalation", crash)
    with pytest.raises(RuntimeError):
        mod.migrate(source, output)
    assert (output / "patient_state.v1.db").exists()
    assert not (output / "alert_escalation.v1.db").exists()  # only a .partial was left
    assert (output / "alert_escalation.v1.db.partial").exists()

    monkeypatch.setattr(mod, "_verify_escalation", real_verify)
    summary = mod.migrate(source, output)
    assert summary["patient_state"]["status"] == "already migrated and verified"
    assert summary["alert_escalation"]["status"] == "migrated and verified"
    assert not (output / "alert_escalation.v1.db.partial").exists()
    assert summary["sources_unchanged"] is True
    for name in ("patient_state", "alert_escalation"):
        assert summary[name]["mode"] == "0o600"
    assert oct(output.stat().st_mode & 0o777) == "0o700"


def test_a_copy_that_does_not_match_the_source_is_refused(tmp_path, pii_key):
    import sqlite3

    mod = _migration_module()
    source, output = tmp_path / "legacy", tmp_path / "state"
    _legacy_stores(source)
    mod.migrate(source, output)
    conn = sqlite3.connect(output / "patient_state.v1.db")
    conn.execute("UPDATE predictions SET risk_probability = 0.99")
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit, match="does not match"):
        mod.migrate(source, output)
