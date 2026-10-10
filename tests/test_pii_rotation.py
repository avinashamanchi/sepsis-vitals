"""
tests/test_pii_rotation.py — PII keyring, rotation window and the rotation tool.

Disposable data only: throwaway keys generated per test, a temporary SQLite
database per test, synthetic emails and MRNs.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

pytest.importorskip("cryptography")

PASSWORD = "Correct-Horse-Battery-9!"


def new_key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


@pytest.fixture()
def keys():
    return {"legacy": new_key(), "k1": new_key(), "k2": new_key()}


@pytest.fixture()
def use_keys(monkeypatch):
    """Configure the PII keyring through the environment and reload it."""
    from sepsis_vitals.security import FieldEncryptor

    def configure(current_id: str, current: str, previous: dict | None = None) -> None:
        monkeypatch.setenv("SEPSIS_PII_KEY", current)
        if current_id == "legacy":
            monkeypatch.delenv("SEPSIS_PII_KEY_ID", raising=False)
        else:
            monkeypatch.setenv("SEPSIS_PII_KEY_ID", current_id)
        monkeypatch.setenv("SEPSIS_PII_PREVIOUS_KEYS", ",".join(f"{k}:{v}" for k, v in (previous or {}).items()))
        for attr in ("_instance", "_key", "_keyring"):
            monkeypatch.setattr(FieldEncryptor, attr, None, raising=False)

    return configure


@pytest.fixture()
def temp_db(tmp_path):
    from sepsis_vitals.db import Base

    engine = sa.create_engine(f"sqlite:///{tmp_path}/pii.db")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _add_user(engine, email: str, totp: str | None = "JBSWY3DPEHPK3PXP"):
    from sepsis_vitals.db import User
    from sepsis_vitals.security import compute_blind_index

    db = sessionmaker(bind=engine)()
    try:
        user = User(email=email, email_hash=compute_blind_index(email), password_hash="x",
                    role="nurse", site_id="S1", totp_secret=totp)
        db.add(user)
        db.commit()
        return user.id
    finally:
        db.close()


def _add_patient(engine, mrn: str):
    from sepsis_vitals.db import Patient
    from sepsis_vitals.security import compute_blind_index

    db = sessionmaker(bind=engine)()
    try:
        patient = Patient(external_id=mrn, external_id_hash=compute_blind_index(mrn), site_id="S1",
                          age_years=50, sex="F")
        db.add(patient)
        db.commit()
        return patient.id
    finally:
        db.close()


def _raw(engine, sql: str, **params):
    with engine.begin() as conn:
        result = conn.execute(sa.text(sql), params)
        return result.fetchall() if result.returns_rows else None


# -- keyring configuration -------------------------------------------------------------

def test_keyring_reads_current_and_previous_keys(keys):
    from sepsis_vitals.pii_keys import keyring_from_env

    ring = keyring_from_env({"SEPSIS_PII_KEY": keys["k1"], "SEPSIS_PII_KEY_ID": "k1",
                             "SEPSIS_PII_PREVIOUS_KEYS": f"legacy:{keys['legacy']}"})
    assert ring.current.kid == "k1" and [k.kid for k in ring.previous] == ["legacy"]
    assert keyring_from_env({}) is None


@pytest.mark.parametrize("env_patch", [
    {"SEPSIS_PII_KEY": "not base64!!"},
    {"SEPSIS_PII_KEY": base64.b64encode(b"short").decode()},
    {"SEPSIS_PII_KEY_ID": "K1!"},
    {"SEPSIS_PII_PREVIOUS_KEYS": "k1:{k1}"},                    # duplicate id
    {"SEPSIS_PII_PREVIOUS_KEYS": "old:{k1}"},                   # same key as current
    {"SEPSIS_PII_PREVIOUS_KEYS": "{legacy}"},                   # no id
])
def test_invalid_configuration_is_rejected_without_revealing_keys(keys, env_patch):
    from sepsis_vitals.pii_keys import KeyringError, keyring_from_env

    env = {"SEPSIS_PII_KEY": keys["k1"], "SEPSIS_PII_KEY_ID": "k1"}
    env.update({k: v.format(**keys) for k, v in env_patch.items()})
    with pytest.raises(KeyringError) as err:
        keyring_from_env(env)
    assert not any(secret in str(err.value) for secret in keys.values())


def test_previous_keys_without_a_current_key_are_rejected(keys):
    from sepsis_vitals.pii_keys import KeyringError, keyring_from_env

    with pytest.raises(KeyringError):
        keyring_from_env({"SEPSIS_PII_PREVIOUS_KEYS": f"legacy:{keys['legacy']}"})


def test_production_refuses_an_unusable_keyring(monkeypatch, keys, use_keys):
    from sepsis_vitals.security import FieldEncryptor

    use_keys("k1", keys["k1"], {"k1": keys["k2"]})  # duplicate id
    monkeypatch.setenv("SEPSIS_ENV", "production")
    with pytest.raises(RuntimeError):
        FieldEncryptor._load_key()


# -- formats ---------------------------------------------------------------------------

def test_legacy_key_keeps_the_original_ciphertext_and_index(keys):
    """Unchanged behaviour until a key ID is set: existing data and indexes stay valid."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    from sepsis_vitals.pii_keys import Keyring, PIIKey

    secret = base64.b64decode(keys["legacy"])
    ring = Keyring(PIIKey("legacy", secret))
    token = ring.encrypt("nurse.a@example.org")
    assert token.startswith("enc:") and not token.startswith("enc:v2:")
    raw = base64.b64decode(token[4:])
    assert AESGCM(secret).decrypt(raw[:12], raw[12:], None) == b"nurse.a@example.org"
    expected = hmac.new(secret, b"nurse.a@example.org", hashlib.sha256).hexdigest()
    assert ring.blind_index("Nurse.A@Example.org") == expected


def test_versioned_ciphertext_is_bound_to_its_key_id(keys):
    from sepsis_vitals.pii_keys import Keyring, PIIKey

    k1 = PIIKey("k1", base64.b64decode(keys["k1"]))
    ring = Keyring(k1, (PIIKey("k2", base64.b64decode(keys["k2"])),))
    token = ring.encrypt("MRN-0001")
    assert token.startswith("enc:v2:k1:") and ring.decrypt(token) == "MRN-0001"
    relabelled = token.replace("enc:v2:k1:", "enc:v2:k2:")
    with pytest.raises(ValueError):
        ring.decrypt(relabelled)
    assert k1.encryption_key != k1.index_key != k1.secret  # separated subkeys


def test_previous_keys_still_decrypt_and_failures_name_no_values(keys):
    from sepsis_vitals.pii_keys import Keyring, PIIKey

    legacy = PIIKey("legacy", base64.b64decode(keys["legacy"]))
    k1 = PIIKey("k1", base64.b64decode(keys["k1"]))
    old_token = Keyring(legacy).encrypt("MRN-OLD")
    k1_token = Keyring(k1).encrypt("MRN-K1")
    promoted = Keyring(PIIKey("k2", base64.b64decode(keys["k2"])), (k1, legacy))
    assert promoted.decrypt(old_token) == "MRN-OLD" and promoted.decrypt(k1_token) == "MRN-K1"

    retired = Keyring(PIIKey("k2", base64.b64decode(keys["k2"])))
    for token, expected in ((k1_token, "not configured"), (old_token, "no configured key"),
                            ("enc:@@@@", "base64"), ("enc:" + base64.b64encode(b"x" * 8).decode(), "truncated")):
        with pytest.raises(ValueError) as err:
            retired.decrypt(token)
        assert expected in str(err.value)
        assert "MRN" not in str(err.value)


# -- the application during a rotation ---------------------------------------------------

def test_logins_mrn_lookups_and_recovery_codes_survive_key_promotion(temp_db, keys, use_keys, monkeypatch):
    from sepsis_vitals.auth.service import DuplicateEmailError, login_user, register_user, verify_second_factor
    from sepsis_vitals.db import User
    from sepsis_vitals.patients.service import create_patient, get_patient_by_external_id
    from sepsis_vitals.security import compute_blind_index

    monkeypatch.setenv("SEPSIS_JWT_SECRET", "pii-rotation-test-secret-0123456789abcdef")
    Session = sessionmaker(bind=temp_db)
    use_keys("legacy", keys["legacy"])
    db = Session()
    register_user("rotation.nurse@example.org", PASSWORD, "nurse", "S1", db)
    create_patient("MRN-ROT-1", "S1", 70, "M", db)
    user = db.query(User).one()
    user.mfa_recovery_hashes = json.dumps([compute_blind_index("abcd2345efgh")])
    db.commit()
    db.close()

    use_keys("k1", keys["k1"], {"legacy": keys["legacy"]})  # promote k1; legacy still readable
    db = Session()
    assert login_user("rotation.nurse@example.org", PASSWORD, db)["access_token"]
    assert get_patient_by_external_id("MRN-ROT-1", db, site_id="S1") is not None
    with pytest.raises(ValueError):
        create_patient("MRN-ROT-1", "S1", 70, "M", db)  # no duplicate under the new index
    with pytest.raises(DuplicateEmailError):
        register_user("rotation.nurse@example.org", PASSWORD, "nurse", "S1", db)
    assert verify_second_factor(db.query(User).one(), "abcd-2345-efgh", db) is True
    db.close()


def test_prediction_history_is_found_after_promotion_and_rekeyed_on_write(tmp_path, keys, use_keys):
    from sepsis_vitals.ml.state_store import PatientStateStore
    from sepsis_vitals.security import blind_index_candidates

    use_keys("legacy", keys["legacy"])
    store = PatientStateStore(str(tmp_path / "patient_state.v1.db"))
    store.add_prediction("p-1", "2026-10-01T10:00:00", 0.2, "low")
    store.add_prediction("p-1", "2026-10-01T11:00:00", 0.3, "moderate")
    assert store.get_baseline_risk("p-1") == 0.2

    use_keys("k1", keys["k1"], {"legacy": keys["legacy"]})
    assert len(store.get_predictions("p-1")) == 2 and store.get_baseline_risk("p-1") == 0.2
    store.add_prediction("p-1", "2026-10-01T12:00:00", 0.4, "moderate")
    current, old = blind_index_candidates("p-1")
    refs = {r[0] for r in store._conn.execute("SELECT patient_id FROM predictions")}
    assert refs == {current}, "history was not moved to the current reference"
    assert store.get_baseline_risk("p-1") == 0.2 and len(store.get_predictions("p-1")) == 3


# -- the rotation tool ---------------------------------------------------------------------

@pytest.fixture()
def mixed_db(temp_db, keys, use_keys):
    """Rows written under different key configurations over time, plus damage."""
    use_keys("legacy", keys["legacy"])
    ids = {"u_legacy": _add_user(temp_db, "legacy.user@example.org"),
           "p_legacy": _add_patient(temp_db, "MRN-LEGACY")}
    use_keys("k1", keys["k1"], {"legacy": keys["legacy"]})
    ids["u_k1"] = _add_user(temp_db, "k1.user@example.org")
    ids["p_k1"] = _add_patient(temp_db, "MRN-K1")
    use_keys("k2", keys["k2"], {"k1": keys["k1"], "legacy": keys["legacy"]})
    ids["u_k2"] = _add_user(temp_db, "k2.user@example.org", totp=None)
    ids["p_k2"] = _add_patient(temp_db, "MRN-K2")
    ids["p_k2b"] = _add_patient(temp_db, "MRN-K2B")
    return ids


def _ring():
    from sepsis_vitals.pii_keys import keyring_from_env

    return keyring_from_env()


def _store(engine):
    from sepsis_vitals.pii_rotation import _SQLAlchemyStore

    return _SQLAlchemyStore(engine)


def test_inventory_counts_formats_and_indexes_without_values(temp_db, mixed_db, keys):
    from sepsis_vitals.pii_rotation import DB_COLUMNS, inventory

    report = inventory(_ring(), _store(temp_db), DB_COLUMNS)
    assert report["current_key"] == "k2"
    assert report["columns"]["users.email"]["values"] == {"key:k1": 1, "key:k2": 1, "key:legacy": 1}
    assert report["columns"]["users.email"]["blind_index"] == {"current": 1, "previous:k1": 1, "previous:legacy": 1}
    assert report["columns"]["users.totp_secret"]["values"] == {"empty": 1, "key:k1": 1, "key:legacy": 1}
    text = json.dumps(report)
    assert "example.org" not in text and "MRN" not in text
    assert not any(k in text for k in keys.values())


def test_dry_run_writes_nothing(temp_db, mixed_db):
    from sepsis_vitals.pii_rotation import DB_COLUMNS, rotate

    before = _raw(temp_db, "SELECT email, email_hash FROM users ORDER BY id")
    results = rotate(_ring(), _store(temp_db), DB_COLUMNS, apply=False, progress=lambda m: None)
    assert results["users.email"].rekeyed == 2
    assert _raw(temp_db, "SELECT email, email_hash FROM users ORDER BY id") == before


def test_bounded_batches_resume_and_finish_with_readable_current_data(temp_db, mixed_db, keys, use_keys):
    from sepsis_vitals.pii_rotation import DB_COLUMNS, inventory, rotate, verify

    ring = _ring()
    first = rotate(ring, _store(temp_db), DB_COLUMNS, apply=True, batch_size=1, max_batches=1, progress=lambda m: None)
    assert first["users.email"].rekeyed == 1
    assert all(r.rekeyed == 0 for name, r in first.items() if name != "users.email")
    assert verify(ring, inventory(ring, _store(temp_db), DB_COLUMNS)), "should still need work"

    rotate(ring, _store(temp_db), DB_COLUMNS, apply=True, batch_size=1, progress=lambda m: None)
    assert verify(ring, inventory(ring, _store(temp_db), DB_COLUMNS)) == []

    # Retire the previous keys: everything still decrypts and lookups match.
    use_keys("k2", keys["k2"])
    from sepsis_vitals.db import Patient, User
    from sepsis_vitals.security import blind_index_candidates

    db = sessionmaker(bind=temp_db)()
    try:
        emails = sorted(u.email for u in db.query(User))
        assert emails == ["k1.user@example.org", "k2.user@example.org", "legacy.user@example.org"]
        assert sorted(p.external_id for p in db.query(Patient)) == ["MRN-K1", "MRN-K2", "MRN-K2B", "MRN-LEGACY"]
        assert db.query(Patient).filter(Patient.external_id_hash.in_(blind_index_candidates("MRN-LEGACY"))).one()
        assert db.query(User).filter(User.email_hash.in_(blind_index_candidates("legacy.user@example.org"))).one()
    finally:
        db.close()


def test_a_failed_batch_rolls_back_and_a_rerun_completes(temp_db, mixed_db, monkeypatch):
    """users.email (2 rows to re-key) commits as one batch; the users.totp_secret
    batch crashes after re-keying its first row, which must be rolled back."""
    from sepsis_vitals.pii_rotation import DB_COLUMNS, _SQLAlchemyStore, inventory, rotate, verify

    ring = _ring()
    calls = {"n": 0}
    real_update = _SQLAlchemyStore.update

    def crash_on_fourth(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("simulated crash")
        return real_update(self, *args, **kwargs)

    monkeypatch.setattr(_SQLAlchemyStore, "update", crash_on_fourth)
    with pytest.raises(RuntimeError):
        rotate(ring, _store(temp_db), DB_COLUMNS, apply=True, batch_size=2, progress=lambda m: None)
    columns = inventory(ring, _store(temp_db), DB_COLUMNS)["columns"]
    assert columns["users.email"]["values"] == {"key:k2": 3}
    assert columns["users.totp_secret"]["values"] == {"empty": 1, "key:k1": 1, "key:legacy": 1}

    monkeypatch.setattr(_SQLAlchemyStore, "update", real_update)
    rotate(ring, _store(temp_db), DB_COLUMNS, apply=True, batch_size=2, progress=lambda m: None)
    assert verify(ring, inventory(ring, _store(temp_db), DB_COLUMNS)) == []


def test_unreadable_rows_stop_the_run_unless_skipped(temp_db, mixed_db, keys):
    from sepsis_vitals.pii_rotation import DB_COLUMNS, RotationError, inventory, rotate, verify

    foreign = "enc:v2:gone:" + base64.b64encode(os.urandom(40)).decode()
    _raw(temp_db, "UPDATE patients SET external_id = :v WHERE id = :pk", v=foreign, pk=mixed_db["p_k2b"])
    _raw(temp_db, "UPDATE users SET totp_secret = 'enc:@@@' WHERE id = :pk", pk=mixed_db["u_k1"])
    ring = _ring()
    report = inventory(ring, _store(temp_db), DB_COLUMNS)
    assert report["columns"]["patients.external_id"]["values"]["unreadable:key gone not configured"] == 1
    assert report["columns"]["users.totp_secret"]["values"]["unreadable:malformed"] == 1

    with pytest.raises(RotationError) as err:
        rotate(ring, _store(temp_db), DB_COLUMNS, apply=True, progress=lambda m: None)
    assert "ref:" in str(err.value) and str(mixed_db["u_k1"]) not in str(err.value)

    messages = []
    results = rotate(ring, _store(temp_db), DB_COLUMNS, apply=True, skip_unreadable=True, progress=messages.append)
    assert results["patients.external_id"].unreadable == 1 and results["users.totp_secret"].unreadable == 1
    problems = verify(ring, inventory(ring, _store(temp_db), DB_COLUMNS))
    assert any("unreadable" in p for p in problems) and len(problems) == 2
    assert not any("MRN" in m or "example.org" in m for m in messages)


def test_rotation_requires_a_versioned_current_key(temp_db, keys, use_keys):
    from sepsis_vitals.pii_rotation import DB_COLUMNS, RotationError, rotate

    use_keys("legacy", keys["legacy"])
    with pytest.raises(RotationError):
        rotate(_ring(), _store(temp_db), DB_COLUMNS, apply=True)


def test_a_row_changed_since_it_was_read_is_not_overwritten(temp_db, keys, use_keys):
    from sepsis_vitals.pii_rotation import DB_COLUMNS, _SQLAlchemyStore

    use_keys("legacy", keys["legacy"])
    pk = _add_patient(temp_db, "MRN-CAS")
    store = _SQLAlchemyStore(temp_db)
    with store.transaction():
        assert store.update(DB_COLUMNS[2], pk, "enc:stale-value", "enc:new", "idx") is False


def test_encrypted_column_list_covers_every_encrypted_orm_column():
    import sepsis_vitals.billing.models  # noqa: F401
    import sepsis_vitals.bundles.models  # noqa: F401
    from sepsis_vitals.db import Base, EncryptedString
    from sepsis_vitals.pii_rotation import DB_COLUMNS

    orm = {f"{t.name}.{c.name}" for t in Base.metadata.sorted_tables for c in t.columns
           if isinstance(c.type, EncryptedString)}
    assert orm == {c.name for c in DB_COLUMNS}


def test_escalation_store_values_are_rekeyed(tmp_path, keys, use_keys):
    from sepsis_vitals.pii_rotation import ESCALATION_COLUMNS, _SQLiteStore, inventory, rotate, verify

    use_keys("legacy", keys["legacy"])
    from sepsis_vitals.alerts.escalation import AlertEscalationManager

    path = tmp_path / "alert_escalation.v1.db"
    manager = AlertEscalationManager(db_path=str(path))
    manager.register_alert("alert-1", patient_id="p-esc-1", risk_level="high")
    manager.acknowledge_alert("alert-1", user_id="nurse-1")

    use_keys("k1", keys["k1"], {"legacy": keys["legacy"]})
    ring = _ring()
    rotate(ring, _SQLiteStore(path), ESCALATION_COLUMNS, apply=True, progress=lambda m: None)
    assert verify(ring, inventory(ring, _SQLiteStore(path), ESCALATION_COLUMNS)) == []
    use_keys("k1", keys["k1"])  # previous key retired
    reread = AlertEscalationManager(db_path=str(path))
    assert reread.get_alert_status("alert-1")["patient_id"] == "p-esc-1"
    assert any(e["user_id"] == "nurse-1" for e in reread.get_alert_lifecycle("alert-1"))


def test_cli_inventory_and_verify_print_no_values_or_keys(temp_db, mixed_db, keys, monkeypatch, capsys):
    from sepsis_vitals import db as db_module
    from sepsis_vitals.pii_rotation import main

    monkeypatch.setattr(db_module, "engine", temp_db)
    assert main(["inventory", "--json"]) == 0
    assert main(["verify"]) == 1
    assert main(["rotate"]) == 0  # dry run
    out = capsys.readouterr().out
    assert "Dry run" in out and "previous keys are still needed" in out
    assert "example.org" not in out and "MRN" not in out
    assert not any(k in out for k in keys.values())
