"""Tests for the Hour-1 bundle tracker (sepsis_vitals.bundles)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sepsis_vitals.db import Base, Patient
# Importing models registers the bundle tables on the shared Base.metadata.
from sepsis_vitals.bundles import service
from sepsis_vitals.bundles.models import BundleTask, SepsisBundle
from sepsis_vitals.bundles import protocol


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def patient(db):
    p = Patient(external_id="ext-1", external_id_hash="h1", site_id="site-a")
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_protocol_has_five_core_critical_tasks():
    critical = [t for t in protocol.HOUR1_BUNDLE.tasks if t.critical]
    assert len(critical) == 5
    keys = {t.key for t in critical}
    assert {"lactate_initial", "blood_cultures", "antibiotics",
            "fluids", "vasopressors"} <= keys


def test_conditional_tasks_gate_on_vitals():
    # Normotensive, normal lactate -> no vasopressors, no fluids, no repeat.
    stable = protocol.applicable_tasks(lactate=1.0, map_pressure=80, sbp=120)
    stable_keys = {t.key for t in stable}
    assert "vasopressors" not in stable_keys
    assert "fluids" not in stable_keys
    assert "lactate_repeat" not in stable_keys

    # Hypotensive, high lactate -> everything applies.
    shocked = protocol.applicable_tasks(lactate=5.0, map_pressure=55, sbp=80)
    shocked_keys = {t.key for t in shocked}
    assert {"fluids", "vasopressors", "lactate_repeat"} <= shocked_keys


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_bundle_seeds_applicable_tasks(db, patient):
    b = service.start_bundle(
        patient.id, db, risk_level="critical",
        vitals={"lactate": 5.0, "map": 55, "sbp": 80},
    )
    assert b.status == "open"
    keys = {t.task_key for t in b.tasks}
    assert "antibiotics" in keys and "vasopressors" in keys


def test_start_bundle_is_idempotent(db, patient):
    b1 = service.start_bundle(patient.id, db)
    b2 = service.start_bundle(patient.id, db)
    assert b1.id == b2.id
    open_count = (
        db.query(SepsisBundle)
        .filter(SepsisBundle.patient_id == patient.id, SepsisBundle.status == "open")
        .count()
    )
    assert open_count == 1


def test_complete_task_sets_timestamp_and_compliance(db, patient):
    b = service.start_bundle(
        patient.id, db, vitals={"lactate": 1.0, "map": 80, "sbp": 120}
    )
    b = service.complete_task(b.id, "antibiotics", db, completed_by="nurse-1")
    abx = next(t for t in b.tasks if t.task_key == "antibiotics")
    assert abx.completed and abx.completed_at is not None
    assert b.time_to_antibiotics_s is not None
    assert 0 <= b.compliance_pct <= 100


def test_completing_all_critical_tasks_completes_bundle(db, patient):
    b = service.start_bundle(
        patient.id, db, vitals={"lactate": 1.0, "map": 80, "sbp": 120}
    )
    critical_keys = [t.task_key for t in b.tasks if t.critical]
    for key in critical_keys:
        b = service.complete_task(b.id, key, db)
    assert b.status == "completed"
    assert b.compliance_pct == 100.0
    assert b.closed_at is not None


def test_undo_task(db, patient):
    b = service.start_bundle(patient.id, db, vitals={"lactate": 1.0})
    b = service.complete_task(b.id, "blood_cultures", db)
    b = service.complete_task(b.id, "blood_cultures", db, completed=False)
    bc = next(t for t in b.tasks if t.task_key == "blood_cultures")
    assert not bc.completed and bc.completed_at is None


def test_cannot_edit_closed_bundle(db, patient):
    b = service.start_bundle(patient.id, db)
    service.cancel_bundle(b.id, db, reason="sepsis ruled out")
    with pytest.raises(ValueError):
        service.complete_task(b.id, "antibiotics", db)


def test_expire_stale_bundle(db, patient):
    b = service.start_bundle(
        patient.id, db, vitals={"lactate": 1.0, "map": 80, "sbp": 120}
    )
    # Backdate the start beyond the 60-minute window.
    b.started_at = datetime.now(timezone.utc) - timedelta(minutes=75)
    db.commit()
    n = service.expire_stale_bundles(db)
    assert n == 1
    db.refresh(b)
    assert b.status == "expired"


def test_serialisation_countdown(db, patient):
    b = service.start_bundle(
        patient.id, db, vitals={"lactate": 1.0, "map": 80, "sbp": 120}
    )
    d = service.bundle_to_dict(b)
    assert d["status"] == "open"
    assert d["seconds_remaining"] is not None
    assert isinstance(d["tasks"], list) and d["tasks"]
