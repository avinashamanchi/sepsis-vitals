"""
sepsis_vitals.patients.service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Patient data management and vitals persistence layer.

Provides CRUD operations for patients, vital-sign recording with automatic
clinical score computation, alert lifecycle management, and site-level
dashboard statistics.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from sepsis_vitals.db import Alert, Patient, Score, VitalReading
from sepsis_vitals.scores import ScoreBundle, compute_scores


# ---------------------------------------------------------------------------
# Patient CRUD
# ---------------------------------------------------------------------------


def create_patient(
    external_id: str,
    site_id: str,
    age_years: int | None,
    sex: str | None,
    db: Session,
) -> Patient:
    """Create a new patient record.

    Raises
    ------
    ValueError
        If a patient with the same *external_id* already exists.
    """
    existing = (
        db.query(Patient)
        .filter(Patient.external_id == external_id)
        .first()
    )
    if existing is not None:
        raise ValueError(
            f"Patient with external_id '{external_id}' already exists"
        )

    patient = Patient(
        external_id=external_id,
        site_id=site_id,
        age_years=age_years,
        sex=sex,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_patient(patient_id: str, db: Session) -> Patient | None:
    """Return a single patient by primary key, or ``None``."""
    return db.query(Patient).filter(Patient.id == patient_id).first()


def get_patient_by_external_id(
    external_id: str, db: Session
) -> Patient | None:
    """Return a patient by their external (site-assigned) identifier."""
    return (
        db.query(Patient)
        .filter(Patient.external_id == external_id)
        .first()
    )


def list_patients(
    site_id: str | None,
    db: Session,
    skip: int = 0,
    limit: int = 50,
) -> list[Patient]:
    """Return a paginated list of patients, optionally filtered by *site_id*."""
    query = db.query(Patient)
    if site_id is not None:
        query = query.filter(Patient.site_id == site_id)
    return query.order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()


def update_patient(
    patient_id: str,
    updates: dict[str, Any],
    db: Session,
) -> Patient:
    """Apply *updates* to the patient identified by *patient_id*.

    Only the fields ``age_years``, ``sex``, and ``site_id`` may be updated.

    Raises
    ------
    ValueError
        If the patient does not exist or an invalid field is supplied.
    """
    patient = get_patient(patient_id, db)
    if patient is None:
        raise ValueError(f"Patient '{patient_id}' not found")

    allowed_fields = {"age_years", "sex", "site_id"}
    invalid_fields = set(updates.keys()) - allowed_fields
    if invalid_fields:
        raise ValueError(
            f"Cannot update field(s): {', '.join(sorted(invalid_fields))}"
        )

    for field_name, value in updates.items():
        setattr(patient, field_name, value)

    patient.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(patient)
    return patient


# ---------------------------------------------------------------------------
# Vital-sign recording with automatic scoring and alerting
# ---------------------------------------------------------------------------


def record_vitals(
    patient_id: str,
    vitals_dict: dict[str, Any],
    recorded_at: datetime | None,
    db: Session,
) -> tuple[VitalReading, Score]:
    """Persist a vital-sign observation and compute clinical scores.

    The function:
    1. Creates a :class:`VitalReading` row.
    2. Runs :func:`compute_scores` to derive qSOFA, SIRS, NEWS2, etc.
    3. Persists a :class:`Score` row linked to the reading.
    4. If the risk level is ``high`` or ``critical``, creates an
       :class:`Alert` row with status ``active``.

    Parameters
    ----------
    patient_id:
        UUID of the patient.
    vitals_dict:
        Dictionary whose keys are a subset of ``temperature``,
        ``heart_rate``, ``resp_rate``, ``sbp``, ``spo2``, ``gcs``.
    recorded_at:
        Timestamp of the observation.  Defaults to *now* (UTC).
    db:
        Active SQLAlchemy session.

    Returns
    -------
    tuple[VitalReading, Score]

    Raises
    ------
    ValueError
        If the patient does not exist.
    """
    patient = get_patient(patient_id, db)
    if patient is None:
        raise ValueError(f"Patient '{patient_id}' not found")

    if recorded_at is None:
        recorded_at = datetime.now(timezone.utc)

    # 1. Persist VitalReading
    vital = VitalReading(
        patient_id=patient_id,
        recorded_at=recorded_at,
        temperature=vitals_dict.get("temperature"),
        heart_rate=vitals_dict.get("heart_rate"),
        resp_rate=vitals_dict.get("resp_rate"),
        sbp=vitals_dict.get("sbp"),
        spo2=vitals_dict.get("spo2"),
        gcs=vitals_dict.get("gcs"),
    )
    db.add(vital)
    db.flush()  # populate vital.id before creating Score

    # 2. Compute clinical scores
    bundle: ScoreBundle = compute_scores(vitals_dict)

    # Serialise component_flags for the JSON column
    component_flags_value: Any
    try:
        component_flags_value = json.dumps(bundle.component_flags)
    except (TypeError, ValueError):
        component_flags_value = None

    # 3. Persist Score
    score = Score(
        vital_id=vital.id,
        qsofa=bundle.qsofa,
        sirs_count=bundle.sirs_count,
        shock_index=bundle.shock_index,
        news2_style=bundle.news2_style,
        uva_style=bundle.uva_style,
        risk_level=bundle.risk_level,
        alert_flag=bundle.alert_flag,
        component_flags=component_flags_value,
    )
    db.add(score)
    db.flush()

    # 4. Create Alert when risk is high or critical
    if bundle.risk_level in ("high", "critical"):
        alert = Alert(
            score_id=score.id,
            patient_id=patient_id,
            risk_level=bundle.risk_level,
            status="active",
        )
        db.add(alert)

    db.commit()
    db.refresh(vital)
    db.refresh(score)
    return vital, score


# ---------------------------------------------------------------------------
# Vitals & history queries
# ---------------------------------------------------------------------------


def get_patient_vitals(
    patient_id: str,
    db: Session,
    hours_back: int = 24,
) -> list[VitalReading]:
    """Return vital-sign readings for *patient_id* within the last *hours_back* hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    return (
        db.query(VitalReading)
        .filter(
            VitalReading.patient_id == patient_id,
            VitalReading.recorded_at >= cutoff,
        )
        .order_by(VitalReading.recorded_at.desc())
        .all()
    )


def get_patient_history(
    patient_id: str,
    db: Session,
) -> dict[str, Any]:
    """Build a complete clinical timeline for *patient_id*.

    Returns a dictionary with three keys:

    * ``vitals`` -- list of vital-reading dicts ordered by time.
    * ``scores`` -- list of score dicts ordered by time.
    * ``alerts`` -- list of alert dicts ordered by time.
    """
    vitals = (
        db.query(VitalReading)
        .filter(VitalReading.patient_id == patient_id)
        .order_by(VitalReading.recorded_at.asc())
        .all()
    )

    vital_ids = [v.id for v in vitals]

    scores: list[Score] = []
    if vital_ids:
        scores = (
            db.query(Score)
            .filter(Score.vital_id.in_(vital_ids))
            .order_by(Score.created_at.asc())
            .all()
        )

    alerts = (
        db.query(Alert)
        .filter(Alert.patient_id == patient_id)
        .order_by(Alert.created_at.asc())
        .all()
    )

    def _vital_dict(v: VitalReading) -> dict[str, Any]:
        return {
            "id": v.id,
            "recorded_at": v.recorded_at.isoformat() if v.recorded_at else None,
            "temperature": v.temperature,
            "heart_rate": v.heart_rate,
            "resp_rate": v.resp_rate,
            "sbp": v.sbp,
            "spo2": v.spo2,
            "gcs": v.gcs,
        }

    def _score_dict(s: Score) -> dict[str, Any]:
        return {
            "id": s.id,
            "vital_id": s.vital_id,
            "qsofa": s.qsofa,
            "sirs_count": s.sirs_count,
            "shock_index": s.shock_index,
            "news2_style": s.news2_style,
            "uva_style": s.uva_style,
            "risk_level": s.risk_level,
            "alert_flag": s.alert_flag,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }

    def _alert_dict(a: Alert) -> dict[str, Any]:
        return {
            "id": a.id,
            "score_id": a.score_id,
            "risk_level": a.risk_level,
            "status": a.status,
            "action_by": a.action_by,
            "action_reason": a.action_reason,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "actioned_at": a.actioned_at.isoformat() if a.actioned_at else None,
        }

    return {
        "vitals": [_vital_dict(v) for v in vitals],
        "scores": [_score_dict(s) for s in scores],
        "alerts": [_alert_dict(a) for a in alerts],
    }


# ---------------------------------------------------------------------------
# Alert management
# ---------------------------------------------------------------------------


def get_active_alerts(
    site_id: str | None,
    db: Session,
) -> list[Alert]:
    """Return all unacknowledged (``active``) alerts.

    When *site_id* is provided the results are restricted to patients
    belonging to that site.
    """
    query = db.query(Alert).filter(Alert.status == "active")
    if site_id is not None:
        query = query.join(Patient).filter(Patient.site_id == site_id)
    return query.order_by(Alert.created_at.desc()).all()


def acknowledge_alert(
    alert_id: str,
    user_id: str,
    reason: str,
    db: Session,
) -> Alert:
    """Mark an alert as acknowledged.

    Raises
    ------
    ValueError
        If the alert does not exist or is not in ``active`` status.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        raise ValueError(f"Alert '{alert_id}' not found")
    if alert.status != "active":
        raise ValueError(
            f"Alert '{alert_id}' is already '{alert.status}', cannot acknowledge"
        )

    now = datetime.now(timezone.utc)
    alert.status = "acknowledged"
    alert.action_by = user_id
    alert.action_reason = reason
    alert.actioned_at = now

    # Compute time-to-action if creation timestamp is available
    if alert.created_at is not None:
        created = alert.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        alert.time_to_action_s = (now - created).total_seconds()

    db.commit()
    db.refresh(alert)
    return alert


# ---------------------------------------------------------------------------
# Dashboard / aggregate statistics
# ---------------------------------------------------------------------------


def get_site_dashboard_stats(
    site_id: str,
    db: Session,
) -> dict[str, Any]:
    """Return aggregate statistics for a site dashboard.

    Keys returned:

    * ``patient_count`` -- total patients at this site.
    * ``active_alerts`` -- number of unacknowledged alerts.
    * ``recent_predictions`` -- number of scores computed in the last 24 h.
    """
    patient_count: int = (
        db.query(func.count(Patient.id))
        .filter(Patient.site_id == site_id)
        .scalar()
        or 0
    )

    patient_ids_subq = (
        db.query(Patient.id).filter(Patient.site_id == site_id).subquery()
    )

    active_alerts: int = (
        db.query(func.count(Alert.id))
        .filter(
            Alert.patient_id.in_(db.query(patient_ids_subq.c.id)),
            Alert.status == "active",
        )
        .scalar()
        or 0
    )

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    recent_vital_ids_subq = (
        db.query(VitalReading.id)
        .filter(
            VitalReading.patient_id.in_(db.query(patient_ids_subq.c.id)),
            VitalReading.recorded_at >= cutoff,
        )
        .subquery()
    )

    recent_predictions: int = (
        db.query(func.count(Score.id))
        .filter(Score.vital_id.in_(db.query(recent_vital_ids_subq.c.id)))
        .scalar()
        or 0
    )

    return {
        "patient_count": patient_count,
        "active_alerts": active_alerts,
        "recent_predictions": recent_predictions,
    }
