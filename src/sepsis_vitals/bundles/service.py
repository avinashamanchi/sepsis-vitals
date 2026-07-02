"""
sepsis_vitals.bundles.service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Business logic for Hour-1 bundle tracking.

Responsibilities:

* Open a bundle (idempotent -- one open bundle per patient).
* Seed it with the applicable protocol tasks given current vitals.
* Mark tasks complete, stamping who/when and minutes-from-start.
* Recompute derived KPIs (time-to-antibiotics, compliance %).
* Auto-expire bundles whose 60-minute window has lapsed with critical
  tasks still outstanding -- the hook a background sweeper calls.

All timestamps are timezone-aware UTC.  The layer is deliberately free of
FastAPI/Pydantic types so it can be unit-tested with a bare SQLite session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from sepsis_vitals.bundles import protocol
from sepsis_vitals.bundles.models import BundleTask, SepsisBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Coerce a possibly-naive datetime to UTC-aware."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _minutes_between(start: datetime, end: datetime) -> float:
    return (_aware(end) - _aware(start)).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Bundle lifecycle
# ---------------------------------------------------------------------------


def get_open_bundle(patient_id: str, db: Session) -> Optional[SepsisBundle]:
    """Return the patient's currently-open bundle, if any."""
    return (
        db.query(SepsisBundle)
        .filter(
            SepsisBundle.patient_id == patient_id,
            SepsisBundle.status == "open",
        )
        .order_by(SepsisBundle.started_at.desc())
        .first()
    )


def start_bundle(
    patient_id: str,
    db: Session,
    *,
    started_by: Optional[str] = None,
    alert_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    vitals: Optional[Dict[str, Any]] = None,
) -> SepsisBundle:
    """Open a new Hour-1 bundle for *patient_id*.

    Idempotent: if an open bundle already exists it is returned unchanged
    rather than creating a duplicate.  Tasks are seeded from
    :func:`protocol.applicable_tasks` using any supplied *vitals* so that
    conditional interventions (fluids, vasopressors, repeat lactate) only
    appear when clinically indicated.
    """
    existing = get_open_bundle(patient_id, db)
    if existing is not None:
        return existing

    vitals = vitals or {}
    bundle = SepsisBundle(
        patient_id=patient_id,
        alert_id=alert_id,
        protocol_version=protocol.PROTOCOL_VERSION,
        status="open",
        started_by=started_by,
        risk_level_at_start=risk_level,
        started_at=datetime.now(timezone.utc),
    )
    db.add(bundle)
    db.flush()  # populate bundle.id

    specs = protocol.applicable_tasks(
        lactate=vitals.get("lactate"),
        map_pressure=vitals.get("map"),
        sbp=vitals.get("sbp"),
    )
    for spec in specs:
        db.add(
            BundleTask(
                bundle_id=bundle.id,
                task_key=spec.key,
                order_index=spec.order,
                target_minutes=spec.target_minutes,
                critical=spec.critical,
            )
        )

    db.commit()
    db.refresh(bundle)
    return bundle


def complete_task(
    bundle_id: str,
    task_key: str,
    db: Session,
    *,
    completed_by: Optional[str] = None,
    note: Optional[str] = None,
    completed: bool = True,
) -> SepsisBundle:
    """Toggle a task's completion state and recompute bundle KPIs.

    Passing ``completed=False`` un-completes a task (correcting a mistap),
    clearing its timestamp.  Returns the refreshed bundle.

    Raises
    ------
    ValueError
        If the bundle or task does not exist, or the bundle is not open.
    """
    bundle = db.query(SepsisBundle).filter(SepsisBundle.id == bundle_id).first()
    if bundle is None:
        raise ValueError(f"Bundle '{bundle_id}' not found")
    if bundle.status != "open":
        raise ValueError(
            f"Bundle '{bundle_id}' is '{bundle.status}', cannot edit tasks"
        )

    task = (
        db.query(BundleTask)
        .filter(
            BundleTask.bundle_id == bundle_id,
            BundleTask.task_key == task_key,
        )
        .first()
    )
    if task is None:
        raise ValueError(
            f"Task '{task_key}' is not part of bundle '{bundle_id}'"
        )

    now = datetime.now(timezone.utc)
    if completed:
        task.completed = True
        task.completed_by = completed_by
        task.completed_at = now
        task.minutes_from_start = _minutes_between(bundle.started_at, now)
    else:
        task.completed = False
        task.completed_by = None
        task.completed_at = None
        task.minutes_from_start = None
    if note is not None:
        task.note = note

    db.flush()
    _recompute_kpis(bundle, db)

    # Auto-complete the bundle when every critical applicable task is done.
    if _all_critical_done(bundle, db):
        bundle.status = "completed"
        bundle.closed_at = now

    db.commit()
    db.refresh(bundle)
    return bundle


def cancel_bundle(
    bundle_id: str,
    db: Session,
    *,
    cancelled_by: Optional[str] = None,
    reason: Optional[str] = None,
) -> SepsisBundle:
    """Close an open bundle without completing it (e.g. sepsis ruled out)."""
    bundle = db.query(SepsisBundle).filter(SepsisBundle.id == bundle_id).first()
    if bundle is None:
        raise ValueError(f"Bundle '{bundle_id}' not found")
    if bundle.status != "open":
        raise ValueError(f"Bundle '{bundle_id}' is already '{bundle.status}'")

    bundle.status = "cancelled"
    bundle.closed_at = datetime.now(timezone.utc)
    _recompute_kpis(bundle, db)
    db.commit()
    db.refresh(bundle)
    return bundle


def expire_stale_bundles(db: Session, *, grace_minutes: int = 0) -> int:
    """Mark open bundles as ``expired`` once their window has lapsed.

    A bundle expires when the maximum ``target_minutes`` across its still
    outstanding *critical* tasks has been exceeded (plus an optional grace
    period).  Returns the number of bundles expired.  Intended to be called
    periodically by a background sweeper (see
    :func:`sepsis_vitals.bundles.service.sweep`).
    """
    now = datetime.now(timezone.utc)
    open_bundles = (
        db.query(SepsisBundle).filter(SepsisBundle.status == "open").all()
    )
    expired = 0
    for bundle in open_bundles:
        outstanding = [
            t for t in bundle.tasks if t.critical and not t.completed
        ]
        if not outstanding:
            continue
        window = max(t.target_minutes for t in outstanding) + grace_minutes
        if _minutes_between(bundle.started_at, now) > window:
            bundle.status = "expired"
            bundle.closed_at = now
            _recompute_kpis(bundle, db)
            expired += 1
    if expired:
        db.commit()
    return expired


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def _all_critical_done(bundle: SepsisBundle, db: Session) -> bool:
    critical = [t for t in bundle.tasks if t.critical]
    return bool(critical) and all(t.completed for t in critical)


def _recompute_kpis(bundle: SepsisBundle, db: Session) -> None:
    """Refresh time-to-antibiotics and compliance on the bundle row."""
    tasks = list(bundle.tasks)

    abx = next((t for t in tasks if t.task_key == "antibiotics"), None)
    if abx is not None and abx.completed and abx.completed_at is not None:
        bundle.time_to_antibiotics_s = (
            _aware(abx.completed_at) - _aware(bundle.started_at)
        ).total_seconds()
    else:
        bundle.time_to_antibiotics_s = None

    critical = [t for t in tasks if t.critical]
    if critical:
        done = sum(1 for t in critical if t.completed)
        bundle.compliance_pct = round(done / len(critical) * 100.0, 1)
    else:
        bundle.compliance_pct = None


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def bundle_to_dict(bundle: SepsisBundle) -> Dict[str, Any]:
    """Serialise a bundle (plus a live countdown) to a JSON-safe dict."""
    now = datetime.now(timezone.utc)
    elapsed_s = (now - _aware(bundle.started_at)).total_seconds()

    specs = protocol.HOUR1_BUNDLE.task_map()
    tasks_out: List[Dict[str, Any]] = []
    for t in sorted(bundle.tasks, key=lambda x: x.order_index):
        spec = specs.get(t.task_key)
        deadline_s = t.target_minutes * 60
        overdue = (
            not t.completed
            and bundle.status == "open"
            and elapsed_s > deadline_s
        )
        tasks_out.append(
            {
                "task_key": t.task_key,
                "label": spec.label if spec else t.task_key,
                "conditional": spec.conditional if spec else None,
                "depends_on": spec.depends_on if spec else None,
                "order": t.order_index,
                "target_minutes": t.target_minutes,
                "critical": t.critical,
                "completed": t.completed,
                "completed_at": (
                    t.completed_at.isoformat() if t.completed_at else None
                ),
                "minutes_from_start": (
                    round(t.minutes_from_start, 1)
                    if t.minutes_from_start is not None
                    else None
                ),
                "overdue": overdue,
                "note": t.note,
            }
        )

    # Countdown against the tightest outstanding critical deadline.
    remaining_s: Optional[float] = None
    if bundle.status == "open":
        outstanding = [
            t for t in bundle.tasks if t.critical and not t.completed
        ]
        if outstanding:
            tightest = min(t.target_minutes for t in outstanding) * 60
            remaining_s = tightest - elapsed_s

    return {
        "id": bundle.id,
        "patient_id": bundle.patient_id,
        "alert_id": bundle.alert_id,
        "protocol_version": bundle.protocol_version,
        "status": bundle.status,
        "risk_level_at_start": bundle.risk_level_at_start,
        "started_at": bundle.started_at.isoformat() if bundle.started_at else None,
        "closed_at": bundle.closed_at.isoformat() if bundle.closed_at else None,
        "elapsed_seconds": round(elapsed_s, 1),
        "seconds_remaining": (
            round(remaining_s, 1) if remaining_s is not None else None
        ),
        "time_to_antibiotics_s": bundle.time_to_antibiotics_s,
        "compliance_pct": bundle.compliance_pct,
        "tasks": tasks_out,
    }
