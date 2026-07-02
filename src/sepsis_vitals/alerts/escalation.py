"""
sepsis_vitals.alerts.escalation — Alert lifecycle state machine.

Manages alert acknowledgement, escalation, snooze, and resolution with
a full audit trail.  Escalation tiers:

    Tier 1  (1x timeout) → Charge Nurse
    Tier 2  (2x timeout) → Attending Physician
    Tier 3  (3x timeout) → Rapid Response Team

Thread-safe, in-memory with optional SQLite persistence (same pattern
as ``sepsis_vitals.ml.state_store``).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class AlertStatus(str, Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SNOOZED = "snoozed"


class EscalationTier(int, Enum):
    NONE = 0
    CHARGE_NURSE = 1
    ATTENDING = 2
    RAPID_RESPONSE = 3


TIER_LABELS = {
    EscalationTier.NONE: "none",
    EscalationTier.CHARGE_NURSE: "charge_nurse",
    EscalationTier.ATTENDING: "attending_physician",
    EscalationTier.RAPID_RESPONSE: "rapid_response_team",
}

DEFAULT_ESCALATION_TIMEOUT_MINUTES = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrackedAlert:
    """In-memory representation of a tracked alert."""

    alert_id: str
    patient_id: str
    risk_level: str
    created_at: datetime
    status: AlertStatus = AlertStatus.PENDING
    current_tier: EscalationTier = EscalationTier.NONE
    snoozed_until: Optional[datetime] = None
    audit_trail: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Escalation Manager
# ---------------------------------------------------------------------------


class AlertEscalationManager:
    """Manages the lifecycle of sepsis alerts with tiered escalation.

    Parameters
    ----------
    escalation_timeout_minutes : int
        Minutes before an unacknowledged alert escalates to the next tier.
    db_path : str or None
        Path to a SQLite file for persistence.  ``None`` keeps state
        purely in-memory (tests / dev).
    """

    def __init__(
        self,
        escalation_timeout_minutes: int = DEFAULT_ESCALATION_TIMEOUT_MINUTES,
        db_path: Optional[str] = "models/alert_escalation.db",
    ) -> None:
        self.escalation_timeout_minutes = escalation_timeout_minutes
        self._lock = threading.Lock()
        self._alerts: dict[str, TrackedAlert] = {}

        # Optional SQLite persistence
        self._conn: Optional[sqlite3.Connection] = None
        if db_path is not None:
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(p), check_same_thread=False, timeout=10.0
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.row_factory = sqlite3.Row
            self._init_db()
            self._load_active_alerts()

    # ------------------------------------------------------------------
    # SQLite schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tracked_alerts (
                alert_id     TEXT PRIMARY KEY,
                patient_id   TEXT NOT NULL,
                risk_level   TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                current_tier INTEGER NOT NULL DEFAULT 0,
                snoozed_until TEXT
            );

            CREATE TABLE IF NOT EXISTS alert_audit_trail (
                id         TEXT PRIMARY KEY,
                alert_id   TEXT NOT NULL,
                action     TEXT NOT NULL,
                user_id    TEXT,
                detail     TEXT,
                timestamp  TEXT NOT NULL,
                FOREIGN KEY (alert_id) REFERENCES tracked_alerts(alert_id)
            );

            CREATE INDEX IF NOT EXISTS idx_audit_alert
                ON alert_audit_trail(alert_id);
            CREATE INDEX IF NOT EXISTS idx_tracked_status
                ON tracked_alerts(status);
            """
        )
        self._conn.commit()

    def _load_active_alerts(self) -> None:
        """Restore non-terminal alerts from SQLite into memory."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM tracked_alerts WHERE status NOT IN ('resolved')"
        ).fetchall()
        for row in rows:
            alert = TrackedAlert(
                alert_id=row["alert_id"],
                patient_id=row["patient_id"],
                risk_level=row["risk_level"],
                created_at=datetime.fromisoformat(row["created_at"]),
                status=AlertStatus(row["status"]),
                current_tier=EscalationTier(row["current_tier"]),
                snoozed_until=(
                    datetime.fromisoformat(row["snoozed_until"])
                    if row["snoozed_until"]
                    else None
                ),
            )
            # Restore audit trail
            trail_rows = self._conn.execute(
                "SELECT action, user_id, detail, timestamp "
                "FROM alert_audit_trail WHERE alert_id = ? ORDER BY timestamp",
                (alert.alert_id,),
            ).fetchall()
            alert.audit_trail = [
                {
                    "action": r["action"],
                    "user_id": r["user_id"],
                    "detail": r["detail"],
                    "timestamp": r["timestamp"],
                }
                for r in trail_rows
            ]
            self._alerts[alert.alert_id] = alert

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_alert(self, alert: TrackedAlert) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tracked_alerts
                (alert_id, patient_id, risk_level, created_at, status,
                 current_tier, snoozed_until)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert.alert_id,
                alert.patient_id,
                alert.risk_level,
                alert.created_at.isoformat(),
                alert.status.value,
                alert.current_tier.value,
                alert.snoozed_until.isoformat() if alert.snoozed_until else None,
            ),
        )
        self._conn.commit()

    def _persist_audit_entry(self, alert_id: str, entry: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT INTO alert_audit_trail (id, alert_id, action, user_id, detail, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                alert_id,
                entry["action"],
                entry.get("user_id"),
                entry.get("detail"),
                entry["timestamp"],
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Audit helper
    # ------------------------------------------------------------------

    def _add_audit(
        self,
        alert: TrackedAlert,
        action: str,
        user_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> dict[str, Any]:
        entry = {
            "action": action,
            "user_id": user_id,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        alert.audit_trail.append(entry)
        self._persist_audit_entry(alert.alert_id, entry)
        return entry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_alert(
        self,
        alert_id: str,
        patient_id: str,
        risk_level: str,
        created_at: Optional[datetime] = None,
    ) -> TrackedAlert:
        """Start tracking a new alert for potential escalation."""
        now = created_at or datetime.now(timezone.utc)
        with self._lock:
            if alert_id in self._alerts:
                return self._alerts[alert_id]

            alert = TrackedAlert(
                alert_id=alert_id,
                patient_id=patient_id,
                risk_level=risk_level,
                created_at=now,
            )
            self._add_audit(alert, "registered", detail=f"risk_level={risk_level}")
            self._alerts[alert_id] = alert
            self._persist_alert(alert)
            logger.info(
                "Alert registered for escalation tracking: %s (patient=%s, risk=%s)",
                alert_id, patient_id, risk_level,
            )
            return alert

    def acknowledge_alert(
        self, alert_id: str, user_id: str
    ) -> dict[str, Any]:
        """Acknowledge an alert, stopping the escalation clock.

        Returns a dict with the acknowledgement timestamp and time-to-ack.
        Raises ``KeyError`` if the alert_id is not tracked.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise KeyError(f"Alert {alert_id} not found")

            now = datetime.now(timezone.utc)
            time_to_ack_s = (now - alert.created_at).total_seconds()

            alert.status = AlertStatus.ACKNOWLEDGED
            alert.snoozed_until = None
            self._add_audit(
                alert, "acknowledged", user_id=user_id,
                detail=f"time_to_ack_s={time_to_ack_s:.1f}",
            )
            self._persist_alert(alert)

            logger.info(
                "Alert acknowledged: %s by user %s (%.1fs)",
                alert_id, user_id, time_to_ack_s,
            )
            return {
                "alert_id": alert_id,
                "acknowledged_at": now.isoformat(),
                "time_to_ack_seconds": round(time_to_ack_s, 1),
                "status": alert.status.value,
            }

    def resolve_alert(
        self, alert_id: str, user_id: str, reason: Optional[str] = None
    ) -> dict[str, Any]:
        """Mark an alert as resolved.

        Raises ``KeyError`` if the alert_id is not tracked.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise KeyError(f"Alert {alert_id} not found")

            now = datetime.now(timezone.utc)
            alert.status = AlertStatus.RESOLVED
            alert.snoozed_until = None
            self._add_audit(
                alert, "resolved", user_id=user_id,
                detail=reason,
            )
            self._persist_alert(alert)

            logger.info("Alert resolved: %s by user %s", alert_id, user_id)
            return {
                "alert_id": alert_id,
                "resolved_at": now.isoformat(),
                "reason": reason,
                "status": alert.status.value,
            }

    def snooze_alert(
        self, alert_id: str, user_id: str, snooze_minutes: int = 15
    ) -> dict[str, Any]:
        """Delay escalation for the given number of minutes.

        Raises ``KeyError`` if the alert_id is not tracked.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise KeyError(f"Alert {alert_id} not found")

            now = datetime.now(timezone.utc)
            from datetime import timedelta

            alert.snoozed_until = now + timedelta(minutes=snooze_minutes)
            alert.status = AlertStatus.SNOOZED
            self._add_audit(
                alert, "snoozed", user_id=user_id,
                detail=f"snooze_minutes={snooze_minutes}",
            )
            self._persist_alert(alert)

            logger.info(
                "Alert snoozed: %s for %d min by user %s",
                alert_id, snooze_minutes, user_id,
            )
            return {
                "alert_id": alert_id,
                "snoozed_until": alert.snoozed_until.isoformat(),
                "snooze_minutes": snooze_minutes,
                "status": alert.status.value,
            }

    def check_escalations(self) -> list[dict[str, Any]]:
        """Return alerts that need escalation based on elapsed time.

        Each returned dict contains ``alert_id``, ``patient_id``,
        ``risk_level``, ``tier``, and ``tier_label``.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        timeout = timedelta(minutes=self.escalation_timeout_minutes)
        escalations: list[dict[str, Any]] = []

        with self._lock:
            for alert in self._alerts.values():
                # Only escalate pending or previously-escalated alerts
                if alert.status not in (
                    AlertStatus.PENDING,
                    AlertStatus.ESCALATED,
                    AlertStatus.SNOOZED,
                ):
                    continue

                # Respect snooze window
                if (
                    alert.status == AlertStatus.SNOOZED
                    and alert.snoozed_until is not None
                    and now < alert.snoozed_until
                ):
                    continue

                # If snoozed and snooze expired, treat as pending again
                if (
                    alert.status == AlertStatus.SNOOZED
                    and alert.snoozed_until is not None
                    and now >= alert.snoozed_until
                ):
                    alert.status = AlertStatus.PENDING
                    alert.snoozed_until = None

                elapsed = now - alert.created_at

                # Determine which tier we should be at
                if elapsed >= timeout * 3:
                    target_tier = EscalationTier.RAPID_RESPONSE
                elif elapsed >= timeout * 2:
                    target_tier = EscalationTier.ATTENDING
                elif elapsed >= timeout:
                    target_tier = EscalationTier.CHARGE_NURSE
                else:
                    continue  # not yet due for escalation

                # Only escalate if we haven't already reached this tier
                if target_tier.value <= alert.current_tier.value:
                    continue

                alert.current_tier = target_tier
                alert.status = AlertStatus.ESCALATED
                self._add_audit(
                    alert, "escalated",
                    detail=f"tier={target_tier.value} ({TIER_LABELS[target_tier]})",
                )
                self._persist_alert(alert)

                escalations.append({
                    "alert_id": alert.alert_id,
                    "patient_id": alert.patient_id,
                    "risk_level": alert.risk_level,
                    "tier": target_tier.value,
                    "tier_label": TIER_LABELS[target_tier],
                    "elapsed_minutes": round(elapsed.total_seconds() / 60, 1),
                })

                logger.warning(
                    "Alert escalated: %s → %s (patient=%s, elapsed=%.1fm)",
                    alert.alert_id,
                    TIER_LABELS[target_tier],
                    alert.patient_id,
                    elapsed.total_seconds() / 60,
                )

        return escalations

    def get_alert_lifecycle(self, alert_id: str) -> list[dict[str, Any]]:
        """Return the full audit trail for an alert.

        Raises ``KeyError`` if the alert_id is not tracked.  If the alert
        has been resolved and evicted from memory, falls back to SQLite.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is not None:
                return list(alert.audit_trail)

        # Fallback: check SQLite for resolved alerts
        if self._conn is not None:
            rows = self._conn.execute(
                "SELECT action, user_id, detail, timestamp "
                "FROM alert_audit_trail WHERE alert_id = ? ORDER BY timestamp",
                (alert_id,),
            ).fetchall()
            if rows:
                return [
                    {
                        "action": r["action"],
                        "user_id": r["user_id"],
                        "detail": r["detail"],
                        "timestamp": r["timestamp"],
                    }
                    for r in rows
                ]

        raise KeyError(f"Alert {alert_id} not found")

    def get_alert_status(self, alert_id: str) -> dict[str, Any]:
        """Return current status info for a tracked alert."""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise KeyError(f"Alert {alert_id} not found")
            return {
                "alert_id": alert.alert_id,
                "patient_id": alert.patient_id,
                "risk_level": alert.risk_level,
                "status": alert.status.value,
                "current_tier": alert.current_tier.value,
                "tier_label": TIER_LABELS[alert.current_tier],
                "created_at": alert.created_at.isoformat(),
                "snoozed_until": (
                    alert.snoozed_until.isoformat()
                    if alert.snoozed_until
                    else None
                ),
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[AlertEscalationManager] = None
_init_lock = threading.Lock()


def get_escalation_manager() -> AlertEscalationManager:
    """Return (or create) the module-level escalation manager singleton."""
    global _manager
    if _manager is None:
        with _init_lock:
            if _manager is None:
                _manager = AlertEscalationManager()
    return _manager
