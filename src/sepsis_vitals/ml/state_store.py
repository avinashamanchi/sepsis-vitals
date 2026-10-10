"""
sepsis_vitals.ml.state_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQLite-backed persistent patient state storage.

Replaces the ephemeral in-memory dict that was wiped on server restart.
Supports multi-worker deployments via WAL mode.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sepsis_vitals.security import blind_index_candidates, compute_blind_index, log_ref

logger = logging.getLogger(__name__)

#: Bumped from the legacy ``patient_state.db``, which stored identifiers in
#: plaintext. Legacy files are never rewritten automatically; see
#: scripts/migrate_state_stores.py.
STORE_FILENAME = "patient_state.v1.db"
LEGACY_FILENAME = "patient_state.db"


def _ref(patient_id: str) -> str:
    """Keyed, irreversible reference stored instead of the patient identifier.

    Lookups only ever need equality, so the store never holds an identifier
    it could reveal. HMAC-SHA256 with the current PII key (plain SHA-256 in
    development when no key is configured).
    """
    return compute_blind_index(str(patient_id))


def _refs(patient_id: str) -> List[str]:
    """References under every configured PII key, current first.

    During a key rotation a patient's rows may still carry the reference
    computed with a previous key; reads match all of them, and the next
    write moves them to the current reference (see ``_rekey``).
    """
    return blind_index_candidates(str(patient_id))


def _in(refs: List[str]) -> str:
    return "(" + ",".join("?" * len(refs)) + ")"


def default_store_path(state_dir: Optional[str] = None) -> Path:
    base = Path(state_dir or os.getenv("SEPSIS_STATE_DIR") or "models")
    return base / STORE_FILENAME


def secure_sqlite_file(path: Path) -> None:
    """Owner-only permissions on the database (SQLite gives WAL/SHM the same mode)."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.warning("Could not restrict permissions on %s", path)


def warn_if_legacy(path: Path, legacy_name: str) -> None:
    legacy = path.parent / legacy_name
    if legacy.exists():
        logger.warning(
            "Legacy plaintext state store %s was left untouched and is not read. "
            "Migrate or delete it deliberately (scripts/migrate_state_stores.py).",
            legacy,
        )


@dataclass
class StoredPrediction:
    """Minimal prediction record for persistence."""

    timestamp: str
    risk_probability: float
    risk_level: str


class PatientStateStore:
    """SQLite-backed patient state persistence.

    Stores prediction history and baseline risk so that patient
    monitoring survives server restarts and works across multiple
    Uvicorn/Gunicorn workers.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database file. Defaults to
        ``$SEPSIS_STATE_DIR/patient_state.v1.db``. Patient identifiers are
        stored only as keyed references (:func:`_ref`), and the file is
        created owner-read/write only. This protects identifiers, not the
        whole file: risk values and timestamps are stored in clear, so the
        state directory still belongs on an encrypted volume.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = Path(db_path) if db_path else default_store_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        warn_if_legacy(self._db_path, LEGACY_FILENAME)

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        secure_sqlite_file(self._db_path)
        # Enable WAL for concurrent reader/writer access across workers.
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Return rows as sqlite3.Row so we can access columns by name.
        self._conn.row_factory = sqlite3.Row

        self._lock = threading.Lock()

        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indices if they don't already exist."""
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id   TEXT PRIMARY KEY,
                    baseline_risk REAL,
                    created_at   REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id       TEXT    NOT NULL,
                    timestamp        TEXT    NOT NULL,
                    risk_probability REAL    NOT NULL,
                    risk_level       TEXT    NOT NULL,
                    created_at       REAL    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_patient_created
                    ON predictions (patient_id, created_at);
                """
            )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_prediction(
        self,
        patient_id: str,
        timestamp: str,
        risk_probability: float,
        risk_level: str,
    ) -> None:
        """Store a prediction and update baseline risk if needed.

        Baseline risk is set from the *first* prediction's probability
        once a patient has at least two recorded predictions (matching
        the original ``PatientMonitor`` logic).
        """
        now = time.time()

        try:
            with self._lock, self._conn:
                self._rekey(patient_id)
                # Ensure the patient row exists.
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO patients (patient_id, baseline_risk, created_at)
                    VALUES (?, NULL, ?)
                    """,
                    (_ref(patient_id), now),
                )

                # Insert the prediction.
                self._conn.execute(
                    """
                    INSERT INTO predictions
                        (patient_id, timestamp, risk_probability, risk_level, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (_ref(patient_id), timestamp, risk_probability, risk_level, now),
                )

                # Set baseline_risk from the 1st prediction once we have >= 2.
                # Only do this once (when baseline_risk is still NULL).
                row = self._conn.execute(
                    "SELECT baseline_risk FROM patients WHERE patient_id = ?",
                    (_ref(patient_id),),
                ).fetchone()

                if row is not None and row["baseline_risk"] is None:
                    count = self._conn.execute(
                        "SELECT COUNT(*) AS cnt FROM predictions WHERE patient_id = ?",
                        (_ref(patient_id),),
                    ).fetchone()["cnt"]

                    if count >= 2:
                        first = self._conn.execute(
                            """
                            SELECT risk_probability FROM predictions
                            WHERE patient_id = ?
                            ORDER BY created_at ASC
                            LIMIT 1
                            """,
                            (_ref(patient_id),),
                        ).fetchone()

                        if first is not None:
                            self._conn.execute(
                                "UPDATE patients SET baseline_risk = ? WHERE patient_id = ?",
                                (first["risk_probability"], _ref(patient_id)),
                            )
        except sqlite3.Error:
            logger.exception("Failed to store prediction for patient %s", log_ref(patient_id))
            raise

    def _rekey(self, patient_id: str) -> None:
        """Move a patient's rows from previous-key references to the current one.

        Called inside the write transaction, so after a PII key rotation each
        patient's history is re-keyed the next time the patient is scored.
        """
        current, *old = _refs(patient_id)
        if not old:
            return
        placeholders = _in(old)
        moved = self._conn.execute(
            f"UPDATE predictions SET patient_id = ? WHERE patient_id IN {placeholders}",  # nosec B608
            (current, *old),
        ).rowcount
        prior = self._conn.execute(
            f"SELECT baseline_risk, created_at FROM patients WHERE patient_id IN {placeholders} "  # nosec B608
            "ORDER BY created_at LIMIT 1",
            old,
        ).fetchone()
        if prior is not None:
            self._conn.execute(
                "INSERT OR IGNORE INTO patients (patient_id, baseline_risk, created_at) VALUES (?, ?, ?)",
                (current, prior["baseline_risk"], prior["created_at"]),
            )
            self._conn.execute(
                "UPDATE patients SET baseline_risk = COALESCE(baseline_risk, ?) WHERE patient_id = ?",
                (prior["baseline_risk"], current),
            )
            self._conn.execute(f"DELETE FROM patients WHERE patient_id IN {placeholders}", old)  # nosec B608
        if moved or prior is not None:
            logger.info("Re-keyed stored predictions for patient %s", log_ref(patient_id))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_predictions(
        self, patient_id: str, limit: int = 50
    ) -> List[StoredPrediction]:
        """Return recent predictions for *patient_id*, newest first."""
        try:
            with self._lock:
                refs = _refs(patient_id)
                rows = self._conn.execute(
                    f"""
                    SELECT timestamp, risk_probability, risk_level
                    FROM predictions
                    WHERE patient_id IN {_in(refs)}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,  # nosec B608 - placeholders only
                    (*refs, limit),
                ).fetchall()
        except sqlite3.Error:
            logger.exception(
                "Failed to retrieve predictions for patient %s", log_ref(patient_id)
            )
            return []

        # Reverse so the list is chronological (oldest first) — callers
        # that inspect the last N entries expect ascending time order.
        return [
            StoredPrediction(
                timestamp=r["timestamp"],
                risk_probability=r["risk_probability"],
                risk_level=r["risk_level"],
            )
            for r in reversed(rows)
        ]

    def get_baseline_risk(self, patient_id: str) -> Optional[float]:
        """Return the baseline risk for *patient_id*, or ``None``."""
        try:
            with self._lock:
                refs = _refs(patient_id)
                row = self._conn.execute(
                    f"SELECT baseline_risk FROM patients WHERE patient_id IN {_in(refs)} "  # nosec B608
                    "ORDER BY baseline_risk IS NULL LIMIT 1",
                    refs,
                ).fetchone()
        except sqlite3.Error:
            logger.exception(
                "Failed to retrieve baseline risk for patient %s", log_ref(patient_id)
            )
            return None

        if row is None:
            return None
        return row["baseline_risk"]

    # ------------------------------------------------------------------
    # Trend / deterioration
    # ------------------------------------------------------------------

    def get_trend(self, patient_id: str) -> Dict[str, Any]:
        """Compute trend from stored predictions.

        Returns the same shape as ``SepsisPredictor.get_patient_trend()``::

            {
                "patient_id": str,
                "n_observations": int,
                "trend": str,
                "risk_history": [{"timestamp", "risk_probability", "risk_level"}, ...],
                "deterioration": { ... },
            }
        """
        predictions = self.get_predictions(patient_id)

        trend_label = self.compute_trend_label(predictions)
        deterioration = self.detect_deterioration(predictions)

        return {
            "patient_id": patient_id,
            "n_observations": len(predictions),
            "trend": trend_label,
            "risk_history": [
                {
                    "timestamp": p.timestamp,
                    "risk_probability": p.risk_probability,
                    "risk_level": p.risk_level,
                }
                for p in predictions
            ],
            "deterioration": deterioration,
        }

    def compute_trend_label(self, predictions: List[StoredPrediction]) -> str:
        """Compute a human-readable trend label.

        Uses the last 3 predictions (or fewer, if not enough data).

        Returns one of: ``rapidly_worsening``, ``worsening``, ``stable``,
        ``improving``, ``rapidly_improving``, ``insufficient_data``.
        """
        if len(predictions) < 2:
            return "insufficient_data"

        recent = [p.risk_probability for p in predictions[-3:]]
        if len(recent) >= 2:
            delta = recent[-1] - recent[0]
            if delta > 0.1:
                return "rapidly_worsening"
            elif delta > 0.03:
                return "worsening"
            elif delta < -0.1:
                return "rapidly_improving"
            elif delta < -0.03:
                return "improving"
        return "stable"

    def detect_deterioration(
        self, predictions: List[StoredPrediction]
    ) -> Dict[str, Any]:
        """Detect deterioration signals in the prediction history.

        Checks for:
        * Risk jump > 0.15 in one observation
        * Crossing into *high* or *critical* risk level
        * Sustained high risk (> 60 %) for 3+ observations
        """
        if len(predictions) < 2:
            return {"detected": False, "reason": "insufficient_data"}

        current = predictions[-1]
        previous = predictions[-2]

        reasons: List[str] = []

        # 1. Rapid risk increase
        delta = current.risk_probability - previous.risk_probability
        if delta > 0.15:
            reasons.append(f"Risk jumped by {delta:.0%} in one observation")

        # 2. Crossed into high/critical
        risk_order = ["low", "moderate", "high", "critical"]
        curr_idx = (
            risk_order.index(current.risk_level)
            if current.risk_level in risk_order
            else 0
        )
        prev_idx = (
            risk_order.index(previous.risk_level)
            if previous.risk_level in risk_order
            else 0
        )
        if curr_idx > prev_idx and curr_idx >= 2:
            reasons.append(
                f"Risk escalated from {previous.risk_level} to {current.risk_level}"
            )

        # 3. Sustained high risk
        if len(predictions) >= 3:
            last_3 = [p.risk_probability for p in predictions[-3:]]
            if all(r > 0.6 for r in last_3):
                reasons.append("Sustained high risk (>60%) for 3+ observations")

        return {
            "detected": len(reasons) > 0,
            "reasons": reasons,
            "current_risk": current.risk_probability,
            "previous_risk": previous.risk_probability,
            "delta": delta,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_old_records(self, max_age_hours: int = 72) -> int:
        """Delete predictions older than *max_age_hours*.

        Also removes patient rows that no longer have any predictions.

        Returns the number of prediction rows deleted.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM predictions WHERE created_at < ?",
                    (cutoff,),
                )
                deleted = cursor.rowcount

                # Clean up orphaned patient rows.
                self._conn.execute(
                    """
                    DELETE FROM patients
                    WHERE patient_id NOT IN (
                        SELECT DISTINCT patient_id FROM predictions
                    )
                    """
                )

            logger.info(
                "Cleaned up %d prediction(s) older than %d hours",
                deleted,
                max_age_hours,
            )
            return deleted
        except sqlite3.Error:
            logger.exception("Failed to clean up old records")
            return 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            logger.exception("Error closing state store database connection")

    def __enter__(self) -> "PatientStateStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 — destructor must never raise
            pass
