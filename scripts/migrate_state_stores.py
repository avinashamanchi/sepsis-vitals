#!/usr/bin/env python3
"""Copy legacy plaintext state stores into the protected v1 format.

Legacy files (``patient_state.db``, ``alert_escalation.db``) stored patient
identifiers, user IDs and audit text in plaintext. The application no longer
reads them. This script writes protected copies to a *separate* output
directory; it never modifies or deletes the source files. Review the output,
point SEPSIS_STATE_DIR at it, then dispose of the legacy files under your
data-retention policy.

Run with the same SEPSIS_PII_KEY the application uses (identifiers become
keyed references or AES-GCM ciphertext under that key).

    python scripts/migrate_state_stores.py --source models --output state --dry-run
    python scripts/migrate_state_stores.py --source models --output state
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _rows(path: Path, query: str) -> list:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


def migrate(source: Path, output: Path, dry_run: bool = False) -> dict:
    from sepsis_vitals.alerts.escalation import STORE_FILENAME as ESC_V1, _seal
    from sepsis_vitals.ml.state_store import STORE_FILENAME as STATE_V1, PatientStateStore, _ref

    if source.resolve() == output.resolve():
        raise SystemExit("--output must differ from --source (sources are never rewritten)")
    summary: dict = {}

    legacy_state = source / "patient_state.db"
    if legacy_state.exists():
        patients = _rows(legacy_state, "SELECT patient_id, baseline_risk, created_at FROM patients")
        predictions = _rows(
            legacy_state,
            "SELECT patient_id, timestamp, risk_probability, risk_level, created_at FROM predictions",
        )
        summary["patient_state"] = {"patients": len(patients), "predictions": len(predictions)}
        if not dry_run:
            output.mkdir(parents=True, exist_ok=True)
            target = output / STATE_V1
            if target.exists():
                raise SystemExit(f"{target} already exists; refusing to overwrite")
            store = PatientStateStore(str(target))  # creates schema and permissions
            with store._lock:
                store._conn.executemany(
                    "INSERT INTO patients (patient_id, baseline_risk, created_at) VALUES (?, ?, ?)",
                    [(_ref(r["patient_id"]), r["baseline_risk"], r["created_at"]) for r in patients],
                )
                store._conn.executemany(
                    "INSERT INTO predictions (patient_id, timestamp, risk_probability, risk_level, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(_ref(r["patient_id"]), r["timestamp"], r["risk_probability"], r["risk_level"],
                      r["created_at"]) for r in predictions],
                )
                store._conn.commit()
            store.close()

    legacy_esc = source / "alert_escalation.db"
    if legacy_esc.exists():
        alerts = _rows(legacy_esc, "SELECT * FROM tracked_alerts")
        audit = _rows(legacy_esc, "SELECT * FROM alert_audit_trail")
        summary["alert_escalation"] = {"alerts": len(alerts), "audit_entries": len(audit)}
        if not dry_run:
            from sepsis_vitals.alerts.escalation import AlertEscalationManager

            output.mkdir(parents=True, exist_ok=True)
            target = output / ESC_V1
            if target.exists():
                raise SystemExit(f"{target} already exists; refusing to overwrite")
            manager = AlertEscalationManager(db_path=str(target))  # schema and permissions
            conn = manager._conn
            assert conn is not None
            conn.executemany(
                "INSERT INTO tracked_alerts (alert_id, patient_id, risk_level, created_at, status, "
                "current_tier, snoozed_until) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(r["alert_id"], _seal(r["patient_id"]), r["risk_level"], r["created_at"], r["status"],
                  r["current_tier"], r["snoozed_until"]) for r in alerts],
            )
            conn.executemany(
                "INSERT INTO alert_audit_trail (id, alert_id, action, user_id, detail, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(r["id"], r["alert_id"], r["action"], _seal(r["user_id"]), _seal(r["detail"]),
                  r["timestamp"]) for r in audit],
            )
            conn.commit()
            conn.close()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="directory with legacy stores")
    parser.add_argument("--output", type=Path, required=True, help="new, separate state directory")
    parser.add_argument("--dry-run", action="store_true", help="count rows only")
    opts = parser.parse_args()
    summary = migrate(opts.source, opts.output, dry_run=opts.dry_run)
    print(summary or "no legacy stores found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
