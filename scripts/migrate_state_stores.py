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


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_state(patients: list, predictions: list) -> tuple:
    from sepsis_vitals.ml.state_store import _ref

    return (
        sorted((_ref(r["patient_id"]), r["baseline_risk"], r["created_at"]) for r in patients),
        sorted((_ref(r["patient_id"]), r["timestamp"], r["risk_probability"], r["risk_level"], r["created_at"])
               for r in predictions),
    )


def _verify_state(path: Path, patients: list, predictions: list) -> None:
    got = (
        sorted(tuple(r) for r in _rows(path, "SELECT patient_id, baseline_risk, created_at FROM patients")),
        sorted(tuple(r) for r in _rows(
            path, "SELECT patient_id, timestamp, risk_probability, risk_level, created_at FROM predictions")),
    )
    if got != _expected_state(patients, predictions):
        raise SystemExit(f"{path.name}: copy does not match the source; not using it")


def _verify_escalation(path: Path, alerts: list, audit: list) -> None:
    from sepsis_vitals.alerts.escalation import _open

    copied_alerts = {r["alert_id"]: r for r in _rows(path, "SELECT * FROM tracked_alerts")}
    copied_audit = {r["id"]: r for r in _rows(path, "SELECT * FROM alert_audit_trail")}
    ok = len(copied_alerts) == len(alerts) and len(copied_audit) == len(audit)
    for r in alerts if ok else []:
        c = copied_alerts.get(r["alert_id"])
        ok = ok and c is not None and _open(c["patient_id"]) == r["patient_id"] and all(
            c[k] == r[k] for k in ("risk_level", "created_at", "status", "current_tier", "snoozed_until"))
    for r in audit if ok else []:
        c = copied_audit.get(r["id"])
        ok = ok and c is not None and _open(c["user_id"]) == r["user_id"] and _open(c["detail"]) == r["detail"] \
            and all(c[k] == r[k] for k in ("alert_id", "action", "timestamp"))
    if not ok:
        raise SystemExit(f"{path.name}: copy does not match the source; not using it")


def _partial(target: Path) -> Path:
    """Fresh temporary path next to *target* (a leftover from an interrupted run is replaced)."""
    tmp = target.with_name(target.name + ".partial")
    for leftover in (tmp, tmp.with_name(tmp.name + "-journal")):
        if leftover.exists():
            leftover.unlink()  # our own incomplete output, never a source file
    return tmp


def migrate(source: Path, output: Path, dry_run: bool = False) -> dict:
    """Write verified, protected copies of the legacy stores in *source* to *output*.

    Each copy is written to ``<name>.partial``, compared row by row with the
    source (keyed references recomputed, sealed values decrypted), and only
    then renamed into place. A run that is interrupted leaves no final file
    behind and can simply be repeated; a final file that already matches the
    source is skipped. Source files are hashed before and after and must be
    unchanged.
    """
    import os

    from sepsis_vitals.alerts.escalation import STORE_FILENAME as ESC_V1, _seal
    from sepsis_vitals.ml.state_store import STORE_FILENAME as STATE_V1, PatientStateStore, _ref
    from sepsis_vitals.security import FieldEncryptor

    if source.resolve() == output.resolve():
        raise SystemExit("--output must differ from --source (sources are never rewritten)")
    if not dry_run and not FieldEncryptor.get().enabled:
        raise SystemExit("SEPSIS_PII_KEY is not configured: the copies would not be protected")
    sources = [p for p in (source / "patient_state.db", source / "alert_escalation.db") if p.exists()]
    before = {p.name: _sha256(p) for p in sources}
    summary: dict = {}
    if not dry_run and sources:
        output.mkdir(mode=0o700, parents=True, exist_ok=True)

    legacy_state = source / "patient_state.db"
    if legacy_state.exists():
        patients = _rows(legacy_state, "SELECT patient_id, baseline_risk, created_at FROM patients")
        predictions = _rows(
            legacy_state,
            "SELECT patient_id, timestamp, risk_probability, risk_level, created_at FROM predictions",
        )
        summary["patient_state"] = {"patients": len(patients), "predictions": len(predictions)}
        if not dry_run:
            target = output / STATE_V1
            if target.exists():
                _verify_state(target, patients, predictions)  # refuses a mismatching file
                summary["patient_state"]["status"] = "already migrated and verified"
            else:
                tmp = _partial(target)
                store = PatientStateStore(str(tmp))  # creates schema and permissions
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
                _verify_state(tmp, patients, predictions)
                os.replace(tmp, target)
                summary["patient_state"]["status"] = "migrated and verified"
            summary["patient_state"]["mode"] = oct(target.stat().st_mode & 0o777)

    legacy_esc = source / "alert_escalation.db"
    if legacy_esc.exists():
        alerts = _rows(legacy_esc, "SELECT * FROM tracked_alerts")
        audit = _rows(legacy_esc, "SELECT * FROM alert_audit_trail")
        summary["alert_escalation"] = {"alerts": len(alerts), "audit_entries": len(audit)}
        if not dry_run:
            from sepsis_vitals.alerts.escalation import AlertEscalationManager

            target = output / ESC_V1
            if target.exists():
                _verify_escalation(target, alerts, audit)
                summary["alert_escalation"]["status"] = "already migrated and verified"
            else:
                tmp = _partial(target)
                manager = AlertEscalationManager(db_path=str(tmp))  # schema and permissions
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
                _verify_escalation(tmp, alerts, audit)
                os.replace(tmp, target)
                summary["alert_escalation"]["status"] = "migrated and verified"
            summary["alert_escalation"]["mode"] = oct(target.stat().st_mode & 0o777)

    after = {p.name: _sha256(p) for p in sources}
    if after != before:
        raise SystemExit("a source file changed during the run; investigate before using the copies")
    if sources:
        summary["sources_unchanged"] = True
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
