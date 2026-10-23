"""Evaluate recorded predictions against adjudicated outcomes, locally.

Patient-level analysis (one standard approach; the study's locked analysis
plan decides the final one): for each patient with an outcome, the highest
model output and the highest rule-based scores (UVA, NEWS2-style, qSOFA)
recorded during the stay are compared with the outcome. Where an onset time
is recorded, the share of positive patients with a recorded alert before
onset, and the lead time, are reported.

Only aggregates are returned: no patient identifiers, values or times.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _json(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


def load_study_rows(db: Any, site_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """One row per registered patient with predictions: maxima and outcome."""
    from sepsis_vitals.db import Patient, PredictionRecord, StudyOutcome

    patients = db.query(Patient)
    if site_id is not None:
        patients = patients.filter(Patient.site_id == site_id)
    by_id = {str(p.id): p for p in patients}
    if not by_id:
        return []
    latest: Dict[str, Any] = {}
    for outcome in db.query(StudyOutcome).filter(StudyOutcome.patient_id.in_(list(by_id))):
        key = str(outcome.patient_id)
        if key not in latest or (outcome.created_at or 0) >= (latest[key].created_at or 0):
            latest[key] = outcome
    rows: Dict[str, Dict[str, Any]] = {}
    for record in db.query(PredictionRecord).filter(PredictionRecord.patient_id.in_(list(by_id))):
        key = str(record.patient_id)
        scores = _json(record.output_scores).get("clinical_scores", {})
        row = rows.setdefault(key, {"model": [], "uva": [], "news2": [], "qsofa": [], "alerts": []})
        row["model"].append(float(record.risk_probability))
        for name, field in (("uva", "uva"), ("news2", "news2_style"), ("qsofa", "qsofa")):
            if scores.get(field) is not None:
                row[name].append(float(scores[field]))
        if record.alert_fired and record.created_at is not None:
            row["alerts"].append(record.created_at)
    out = []
    for key, row in rows.items():
        outcome = latest.get(key)
        out.append({
            "sepsis": None if outcome is None else bool(outcome.sepsis),
            "onset_at": None if outcome is None else outcome.onset_at,
            "definition": None if outcome is None else outcome.definition,
            "max": {name: (max(vals) if vals else None) for name, vals in row.items() if name != "alerts"},
            "alerts": sorted(row["alerts"]),
        })
    return out


def _utc(value: Any) -> Any:
    from datetime import timezone

    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


def evaluate_local_study(db: Any, site_id: Optional[str] = None, n_boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
    from sepsis_vitals.ml.uncertainty import paired_cluster_bootstrap

    rows = [r for r in load_study_rows(db, site_id) if r["sepsis"] is not None]
    y = np.array([int(r["sepsis"]) for r in rows])
    report: Dict[str, Any] = {
        "claim_scope": ("Local retrospective evaluation of recorded outputs against adjudicated outcomes. "
                        "Interpret only under the study's approved protocol and analysis plan."),
        "analysis": "patient level: highest recorded value during the stay against the outcome",
        "patients": len(rows),
        "sepsis_positive": int(y.sum()) if len(y) else 0,
        "definitions": sorted({r["definition"] for r in rows if r["definition"]}),
    }
    if len(rows) == 0 or len(set(y.tolist())) < 2:
        report["status"] = "insufficient outcomes: both positive and negative patients are needed"
        return report
    arms: Dict[str, Sequence[float]] = {}
    for name in ("model", "uva", "news2", "qsofa"):
        values = [r["max"][name] for r in rows]
        if all(v is not None for v in values):
            arms[name] = [float(v) for v in values]
    boot = paired_cluster_bootstrap(list(range(len(rows))), [int(v) for v in y], arms,
                                    reference="model" if "model" in arms else None,
                                    metrics=("auroc", "auprc"), n_boot=n_boot, seed=seed)
    report["discrimination"] = boot
    with_onset = [r for r in rows if r["sepsis"] and r["onset_at"] is not None]
    if with_onset:
        leads = []
        flagged = 0
        for r in with_onset:
            onset = _utc(r["onset_at"])
            before = [a for a in (_utc(t) for t in r["alerts"]) if a <= onset]
            if before:
                flagged += 1
                leads.append((onset - before[0]).total_seconds() / 3600)
        report["alerting"] = {
            "positives_with_onset": len(with_onset),
            "alerted_before_onset": flagged,
            "lead_time_hours_median": round(float(np.median(leads)), 1) if leads else None,
        }
    report["status"] = "computed"
    return report
