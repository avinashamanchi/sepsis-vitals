"""Append-only, hash-sealed record of every risk inference and alert decision.

Every risk result that leaves the service (an API response or a WebSocket
alert) is written here first. If the write fails, the result is withheld:
callers raise or broadcast "no risk calculated" instead (fail closed).

Each row is traceable to:

- the input: the canonical input payload (current observation, the history
  actually used, age, comorbidities, observation time) and its SHA-256;
- the model: model version, artifact and feature-schema digests and the
  preprocessing version (in ``output_scores.versions``);
- the threshold policy: its id (column) and digest (``output_scores``);
- the time and the request: ``created_at`` and ``request_id``;
- the path that produced it: ``source``.

``record_sha256`` seals the row's content; :func:`verify_record` detects any
later change. The table itself refuses UPDATE, DELETE and TRUNCATE (see
``sepsis_vitals.db``). The IP address is stored but not sealed, because
database drivers return it in different forms.

No record content is ever logged: failures log the error class only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SOURCES = ("api", "api-batch", "api-timeseries", "monitor", "copilot")

SEALED_FIELDS = (
    "id", "created_at", "patient_id", "user_id", "risk_probability", "risk_level",
    "alert_fired", "input_vitals", "output_scores", "top_risk_factors",
    "confidence_lower", "confidence_upper", "model_version", "recommendation",
    "request_id", "source", "threshold_policy", "input_sha256",
)
_JSON_FIELDS = ("input_vitals", "output_scores", "top_risk_factors")


class AuditWriteError(RuntimeError):
    """The audit record was not written; the result must not be released."""


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _timestamp(value)
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if hasattr(value, "isoformat"):  # pandas Timestamp, date
        return value.isoformat()
    return str(value)


def canonical(value: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def sha256_of(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _timestamp(value: Optional[datetime]) -> Optional[str]:
    """UTC, without offset, to the microsecond (SQLite drops the offset)."""
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.isoformat(timespec="microseconds")


def _seal(values: Dict[str, Any]) -> str:
    normalised: Dict[str, Any] = {}
    for name in SEALED_FIELDS:
        value = values.get(name)
        if name in _JSON_FIELDS and isinstance(value, str):
            value = json.loads(value)
        elif name == "created_at":
            value = _timestamp(value)
        elif name in ("id", "user_id") and value is not None:
            value = str(value)
        elif name in ("risk_probability", "confidence_lower", "confidence_upper") and value is not None:
            value = float(value)
        elif name == "alert_fired":
            value = bool(value)
        normalised[name] = value
    return sha256_of(normalised)


def input_payload(
    vitals: Dict[str, Any],
    history: Any,
    age_years: Optional[int],
    comorbidities: Optional[Dict[str, Any]],
    observed_at: Optional[str],
) -> Dict[str, Any]:
    """The complete model input, in the form recorded and hashed."""
    return {
        "observation": vitals,
        "observed_at": observed_at,
        "history": list(history or []),
        "age_years": age_years,
        "comorbidities": comorbidities,
    }


def record_prediction(
    *,
    patient_id: str,
    user_id: Optional[str],
    result: Dict[str, Any],
    payload: Dict[str, Any],
    source: str,
    request_id: str,
    ip_address: Optional[str] = None,
) -> str:
    """Persist one inference and its alert decision; return the record id.

    Raises :class:`AuditWriteError` on any failure. The caller must then
    withhold the result.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown audit source {source!r}")
    try:
        from sepsis_vitals.db import PredictionRecord, SessionLocal
    except Exception as exc:  # pragma: no cover - import failure is a deployment error
        raise AuditWriteError(type(exc).__name__) from exc

    versions = result.get("versions") or {}
    policy = result.get("threshold_policy") or {}
    interval = result.get("confidence_interval") or {}
    output = {
        "clinical_scores": result.get("clinical_scores", {}),
        "rule_risk_level": result.get("rule_risk_level"),
        "model_risk_level": result.get("model_risk_level"),
        "provenance": result.get("provenance", {}),
        "versions": versions,
        "threshold_policy": policy,
        "uncertainty": result.get("uncertainty"),
        "explanation": result.get("explanation"),
        "data_quality": result.get("data_quality"),
        "observed_at": result.get("observed_at"),
        "decision_support_notice": result.get("decision_support_notice"),
        "clinical_use": "not-permitted",
    }
    if "monitor_decision" in result:
        output["monitor_decision"] = result["monitor_decision"]
    values: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "patient_id": patient_id,
        "user_id": user_id if user_id not in (None, "anonymous") else None,
        "risk_probability": float(result["risk_probability"]),
        "risk_level": result["risk_level"],
        "alert_fired": bool(result.get("alert", False)),
        "input_vitals": canonical(payload),
        "output_scores": canonical(output),
        "top_risk_factors": canonical(result.get("top_risk_factors", [])),
        "confidence_lower": interval.get("lower"),
        "confidence_upper": interval.get("upper"),
        "model_version": versions.get("model_version") or (result.get("model") or {}).get("version"),
        "recommendation": result.get("recommendation"),
        "request_id": request_id,
        "source": source,
        "threshold_policy": policy.get("id"),
        "input_sha256": sha256_of(payload),
    }
    values["record_sha256"] = _seal(values)

    try:
        db = SessionLocal()
    except Exception as exc:
        logger.error("Audit record not written (%s); result withheld", type(exc).__name__)
        raise AuditWriteError(type(exc).__name__) from exc
    try:
        db.add(PredictionRecord(**values, ip_address=ip_address))
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Audit record not written (%s); result withheld", type(exc).__name__)
        raise AuditWriteError(type(exc).__name__) from exc
    finally:
        db.close()
    return values["id"]


def verify_record(record: Any) -> Optional[bool]:
    """True if the row is unchanged since it was written, False if it changed.

    None for rows written before migration 006 (no seal to check).
    """
    sealed = getattr(record, "record_sha256", None)
    if sealed is None:
        return None
    values = {name: getattr(record, name) for name in SEALED_FIELDS}
    return hmac.compare_digest(_seal(values), sealed)
