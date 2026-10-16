"""
sepsis_vitals.routes.status

Readiness and model-status endpoints, kept distinct from liveness (/health):
/ready (database and migrations), /model/status (prediction readiness,
never clinical readiness) and /model/info.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from sepsis_vitals import dependencies as _deps
from sepsis_vitals.dependencies import check_rate_limit, verify_auth

router = APIRouter()
logger = logging.getLogger("sepsis_vitals.api")


def _core():
    """The application module, for runtime state (model, monitor, metrics).

    Looked up per request, never at import: this module does not import
    ``sepsis_vitals.api``, so it can be imported first, alone or in any order.
    """
    from sepsis_vitals import api

    return api


def _alembic_head() -> Optional[str]:
    """Head revision of the migration scripts shipped with this deployment.

    alembic.ini lives in the working directory in the container (/app) and
    at the repository root in development; the installed package location
    is neither. Returns None when no migration scripts are found.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    candidates = [Path(os.getenv("SEPSIS_ALEMBIC_DIR", "")), Path.cwd(), Path(__file__).resolve().parents[3]]
    for root in candidates:
        if str(root) and (root / "alembic.ini").exists() and (root / "alembic").is_dir():
            cfg = Config(str(root / "alembic.ini"))
            cfg.set_main_option("script_location", str(root / "alembic"))
            return ScriptDirectory.from_config(cfg).get_current_head()
    return None


def _audit_guard() -> str:
    """Whether the append-only triggers on prediction_records exist.

    "present", "missing", "no-table" (fresh development database before the
    first create) or "unknown" (query failed). Readiness requires
    "present" or "no-table": an unguarded audit table is not ready.
    """
    from sqlalchemy import inspect, text

    from sepsis_vitals.db import engine

    try:
        with engine.connect() as conn:
            if not inspect(conn).has_table("prediction_records"):
                return "no-table"
            if conn.dialect.name == "postgresql":
                count = conn.execute(text(
                    "SELECT count(*) FROM pg_trigger WHERE tgrelid = 'prediction_records'::regclass "
                    "AND tgname IN ('prediction_records_no_change', 'prediction_records_no_truncate')"
                )).scalar()
                return "present" if count == 2 else "missing"
            if conn.dialect.name == "sqlite":
                count = conn.execute(text(
                    "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' AND name IN "
                    "('prediction_records_no_update', 'prediction_records_no_delete')"
                )).scalar()
                return "present" if count == 2 else "missing"
    except Exception as exc:
        logger.warning("Readiness: audit guard check failed (%s)", type(exc).__name__)
    return "unknown"


@router.get("/ready")
async def readiness():
    """API readiness: database reachable and (when managed) migrations at head.

    Separate from liveness (/health) and prediction readiness (/model/status):
    the API can serve scores and patient data without a model.
    """
    def _check() -> Dict[str, Any]:
        from sqlalchemy import text as sql_text

        from sepsis_vitals.db import engine
        checks: Dict[str, Any] = {}
        try:
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
                try:
                    current = conn.execute(sql_text("SELECT version_num FROM alembic_version")).scalar()
                except Exception:
                    current = None
        except Exception as exc:
            logger.warning("Readiness: database unreachable (%s)", type(exc).__name__)
            return {"database": "unreachable", "migrations": "unknown"}
        checks["database"] = "ok"
        checks["audit_guard"] = _audit_guard()
        if current is None:
            checks["migrations"] = "unmanaged"
            return checks
        try:
            head = _alembic_head()
        except Exception as exc:
            logger.warning("Readiness: migration scripts unreadable (%s)", type(exc).__name__)
            head = None
        if head is None:
            checks["migrations"] = "unknown"
        else:
            checks["migrations"] = "at-head" if current == head else "behind"
        return checks

    checks = await asyncio.to_thread(_check)
    ready = checks["database"] == "ok" and (
        checks["migrations"] == "at-head"
        or (checks["migrations"] == "unmanaged" and not _deps._is_production)
    ) and checks.get("audit_guard") in ("present", "no-table")
    return JSONResponse(status_code=200 if ready else 503, content={"ready": ready, **checks})


@router.get("/model/status")
async def model_status():
    """Prediction readiness, validation status and provenance of the model.

    ``prediction_ready`` means a verified model is loaded. It never means
    clinically ready: ``clinically_ready`` stays false for every validation
    status this build knows about.
    """
    predictor = await asyncio.to_thread(_core()._get_predictor)
    status = dict(_core()._model_status)
    status.setdefault("prediction_ready", False)
    status.setdefault("clinically_ready", False)
    status.setdefault("clinical_use", "not-permitted")
    state = str(status.get("state"))
    status["checks"] = {
        # Artifacts are verified against manifest.json before unpickling, and
        # the current feature pipeline must produce every model feature.
        "artifact_integrity": {"ready": "verified", "unverified": "not verified (explicitly allowed)",
                               "invalid": "failed"}.get(state, "not checked"),
        "feature_schema": "compatible" if state in ("ready", "unverified")
        else "incompatible" if state == "incompatible" else "not checked",
    }
    if predictor is not None:
        policy = predictor.threshold_policy()
        status["threshold_policy"] = {k: policy[k] for k in ("id", "status", "sha256")}
    status["recent_inference"] = recent_inference_summary(list(_core()._recent_inferences))
    return status


def recent_inference_summary(entries: list) -> Dict[str, Any]:
    """Outcome counts, success rate and latency of the last inference attempts.

    "Success" means a recorded result was released. Contains no patient data.
    """
    import statistics

    outcomes: Dict[str, int] = {}
    for entry in entries:
        outcomes[entry["outcome"]] = outcomes.get(entry["outcome"], 0) + 1
    latencies = sorted(e["latency_ms"] for e in entries if e.get("latency_ms") is not None)
    summary: Dict[str, Any] = {
        "window": len(entries),
        "outcomes": outcomes,
        "success_rate": round(outcomes.get("scored", 0) / len(entries), 4) if entries else None,
        "latency_ms_p50": None,
        "latency_ms_p95": None,
    }
    if latencies:
        summary["latency_ms_p50"] = round(statistics.median(latencies), 1)
        summary["latency_ms_p95"] = round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 1)
    return summary


@router.get("/model/info", dependencies=[Depends(check_rate_limit)])
async def model_info(user: Dict = Depends(verify_auth)):
    """Model metadata, performance metrics, and top features."""
    predictor = await asyncio.to_thread(_core()._get_predictor)
    if predictor is None:
        raise _core()._model_unavailable()

    return {
        "artifact_status": predictor.artifact_status.as_dict(),
        "model_name": predictor.metadata["model_name"],
        "version": predictor.metadata["version"],
        "is_calibrated": predictor.metadata.get("is_calibrated", False),
        "feature_count": len(predictor.feature_names),
        "training_data": predictor.metadata.get("model_card", {}).get("training_data", "Unknown"),
        "metrics": predictor.metadata.get("metrics", {}),
        "feature_importance": dict(list(
            predictor.metadata.get("feature_importance", {}).items()
        )[:15]),
    }


