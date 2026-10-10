"""
sepsis_vitals.routes.metrics

Endpoints moved out of sepsis_vitals.api (behaviour unchanged). They are
declared on this module's ``router``, which ``sepsis_vitals.api`` includes;
shared state is read from the api module per request, so patching
``sepsis_vitals.api`` in tests still works.
"""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import (
    APIRouter,
    Depends,
)
from fastapi.responses import PlainTextResponse

from sepsis_vitals.dependencies import check_rate_limit, verify_auth
from sepsis_vitals.realtime.websocket import manager as ws_manager

router = APIRouter()
logger = logging.getLogger("sepsis_vitals.api")


def _core():
    """The application module, for runtime state (model, monitor, metrics).

    Looked up per request, never at import: this module does not import
    ``sepsis_vitals.api``, so it can be imported first, alone or in any order.
    """
    from sepsis_vitals import api

    return api

# ---------------------------------------------------------------------------
# Prometheus-compatible metrics
# ---------------------------------------------------------------------------

@router.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(check_rate_limit)])
async def prometheus_metrics(user: Dict = Depends(verify_auth)):
    """Prometheus-compatible metrics endpoint. Requires auth in production."""
    from sepsis_vitals.monitoring.drift_monitor import get_drift_monitor
    drift_status = get_drift_monitor().get_drift_status()

    # Build per-vital PSI lines
    drift_lines = []
    for vital, info in drift_status.get("per_vital", {}).items():
        psi_val = info.get("psi", 0.0)
        drift_lines += [
            f'sepsis_psi{{vital="{vital}"}} {psi_val:.6f}',
        ]

    lines = [
        "# HELP sepsis_requests_total Total API requests",
        "# TYPE sepsis_requests_total counter",
        f'sepsis_requests_total {_core()._metrics["requests_total"]}',
        "",
        "# HELP sepsis_predictions_total Total ML predictions made",
        "# TYPE sepsis_predictions_total counter",
        f'sepsis_predictions_total {_core()._metrics["predictions_total"]}',
        "",
        "# HELP sepsis_alerts_total Total sepsis alerts triggered",
        "# TYPE sepsis_alerts_total counter",
        f'sepsis_alerts_total {_core()._metrics["alerts_total"]}',
        "",
        "# HELP sepsis_errors_total Total API errors",
        "# TYPE sepsis_errors_total counter",
        f'sepsis_errors_total {_core()._metrics["errors_total"]}',
        "",
        "# HELP sepsis_copilot_calls_total Total AI copilot calls",
        "# TYPE sepsis_copilot_calls_total counter",
        f'sepsis_copilot_calls_total {_core()._metrics["copilot_calls_total"]}',
        "",
        "# HELP sepsis_rate_limited_total Total rate-limited requests",
        "# TYPE sepsis_rate_limited_total counter",
        f'sepsis_rate_limited_total {_core()._metrics["rate_limited_total"]}',
        "",
        "# HELP sepsis_prediction_latency_ms Average prediction latency",
        "# TYPE sepsis_prediction_latency_ms gauge",
        f'sepsis_prediction_latency_ms {_core()._metrics["avg_prediction_ms"]:.1f}',
        "",
        "# HELP sepsis_websocket_connections Active WebSocket connections",
        "# TYPE sepsis_websocket_connections gauge",
        f"sepsis_websocket_connections {ws_manager.active_connections}",
        "",
        "# HELP sepsis_model_loaded Whether the ML model is loaded",
        "# TYPE sepsis_model_loaded gauge",
        f"sepsis_model_loaded {1 if _core()._predictor is not None else 0}",
        "",
        "# HELP sepsis_drift_overall Whether overall population drift is detected (PSI>0.2)",
        "# TYPE sepsis_drift_overall gauge",
        f"sepsis_drift_overall {1 if drift_status['overall_drift'] else 0}",
        "",
        "# HELP sepsis_psi Population Stability Index per vital sign",
        "# TYPE sepsis_psi gauge",
        *drift_lines,
        "",
        "# HELP sepsis_drift_buffer_size Number of recent predictions buffered for drift detection",
        "# TYPE sepsis_drift_buffer_size gauge",
        *(
            f'sepsis_drift_buffer_size{{vital="{v}"}} {n}'
            for v, n in drift_status.get("buffer_counts", {}).items()
        ),
    ]
    return "\n".join(lines) + "\n"


