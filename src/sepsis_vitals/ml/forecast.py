"""
sepsis_vitals.ml.forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Short-horizon deterioration forecasting.

The point-in-time predictor answers "how septic does this patient look *now*".
This module answers the question that actually drives triage: **"how much lead
time do we have before they cross into critical?"**

It is deliberately transparent -- an exponentially-weighted trend plus a linear
projection with a jack-knife confidence band -- rather than a black box.  In a
district-hospital setting a clinician has to be able to explain and override the
number, so an interpretable projection beats an opaque one even at some cost in
raw accuracy.

Inputs are a time-ordered history of ``(timestamp, risk_probability)`` points
(exactly what :class:`sepsis_vitals.ml.predictor.PatientMonitor` already
accumulates).  Nothing here touches the database or the model artifact, so it
is trivially unit-testable and safe to call on the hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Risk band cut-points, aligned with scores.classify_risk / predictor bands.
CRITICAL_THRESHOLD = 0.70
HIGH_THRESHOLD = 0.40

# EWMA smoothing factor (higher = more responsive to the latest reading).
_EWMA_ALPHA = 0.5


@dataclass
class Forecast:
    """Result of a deterioration forecast."""

    trend_per_hour: float          # slope in risk-probability units / hour
    smoothed_risk: float           # EWMA of the most recent risk values
    projected_risk_1h: float       # projected risk one hour out (clamped 0-1)
    hours_to_critical: Optional[float]  # None = not trending toward critical
    lead_time_band: Optional[Tuple[float, float]]  # (low, high) hours, 1 SD
    horizon_label: str             # human-readable summary
    confidence: str                # 'low' | 'moderate' | 'high'
    n_points: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        band = None
        if self.lead_time_band is not None:
            band = {
                "low_hours": round(self.lead_time_band[0], 2),
                "high_hours": round(self.lead_time_band[1], 2),
            }
        return {
            "trend_per_hour": round(self.trend_per_hour, 4),
            "smoothed_risk": round(self.smoothed_risk, 4),
            "projected_risk_1h": round(self.projected_risk_1h, 4),
            "hours_to_critical": (
                round(self.hours_to_critical, 2)
                if self.hours_to_critical is not None
                else None
            ),
            "lead_time_band": band,
            "horizon_label": self.horizon_label,
            "confidence": self.confidence,
            "n_points": self.n_points,
            "details": self.details,
        }


def _to_epoch_hours(ts: Any) -> float:
    """Convert an ISO string or datetime to epoch *hours* (float)."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp() / 3600.0
    return float(ts)


def _ewma(values: Sequence[float], alpha: float = _EWMA_ALPHA) -> float:
    """Exponentially-weighted moving average, most-recent weighted highest."""
    if not values:
        return 0.0
    acc = values[0]
    for v in values[1:]:
        acc = alpha * v + (1 - alpha) * acc
    return acc


def _ols(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Ordinary least-squares slope & intercept for y = slope*x + intercept."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _slope_std_error(
    xs: Sequence[float], ys: Sequence[float], slope: float, intercept: float
) -> float:
    """Standard error of the OLS slope (used for the lead-time band)."""
    n = len(xs)
    if n <= 2:
        return 0.0
    mean_x = sum(xs) / n
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    if ss_x == 0:
        return 0.0
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    dof = n - 2
    resid_var = sum(r ** 2 for r in residuals) / dof
    return math.sqrt(resid_var / ss_x)


def forecast_deterioration(
    history: Sequence[Tuple[Any, float]],
    *,
    critical_threshold: float = CRITICAL_THRESHOLD,
    max_lookback: int = 8,
) -> Forecast:
    """Project a patient's sepsis-risk trajectory forward.

    Parameters
    ----------
    history:
        Time-ordered sequence of ``(timestamp, risk_probability)`` where
        timestamp is an ISO string, ``datetime``, or epoch seconds.  At least
        two points are required for a trend; fewer yields a low-confidence,
        flat forecast.
    critical_threshold:
        Risk probability that defines the "critical" band.
    max_lookback:
        Only the most recent *max_lookback* points inform the slope, so the
        forecast tracks the current trajectory rather than stale history.

    Returns
    -------
    Forecast
    """
    pts = [(_to_epoch_hours(ts), float(r)) for ts, r in history]
    pts.sort(key=lambda p: p[0])
    if max_lookback > 0:
        pts = pts[-max_lookback:]

    n = len(pts)
    risks = [r for _, r in pts]
    smoothed = _ewma(risks)

    if n < 2:
        current = risks[-1] if risks else 0.0
        return Forecast(
            trend_per_hour=0.0,
            smoothed_risk=smoothed,
            projected_risk_1h=current,
            hours_to_critical=None,
            lead_time_band=None,
            horizon_label="Insufficient history for a trend",
            confidence="low",
            n_points=n,
        )

    xs = [x for x, _ in pts]
    ys = risks
    slope, intercept = _ols(xs, ys)          # risk units per hour
    se = _slope_std_error(xs, ys, slope, intercept)

    now_x = xs[-1]
    current_risk = ys[-1]
    projected_1h = _clamp(current_risk + slope * 1.0)

    hours_to_critical: Optional[float] = None
    band: Optional[Tuple[float, float]] = None

    if current_risk >= critical_threshold:
        hours_to_critical = 0.0
        band = (0.0, 0.0)
    elif slope > 1e-6:
        gap = critical_threshold - current_risk
        hours_to_critical = gap / slope
        # Lead-time uncertainty from the slope's standard error.
        if se > 0:
            lo_slope = max(slope - se, 1e-6)
            hi_slope = slope + se
            band = (gap / hi_slope, gap / lo_slope)

    confidence = _confidence(n, se, slope)
    label = _label(current_risk, slope, hours_to_critical, critical_threshold)

    return Forecast(
        trend_per_hour=slope,
        smoothed_risk=smoothed,
        projected_risk_1h=projected_1h,
        hours_to_critical=hours_to_critical,
        lead_time_band=band,
        horizon_label=label,
        confidence=confidence,
        n_points=n,
        details={
            "slope_std_error": round(se, 5),
            "current_risk": round(current_risk, 4),
            "critical_threshold": critical_threshold,
            "lookback_points": n,
        },
    )


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _confidence(n: int, se: float, slope: float) -> str:
    if n < 3:
        return "low"
    # Relative error of the slope estimate.
    if abs(slope) < 1e-6:
        return "moderate" if n >= 4 else "low"
    rel = se / abs(slope)
    if rel < 0.4 and n >= 4:
        return "high"
    if rel < 1.0:
        return "moderate"
    return "low"


def _label(
    current: float,
    slope: float,
    hours_to_critical: Optional[float],
    threshold: float,
) -> str:
    if current >= threshold:
        return "Already in critical range"
    if slope <= 1e-6:
        return "Stable or improving trajectory"
    if hours_to_critical is None:
        return "Rising, but timing uncertain"
    if hours_to_critical < 1:
        return "Projected to reach critical within the hour"
    if hours_to_critical < 4:
        return f"Projected to reach critical in ~{hours_to_critical:.0f}h"
    return f"Slowly rising; critical in ~{hours_to_critical:.0f}h if trend holds"
