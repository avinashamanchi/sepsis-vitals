"""Tests for sepsis_vitals.ml.forecast."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sepsis_vitals.ml.forecast import forecast_deterioration, CRITICAL_THRESHOLD


def _series(start_risk, step, n, minutes=30):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        (base + timedelta(minutes=minutes * i), start_risk + step * i)
        for i in range(n)
    ]


def test_insufficient_history_is_low_confidence():
    fc = forecast_deterioration([(datetime.now(timezone.utc), 0.3)])
    assert fc.confidence == "low"
    assert fc.hours_to_critical is None


def test_rising_trend_projects_time_to_critical():
    # +0.05 risk every 30 min = +0.1/hour. Last point = 0.25 + 0.2 = 0.45
    # (still below the 0.70 critical line). gap 0.25 / 0.1 per hr = ~2.5 hours.
    fc = forecast_deterioration(_series(0.25, 0.05, 5))
    assert fc.trend_per_hour > 0
    assert fc.hours_to_critical is not None
    assert 1.0 < fc.hours_to_critical < 3.5
    assert fc.lead_time_band is not None
    lo, hi = fc.lead_time_band
    assert lo <= fc.hours_to_critical <= hi


def test_stable_trend_has_no_critical_eta():
    fc = forecast_deterioration(_series(0.3, 0.0, 5))
    assert fc.hours_to_critical is None
    assert "stable" in fc.horizon_label.lower() or fc.trend_per_hour <= 1e-6


def test_already_critical_returns_zero_lead_time():
    fc = forecast_deterioration(_series(0.75, 0.0, 4))
    assert fc.hours_to_critical == 0.0
    assert "critical" in fc.horizon_label.lower()


def test_projected_risk_is_clamped():
    fc = forecast_deterioration(_series(0.6, 0.3, 5))
    assert 0.0 <= fc.projected_risk_1h <= 1.0


def test_accepts_iso_strings():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    hist = [
        (base.isoformat(), 0.3),
        ((base + timedelta(hours=1)).isoformat(), 0.5),
        ((base + timedelta(hours=2)).isoformat(), 0.6),
    ]
    fc = forecast_deterioration(hist)
    assert fc.n_points == 3
    assert fc.to_dict()["trend_per_hour"] > 0
