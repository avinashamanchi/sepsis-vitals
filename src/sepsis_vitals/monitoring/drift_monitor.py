"""
sepsis_vitals.monitoring.drift_monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Background drift monitor using Population Stability Index (PSI).

Collects incoming prediction vitals into a rolling buffer and periodically
computes PSI against the training distribution.  Results are stored in
memory for the /metrics endpoint.

Configuration
-------------
DRIFT_CHECK_INTERVAL_SECONDS : int (default 3600)
    How often to run the PSI check.
DRIFT_BUFFER_SIZE : int (default 500)
    Maximum number of recent predictions to keep in the rolling window.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from sepsis_vitals.monitoring.metrics import check_distribution_drift, compute_psi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vitals we track for drift — the raw inputs sent by callers.
# Engineered features (deltas, roll_means) are not tracked because they are
# derived and not directly observable from a single prediction call.
# ---------------------------------------------------------------------------
_TRACKED_VITALS = [
    "temperature",
    "heart_rate",
    "resp_rate",
    "sbp",
    "dbp",
    "spo2",
    "gcs",
    "map",
    "lactate",
    "wbc",
    "procalcitonin",
]

# Medians from training (imputation_medians.json) used to build the
# reference distribution via a Gaussian approximation when no explicit
# reference arrays are available.  These values are the training-set
# medians and serve as the centre of each vital's reference distribution.
_TRAINING_MEDIANS: Dict[str, float] = {
    "temperature": 36.7,
    "heart_rate": 81.0,
    "resp_rate": 18.0,
    "sbp": 128.0,
    "dbp": 73.0,
    "spo2": 96.0,
    "gcs": 15.0,
    "map": 91.0,
    "lactate": 1.23,
    "wbc": 8.07,
    "procalcitonin": 0.08,
}

# Approximate standard deviations derived from physiological normal ranges.
# These are intentionally conservative so PSI only fires on real population
# shifts, not just natural variation around the training median.
_TRAINING_STDS: Dict[str, float] = {
    "temperature": 0.8,
    "heart_rate": 18.0,
    "resp_rate": 5.0,
    "sbp": 20.0,
    "dbp": 12.0,
    "spo2": 3.0,
    "gcs": 2.0,
    "map": 14.0,
    "lactate": 0.8,
    "wbc": 3.5,
    "procalcitonin": 0.15,
}

_REFERENCE_N = 200  # synthetic reference sample size


def _build_reference_distribution(n: int = _REFERENCE_N) -> Dict[str, list]:
    """Generate a synthetic reference sample from training distribution params.

    Each vital is sampled from a Gaussian centred on the training median with
    the approximate training standard deviation.  Values are clipped to
    physiologically plausible ranges so PSI buckets don't receive junk.
    """
    rng = np.random.default_rng(42)
    ref: Dict[str, list] = {}

    clip_bounds = {
        "temperature": (34.0, 42.0),
        "heart_rate": (30.0, 200.0),
        "resp_rate": (6.0, 50.0),
        "sbp": (60.0, 220.0),
        "dbp": (30.0, 140.0),
        "spo2": (70.0, 100.0),
        "gcs": (3.0, 15.0),
        "map": (40.0, 160.0),
        "lactate": (0.2, 20.0),
        "wbc": (0.5, 50.0),
        "procalcitonin": (0.01, 100.0),
    }

    for vital in _TRACKED_VITALS:
        mu = _TRAINING_MEDIANS.get(vital, 0.0)
        sigma = _TRAINING_STDS.get(vital, 1.0)
        samples = rng.normal(loc=mu, scale=sigma, size=n)
        lo, hi = clip_bounds.get(vital, (-np.inf, np.inf))
        samples = np.clip(samples, lo, hi)
        ref[vital] = samples.tolist()

    return ref


class DriftMonitor:
    """Background PSI drift monitor.

    Usage
    -----
    monitor = DriftMonitor()
    await monitor.start()          # kick off the background loop
    monitor.record_prediction({"heart_rate": 92, "spo2": 95, ...})
    status = monitor.get_drift_status()
    await monitor.stop()
    """

    def __init__(
        self,
        check_interval_s: Optional[int] = None,
        buffer_size: Optional[int] = None,
        reference_data: Optional[Dict[str, list]] = None,
    ) -> None:
        self._interval = int(
            check_interval_s
            if check_interval_s is not None
            else os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", 3600)
        )
        self._buffer_size = int(
            buffer_size
            if buffer_size is not None
            else os.getenv("DRIFT_BUFFER_SIZE", 500)
        )

        # Rolling buffer: one deque per vital
        self._buffers: Dict[str, Deque[float]] = {
            v: deque(maxlen=self._buffer_size) for v in _TRACKED_VITALS
        }

        # Reference distribution (training baseline)
        self._reference: Dict[str, list] = (
            reference_data if reference_data is not None else _build_reference_distribution()
        )

        # Latest drift check results — safe to read from any coroutine
        self._last_check_ts: Optional[float] = None
        self._last_results: Dict[str, Any] = {}
        self._overall_drift: bool = False

        self._task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_prediction(self, vitals: Dict[str, Any]) -> None:
        """Append a single observation to each vital's rolling buffer.

        Missing vitals in *vitals* are silently skipped so callers never
        need to fill in every field.
        """
        for vital in _TRACKED_VITALS:
            value = vitals.get(vital)
            if value is not None:
                try:
                    self._buffers[vital].append(float(value))
                except (TypeError, ValueError):
                    pass

    def get_drift_status(self) -> Dict[str, Any]:
        """Return the most recent drift check results.

        Safe to call at any time; returns an empty/pending state if no
        check has run yet.
        """
        n_buffered = {v: len(self._buffers[v]) for v in _TRACKED_VITALS}
        return {
            "last_check_timestamp": self._last_check_ts,
            "overall_drift": self._overall_drift,
            "per_vital": self._last_results,
            "buffer_counts": n_buffered,
            "check_interval_seconds": self._interval,
            "buffer_size": self._buffer_size,
        }

    async def start(self) -> None:
        """Start the periodic background check loop."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="drift_monitor")
        logger.info(
            "DriftMonitor started — checking every %ds, buffer %d observations",
            self._interval,
            self._buffer_size,
        )

    async def stop(self) -> None:
        """Signal the background loop to stop and wait for it to exit."""
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        logger.info("DriftMonitor stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Periodic check loop.  Runs the PSI computation in a thread so it
        doesn't block the event loop (numpy can be non-trivial at large N)."""
        # Stagger the first check by the full interval so we collect enough
        # data before comparing against the reference distribution.
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            return  # stop was requested before first check
        except asyncio.TimeoutError:
            pass

        while not self._stop_event.is_set():
            try:
                await asyncio.to_thread(self._run_check)
            except Exception:
                logger.exception("DriftMonitor: unexpected error during PSI check")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
                break  # stop requested
            except asyncio.TimeoutError:
                pass

    def _run_check(self) -> None:
        """Synchronous PSI check — called from a thread pool worker."""
        # Snapshot current buffers so we hold a consistent view
        current: Dict[str, list] = {}
        for vital in _TRACKED_VITALS:
            current[vital] = list(self._buffers[vital])

        # Only run the check for vitals that have enough observations
        ref_filtered: Dict[str, list] = {}
        cur_filtered: Dict[str, list] = {}
        for vital in _TRACKED_VITALS:
            if len(current[vital]) >= 10 and len(self._reference.get(vital, [])) >= 10:
                ref_filtered[vital] = self._reference[vital]
                cur_filtered[vital] = current[vital]

        if not cur_filtered:
            logger.debug("DriftMonitor: insufficient data for PSI check — skipping")
            return

        results = check_distribution_drift(ref_filtered, cur_filtered, threshold=0.2)
        overall_drift: bool = bool(results.pop("overall_drift", False))

        self._last_results = results
        self._overall_drift = overall_drift
        self._last_check_ts = time.time()

        # Emit per-vital log lines
        for vital, info in results.items():
            psi = info.get("psi", 0.0)
            if psi > 0.2:
                logger.critical(
                    "DriftMonitor: SIGNIFICANT drift detected — %s PSI=%.4f (>0.2)",
                    vital,
                    psi,
                )
            elif psi > 0.1:
                logger.warning(
                    "DriftMonitor: moderate drift detected — %s PSI=%.4f (>0.1)",
                    vital,
                    psi,
                )
            else:
                logger.debug("DriftMonitor: %s PSI=%.4f (stable)", vital, psi)

        if overall_drift:
            logger.critical(
                "DriftMonitor: overall population drift detected (at least one vital PSI>0.2)"
            )
        else:
            logger.info("DriftMonitor: PSI check complete — no significant drift detected")


# ---------------------------------------------------------------------------
# Module-level singleton — imported by api.py
# ---------------------------------------------------------------------------

_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """Return the module-level DriftMonitor singleton, creating it if needed."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor
