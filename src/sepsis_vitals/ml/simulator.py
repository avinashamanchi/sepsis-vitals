"""
sepsis_vitals.ml.simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~
Simulation engine for replaying real MIMIC-IV cases and generating
synthetic wards for demos.

Both simulators feed through the real prediction pipeline:
VitalsIngester → SepsisPredictor → DeteriorationTracker → WebSocket.

The simulator controls pacing — each observation is emitted at the correct
real-time interval scaled by a speed multiplier. No independent clock;
the simulator IS the data source.

Usage::

    from sepsis_vitals.ml.simulator import CaseReplay, WardSimulator

    # Replay a single MIMIC case at 720x speed (24h → 2 min)
    replay = CaseReplay(case_meta, timeline, ingester, speed=720)
    await replay.run()

    # Simulate an 8-patient ward
    ward = WardSimulator(ingester, n_patients=8, speed=360)
    await ward.run()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─── CaseReplay ───────────────────────────────────────────────────────

class CaseReplay:
    """Replays a MIMIC-IV patient's vitals timeline at configurable speed.

    Each observation is emitted via VitalsIngester.ingest_single() at
    time-scaled intervals. At 720x speed, observations 1 hour apart
    in real time are emitted ~5 seconds apart.

    Parameters
    ----------
    case_meta : dict
        Case metadata from CaseLibrary: subject_id, stay_id, age_years,
        sex, sepsis_label, icu_los_hours.
    timeline : DataFrame
        Vitals timeline from MIMICLoader.load_vitals() for this stay.
        Columns: charttime, vital_name, valuenum.
    ingester : VitalsIngester
        The ingester to feed observations into.
    speed : float
        Speed multiplier. 720 = 24h of data replayed in 2 minutes.
    """

    def __init__(
        self,
        case_meta: Dict[str, Any],
        timeline: pd.DataFrame,
        ingester: Any,
        speed: float = 720,
    ) -> None:
        self.session_id = str(uuid.uuid4())[:8]
        self.case_meta = case_meta
        self.ingester = ingester
        self.speed = speed
        self.patient_id = f"mimic-{case_meta['subject_id']}"
        self._running = False
        self._cancelled = False
        self.position = 0

        # Pivot timeline: group by charttime, create one vitals dict per timepoint
        self._timepoints = self._pivot_timeline(timeline)
        self.total_observations = len(self._timepoints)

    @staticmethod
    def _pivot_timeline(timeline: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert long-format vitals to list of (timestamp, vitals_dict) pairs."""
        timepoints = []
        for charttime, group in timeline.groupby("charttime"):
            vitals = {}
            for _, row in group.iterrows():
                vitals[row["vital_name"]] = float(row["valuenum"])
            timepoints.append({
                "timestamp": charttime,
                "vitals": vitals,
            })
        # Sort by timestamp
        timepoints.sort(key=lambda t: t["timestamp"])
        return timepoints

    @property
    def is_complete(self) -> bool:
        """Whether all observations have been replayed."""
        return self.position >= self.total_observations

    def next_delay(self) -> float:
        """Compute the delay (in real seconds) before the next observation.

        Returns 0.0 for the first observation. For subsequent observations,
        computes the real-time gap between current and next observation,
        divided by the speed multiplier.
        """
        if self.position == 0:
            return 0.0
        if self.position >= self.total_observations:
            return 0.0

        prev_ts = self._timepoints[self.position - 1]["timestamp"]
        curr_ts = self._timepoints[self.position]["timestamp"]

        # Handle both Timestamp and string
        if isinstance(prev_ts, str):
            prev_ts = pd.Timestamp(prev_ts)
        if isinstance(curr_ts, str):
            curr_ts = pd.Timestamp(curr_ts)

        real_gap_seconds = (curr_ts - prev_ts).total_seconds()
        return max(0.0, real_gap_seconds / self.speed)

    async def step(self) -> Optional[Dict[str, Any]]:
        """Emit the next observation and advance position.

        Returns the prediction result, or None if replay is complete.
        """
        if self.is_complete:
            return None

        tp = self._timepoints[self.position]
        self.position += 1

        demographics = {
            "age": self.case_meta.get("age_years"),
            "sex": self.case_meta.get("sex"),
        }

        result = await self.ingester.ingest_single(
            self.patient_id,
            tp["vitals"],
            demographics,
        )

        return result

    async def run(self) -> None:
        """Run the full replay with time-scaled delays between observations."""
        self._running = True
        self._cancelled = False
        logger.info(
            "Starting replay session %s: patient mimic-%s at %dx speed",
            self.session_id, self.case_meta["subject_id"], self.speed,
        )

        while not self.is_complete and not self._cancelled:
            delay = self.next_delay()
            if delay > 0:
                await asyncio.sleep(delay)

            if self._cancelled:
                break

            await self.step()

        self._running = False
        logger.info("Replay session %s complete", self.session_id)

    def cancel(self) -> None:
        """Cancel the replay."""
        self._cancelled = True

    def status(self) -> Dict[str, Any]:
        """Return current replay status."""
        progress = (self.position / self.total_observations) if self.total_observations > 0 else 0.0
        return {
            "session_id": self.session_id,
            "type": "replay",
            "patient_id": self.patient_id,
            "subject_id": self.case_meta["subject_id"],
            "sepsis_label": self.case_meta.get("sepsis_label"),
            "total_observations": self.total_observations,
            "position": self.position,
            "progress": round(progress, 3),
            "speed": self.speed,
            "is_complete": self.is_complete,
            "is_running": self._running,
        }
