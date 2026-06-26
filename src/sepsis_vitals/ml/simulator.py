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


# ─── WardSimulator ────────────────────────────────────────────────────

class WardSimulator:
    """Generates a synthetic ward with configurable patient mix.

    Creates patients using synthetic_data.generate_patient_trajectory(),
    then replays their vitals through VitalsIngester at accelerated pace.

    Default mix: ~2 sepsis patients (1 scripted to deteriorate severely),
    ~3 sick-but-not-septic confounders, remaining stable patients.

    Parameters
    ----------
    ingester : VitalsIngester
        The ingester to feed observations into.
    n_patients : int
        Number of patients in the ward. Default: 8.
    speed : float
        Speed multiplier. 360 = 24h in 4 minutes.
    sepsis_count : int
        Number of sepsis patients. Default: 2.
    seed : int
        Random seed for reproducibility.
    obs_per_patient : int
        Observations per patient trajectory. Default: 24 (24 hours at hourly).
    """

    SICK_TYPES = ["post_surgical", "dehydration", "pain_anxiety",
                  "copd_exacerbation", "hf_exacerbation", "viral"]

    def __init__(
        self,
        ingester: Any,
        n_patients: int = 8,
        speed: float = 360,
        sepsis_count: int = 2,
        seed: int = 42,
        obs_per_patient: int = 24,
    ) -> None:
        self.session_id = str(uuid.uuid4())[:8]
        self.ingester = ingester
        self.n_patients = n_patients
        self.speed = speed
        self.sepsis_count = min(sepsis_count, n_patients)
        self.seed = seed
        self.obs_per_patient = obs_per_patient
        self._running = False
        self._cancelled = False
        self.position = 0  # global step counter

        # Generate patient configs and trajectories
        self._patient_configs: List[Dict[str, Any]] = []
        self._trajectories: List[List[Dict[str, Any]]] = []
        self.patient_ids: List[str] = []
        self._generate_ward()

        # Flatten to a time-ordered event queue: (timestamp_index, patient_idx, vitals)
        self._event_queue: List[Dict[str, Any]] = []
        self._build_event_queue()
        self.total_steps = len(self._event_queue)

    def _generate_ward(self) -> None:
        """Generate patient trajectories using synthetic_data."""
        import numpy as np
        from sepsis_vitals.ml.synthetic_data import generate_patient_trajectory

        rng = np.random.default_rng(self.seed)
        base_time = pd.Timestamp("2026-01-01 08:00:00")

        # Patient mix: sepsis_count sepsis, ~half of remaining sick, rest stable
        n_remaining = self.n_patients - self.sepsis_count
        n_sick = max(1, n_remaining // 2) if n_remaining > 1 else 0
        n_stable = n_remaining - n_sick

        configs = []

        # First sepsis patient is always severe (guaranteed escalation demo)
        for i in range(self.sepsis_count):
            severity = "severe" if i == 0 else rng.choice(["early", "subtle", "severe"])
            age = int(rng.integers(50, 85))
            sex = rng.choice(["M", "F"])
            configs.append({
                "is_septic": True,
                "sepsis_severity": severity,
                "sick_type": None,
                "age": age,
                "sex": sex,
            })

        # Sick-but-not-septic confounders
        for i in range(n_sick):
            sick_type = self.SICK_TYPES[i % len(self.SICK_TYPES)]
            age = int(rng.integers(40, 80))
            sex = rng.choice(["M", "F"])
            configs.append({
                "is_septic": False,
                "sepsis_severity": "early",
                "sick_type": sick_type,
                "age": age,
                "sex": sex,
            })

        # Stable patients
        for i in range(n_stable):
            age = int(rng.integers(30, 75))
            sex = rng.choice(["M", "F"])
            configs.append({
                "is_septic": False,
                "sepsis_severity": "early",
                "sick_type": None,
                "age": age,
                "sex": sex,
            })

        self._patient_configs = configs
        self.patient_ids = [f"ward-{self.session_id}-{i+1:02d}" for i in range(len(configs))]

        # Generate trajectories
        for idx, config in enumerate(configs):
            comorbidities = []
            if config["age"] > 60:
                comorbidities.append("hypertension")
            if config["age"] > 70:
                comorbidities.append("diabetes")

            trajectory = generate_patient_trajectory(
                rng=rng,
                patient_id=self.patient_ids[idx],
                is_septic=config["is_septic"],
                n_observations=self.obs_per_patient,
                age=config["age"],
                sex=config["sex"],
                ethnicity=rng.choice(["white", "black", "hispanic", "asian"]),
                comorbidities=comorbidities,
                base_time=base_time,
                sepsis_severity=config["sepsis_severity"],
                sick_type=config["sick_type"],
            )
            self._trajectories.append(trajectory)

    def _build_event_queue(self) -> None:
        """Build a time-ordered queue of all observations across all patients."""
        events = []
        for patient_idx, trajectory in enumerate(self._trajectories):
            for obs in trajectory:
                events.append({
                    "patient_idx": patient_idx,
                    "patient_id": self.patient_ids[patient_idx],
                    "timestamp": obs.get("timestamp", pd.Timestamp("2026-01-01")),
                    "vitals": {
                        k: v for k, v in obs.items()
                        if k in ("temperature", "heart_rate", "resp_rate", "sbp",
                                 "dbp", "spo2", "gcs", "map", "lactate", "wbc",
                                 "procalcitonin") and v is not None
                    },
                    "demographics": {
                        "age": self._patient_configs[patient_idx]["age"],
                        "sex": self._patient_configs[patient_idx]["sex"],
                    },
                })

        # Sort by timestamp
        events.sort(key=lambda e: e["timestamp"])
        self._event_queue = events

    @property
    def is_complete(self) -> bool:
        """Whether all observations have been emitted."""
        return self.position >= self.total_steps

    def next_delay(self) -> float:
        """Compute delay before next event (in real seconds)."""
        if self.position == 0 or self.position >= self.total_steps:
            return 0.0

        prev_ts = self._event_queue[self.position - 1]["timestamp"]
        curr_ts = self._event_queue[self.position]["timestamp"]

        if isinstance(prev_ts, str):
            prev_ts = pd.Timestamp(prev_ts)
        if isinstance(curr_ts, str):
            curr_ts = pd.Timestamp(curr_ts)

        real_gap = (curr_ts - prev_ts).total_seconds()
        return max(0.0, real_gap / self.speed)

    async def step(self) -> Optional[Dict[str, Any]]:
        """Emit the next observation event."""
        if self.is_complete:
            return None

        event = self._event_queue[self.position]
        self.position += 1

        result = await self.ingester.ingest_single(
            patient_id=event["patient_id"],
            vitals=event["vitals"],
            demographics=event["demographics"],
        )

        return result

    async def run(self) -> None:
        """Run the full ward simulation with time-scaled delays."""
        self._running = True
        self._cancelled = False
        logger.info(
            "Starting ward simulation %s: %d patients at %dx speed",
            self.session_id, self.n_patients, self.speed,
        )

        while not self.is_complete and not self._cancelled:
            delay = self.next_delay()
            if delay > 0:
                await asyncio.sleep(delay)
            if self._cancelled:
                break
            await self.step()

        self._running = False
        logger.info("Ward simulation %s complete", self.session_id)

    def cancel(self) -> None:
        """Cancel the simulation."""
        self._cancelled = True

    def status(self) -> Dict[str, Any]:
        """Return current ward simulation status."""
        progress = (self.position / self.total_steps) if self.total_steps > 0 else 0.0
        return {
            "session_id": self.session_id,
            "type": "ward",
            "n_patients": self.n_patients,
            "sepsis_count": self.sepsis_count,
            "patient_ids": self.patient_ids,
            "total_steps": self.total_steps,
            "position": self.position,
            "progress": round(progress, 3),
            "speed": self.speed,
            "is_complete": self.is_complete,
            "is_running": self._running,
        }
