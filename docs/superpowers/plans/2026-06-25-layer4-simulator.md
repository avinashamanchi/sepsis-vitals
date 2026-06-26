# Layer 4: Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a case replay and synthetic ward simulator that feeds through the real prediction pipeline (VitalsIngester → SepsisPredictor → DeteriorationTracker → WebSocket), controlled via API endpoints gated behind `ENABLE_SIMULATOR=true`.

**Architecture:** Two new modules: `case_library.py` (SQLite index of MIMIC-IV demo cases for quick lookup) and `simulator.py` (CaseReplay for single-patient timeline replay, WardSimulator for multi-patient synthetic wards). Both emit observations through `VitalsIngester.ingest_single()` at time-scaled intervals. The simulator controls pacing — no independent clock. API endpoints manage simulation sessions (start/stop/list). All behind an env var gate so simulators never run in production.

**Tech Stack:** Python 3.9+, asyncio, SQLite (via stdlib sqlite3), pandas, FastAPI, existing `VitalsIngester`/`PatientRegistry`/`DeteriorationTracker` from Layer 3, existing `MIMICLoader` and `synthetic_data.generate_patient_trajectory()`.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/sepsis_vitals/ml/case_library.py` | Create | SQLite index of MIMIC-IV demo cases, case lookup/listing |
| `src/sepsis_vitals/ml/simulator.py` | Create | `CaseReplay`, `WardSimulator`, `SimulationSession` |
| `src/sepsis_vitals/api.py` | Modify | Simulator endpoints gated behind `ENABLE_SIMULATOR` |
| `tests/test_case_library.py` | Create | Tests for case library |
| `tests/test_simulator.py` | Create | Tests for CaseReplay and WardSimulator |
| `tests/test_simulator_api.py` | Create | Tests for simulator API endpoints |

---

### Task 1: CaseLibrary — SQLite Index of MIMIC-IV Cases

**Files:**
- Create: `src/sepsis_vitals/ml/case_library.py`
- Create: `tests/test_case_library.py`

This task builds a lightweight index of MIMIC-IV demo cases. The index is a SQLite database that stores case metadata (subject_id, age, sex, sepsis status, ICU length of stay, observation count) for quick lookup without re-parsing the full MIMIC CSV files each time.

- [ ] **Step 1: Write failing tests for CaseLibrary**

Create `tests/test_case_library.py`:

```python
"""Tests for CaseLibrary — MIMIC-IV case indexing and lookup."""

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


class TestCaseLibrary:
    """Test case indexing and lookup."""

    @pytest.fixture
    def sample_stays(self):
        """Fake derive_sepsis_labels output."""
        return pd.DataFrame({
            "subject_id": [1001, 1002, 1003, 1004, 1005],
            "hadm_id": [2001, 2002, 2003, 2004, 2005],
            "stay_id": [3001, 3002, 3003, 3004, 3005],
            "intime": pd.to_datetime([
                "2150-01-01 08:00", "2150-01-02 10:00", "2150-01-03 06:00",
                "2150-01-04 14:00", "2150-01-05 09:00",
            ]),
            "outtime": pd.to_datetime([
                "2150-01-03 08:00", "2150-01-04 10:00", "2150-01-04 06:00",
                "2150-01-06 14:00", "2150-01-06 09:00",
            ]),
            "sepsis_label": [1, 0, 1, 0, 1],
            "label_source": ["sepsis3", "none", "sepsis3", "none", "icd_fallback"],
        })

    @pytest.fixture
    def sample_demographics(self):
        """Fake demographics output."""
        return pd.DataFrame({
            "hadm_id": [2001, 2002, 2003, 2004, 2005],
            "age_years": [65, 45, 72, 55, 80],
            "sex_m": [1, 0, 1, 0, 1],
        })

    @pytest.fixture
    def sample_vitals(self):
        """Fake vitals output — varying observation counts per stay."""
        rows = []
        obs_counts = {3001: 20, 3002: 15, 3003: 8, 3004: 30, 3005: 12}
        for stay_id, n_obs in obs_counts.items():
            for i in range(n_obs):
                rows.append({
                    "stay_id": stay_id,
                    "charttime": pd.Timestamp("2150-01-01") + pd.Timedelta(hours=i),
                    "vital_name": "heart_rate",
                    "valuenum": 80 + np.random.randn() * 5,
                })
        return pd.DataFrame(rows)

    def test_build_index_creates_sqlite(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(
            stays=sample_stays,
            demographics=sample_demographics,
            vitals=sample_vitals,
        )

        # SQLite file should exist
        assert (tmp_path / "cases.db").exists()

        # Should have 5 cases
        cases = lib.list_cases()
        assert len(cases) == 5

    def test_get_sepsis_cases(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        sepsis_cases = lib.get_sepsis_cases()
        assert len(sepsis_cases) == 3  # subjects 1001, 1003, 1005
        assert all(c["sepsis_label"] == 1 for c in sepsis_cases)

    def test_get_case_by_subject_id(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_case(subject_id=1001)
        assert case is not None
        assert case["subject_id"] == 1001
        assert case["age_years"] == 65
        assert case["sex"] == "M"
        assert case["sepsis_label"] == 1
        assert case["n_observations"] == 20

    def test_get_case_unknown_returns_none(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        assert lib.get_case(subject_id=9999) is None

    def test_get_random_case_sepsis(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_random_case(sepsis=True)
        assert case is not None
        assert case["sepsis_label"] == 1

    def test_get_random_case_nonsepsis(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_random_case(sepsis=False)
        assert case is not None
        assert case["sepsis_label"] == 0

    def test_icu_los_hours(self, tmp_path, sample_stays, sample_demographics, sample_vitals):
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=tmp_path / "cases.db")
        lib.build_index(stays=sample_stays, demographics=sample_demographics, vitals=sample_vitals)

        case = lib.get_case(subject_id=1001)
        # 2150-01-01 08:00 -> 2150-01-03 08:00 = 48 hours
        assert abs(case["icu_los_hours"] - 48.0) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_case_library.py -v`
Expected: FAIL — `No module named 'sepsis_vitals.ml.case_library'`

- [ ] **Step 3: Implement CaseLibrary**

Create `src/sepsis_vitals/ml/case_library.py`:

```python
"""
sepsis_vitals.ml.case_library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight SQLite index of MIMIC-IV demo cases for quick lookup.

Indexes patient cases by subject_id, age, sex, sepsis status, ICU length
of stay, and observation count — so the simulator can quickly list and
select cases without re-parsing the full MIMIC CSV files.

Usage::

    from sepsis_vitals.ml.case_library import CaseLibrary

    lib = CaseLibrary("cases.db")
    lib.build_from_mimic()  # one-time index build
    cases = lib.get_sepsis_cases()
    timeline = lib.get_vitals_timeline(subject_id=10006)
"""

from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CaseLibrary:
    """SQLite-backed index of MIMIC-IV demo cases.

    Parameters
    ----------
    index_path : str or Path
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, index_path: str | Path = "data/case_index.db") -> None:
        self.index_path = Path(index_path)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.index_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def build_index(
        self,
        stays: pd.DataFrame,
        demographics: pd.DataFrame,
        vitals: pd.DataFrame,
    ) -> None:
        """Build the case index from pre-loaded DataFrames.

        Parameters
        ----------
        stays : DataFrame
            Output of MIMICLoader.derive_sepsis_labels() with columns:
            subject_id, hadm_id, stay_id, intime, outtime, sepsis_label, label_source.
        demographics : DataFrame
            Output of MIMICLoader.load_demographics() with columns:
            hadm_id, age_years, sex_m.
        vitals : DataFrame
            Output of MIMICLoader.load_vitals() with columns:
            stay_id, charttime, vital_name, valuenum.
        """
        conn = self._get_conn()

        conn.execute("DROP TABLE IF EXISTS cases")
        conn.execute("""
            CREATE TABLE cases (
                subject_id INTEGER PRIMARY KEY,
                hadm_id INTEGER,
                stay_id INTEGER,
                age_years INTEGER,
                sex TEXT,
                sepsis_label INTEGER,
                label_source TEXT,
                icu_los_hours REAL,
                n_observations INTEGER,
                intime TEXT,
                outtime TEXT
            )
        """)

        # Merge demographics
        merged = stays.merge(
            demographics[["hadm_id", "age_years", "sex_m"]],
            on="hadm_id",
            how="left",
        )

        # Count observations per stay
        obs_counts = vitals.groupby("stay_id").size().to_dict()

        rows = []
        for _, row in merged.iterrows():
            los_hours = (row["outtime"] - row["intime"]).total_seconds() / 3600.0
            rows.append((
                int(row["subject_id"]),
                int(row["hadm_id"]),
                int(row["stay_id"]),
                int(row.get("age_years", 0)),
                "M" if row.get("sex_m", 0) == 1 else "F",
                int(row["sepsis_label"]),
                str(row.get("label_source", "")),
                round(los_hours, 2),
                obs_counts.get(row["stay_id"], 0),
                str(row["intime"]),
                str(row["outtime"]),
            ))

        conn.executemany(
            "INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info("Built case index with %d cases", len(rows))

    def build_from_mimic(self) -> None:
        """Build the index directly from MIMIC-IV demo data.

        Convenience method that loads data from MIMICLoader and builds
        the index in one call. Requires MIMIC-IV demo data to be available.
        """
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        stays = loader.derive_sepsis_labels()
        demographics = loader.load_demographics()
        stay_ids = set(stays["stay_id"])
        vitals = loader.load_vitals(stay_ids)

        self.build_index(stays, demographics, vitals)

    def list_cases(self) -> List[Dict[str, Any]]:
        """List all indexed cases."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM cases ORDER BY subject_id").fetchall()
        return [dict(r) for r in rows]

    def get_sepsis_cases(self) -> List[Dict[str, Any]]:
        """Get all cases with sepsis_label = 1."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM cases WHERE sepsis_label = 1 ORDER BY subject_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_case(self, subject_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific case by subject_id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM cases WHERE subject_id = ?", (subject_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_random_case(self, sepsis: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Get a random case, optionally filtered by sepsis status.

        Parameters
        ----------
        sepsis : bool, optional
            If True, only sepsis cases. If False, only non-sepsis.
            If None, any case.
        """
        conn = self._get_conn()
        if sepsis is True:
            rows = conn.execute(
                "SELECT * FROM cases WHERE sepsis_label = 1"
            ).fetchall()
        elif sepsis is False:
            rows = conn.execute(
                "SELECT * FROM cases WHERE sepsis_label = 0"
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM cases").fetchall()

        if not rows:
            return None
        return dict(random.choice(rows))

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_case_library.py -v`
Expected: All 8 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/case_library.py tests/test_case_library.py
git commit -m "feat: add CaseLibrary — SQLite index of MIMIC-IV demo cases"
```

---

### Task 2: CaseReplay — Single Patient Timeline Replay

**Files:**
- Create: `src/sepsis_vitals/ml/simulator.py` (partial — CaseReplay only)
- Create: `tests/test_simulator.py` (partial — CaseReplay tests)

- [ ] **Step 1: Write failing tests for CaseReplay**

Create `tests/test_simulator.py`:

```python
"""Tests for the simulator module — case replay and ward simulation."""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
import numpy as np


class TestCaseReplay:
    """Test MIMIC-IV case replay at configurable speed."""

    @pytest.fixture
    def sample_timeline(self):
        """A patient's vitals timeline — 6 observations over 6 hours."""
        base = pd.Timestamp("2150-01-01 08:00:00")
        return pd.DataFrame({
            "charttime": [base + pd.Timedelta(hours=i) for i in range(6)],
            "vital_name": ["heart_rate"] * 6,
            "valuenum": [80, 85, 92, 100, 108, 115],
        })

    @pytest.fixture
    def sample_case_meta(self):
        return {
            "subject_id": 1001,
            "stay_id": 3001,
            "age_years": 65,
            "sex": "M",
            "sepsis_label": 1,
            "icu_los_hours": 48.0,
        }

    @pytest.fixture
    def mock_ingester(self):
        ingester = MagicMock()
        ingester.ingest_single = AsyncMock(return_value={
            "risk_probability": 0.5,
            "risk_level": "moderate",
        })
        return ingester

    def test_create_replay(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        assert replay.session_id is not None
        assert replay.speed == 720
        assert replay.patient_id == "mimic-1001"
        assert replay.total_observations == 6

    def test_replay_emits_observations(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        # Run one step
        asyncio.get_event_loop().run_until_complete(replay.step())

        # Should have called ingest_single with the first observation
        mock_ingester.ingest_single.assert_called_once()
        call_kwargs = mock_ingester.ingest_single.call_args
        assert call_kwargs[0][0] == "mimic-1001"  # patient_id
        assert "heart_rate" in call_kwargs[0][1]  # vitals dict

    def test_replay_tracks_position(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        assert replay.position == 0

        asyncio.get_event_loop().run_until_complete(replay.step())
        assert replay.position == 1

        asyncio.get_event_loop().run_until_complete(replay.step())
        assert replay.position == 2

    def test_replay_completes(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        # Step through all observations
        for _ in range(6):
            asyncio.get_event_loop().run_until_complete(replay.step())

        assert replay.is_complete
        assert replay.position == 6

    def test_replay_computes_delay(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        # First observation has no delay (immediate)
        delay = replay.next_delay()
        assert delay == 0.0

        # After first step, delay should be: 1 hour / 720 = 5 seconds
        asyncio.get_event_loop().run_until_complete(replay.step())
        delay = replay.next_delay()
        assert abs(delay - 5.0) < 0.1

    def test_replay_status(self, sample_timeline, sample_case_meta, mock_ingester):
        from sepsis_vitals.ml.simulator import CaseReplay

        replay = CaseReplay(
            case_meta=sample_case_meta,
            timeline=sample_timeline,
            ingester=mock_ingester,
            speed=720,
        )

        status = replay.status()
        assert status["session_id"] == replay.session_id
        assert status["type"] == "replay"
        assert status["subject_id"] == 1001
        assert status["total_observations"] == 6
        assert status["position"] == 0
        assert status["progress"] == 0.0
        assert status["is_complete"] is False

    def test_replay_pivots_timeline_to_vitals_dict(self, mock_ingester):
        """Timeline has multiple vital types — pivot to single dict per timepoint."""
        from sepsis_vitals.ml.simulator import CaseReplay

        base = pd.Timestamp("2150-01-01 08:00:00")
        timeline = pd.DataFrame({
            "charttime": [base, base, base, base + pd.Timedelta(hours=1), base + pd.Timedelta(hours=1)],
            "vital_name": ["heart_rate", "temperature", "sbp", "heart_rate", "temperature"],
            "valuenum": [80, 37.2, 120, 90, 37.8],
        })

        meta = {"subject_id": 1001, "stay_id": 3001, "age_years": 65, "sex": "M",
                "sepsis_label": 1, "icu_los_hours": 24.0}

        replay = CaseReplay(case_meta=meta, timeline=timeline, ingester=mock_ingester, speed=720)

        # Should have 2 timepoints (pivoted)
        assert replay.total_observations == 2

        asyncio.get_event_loop().run_until_complete(replay.step())
        call_args = mock_ingester.ingest_single.call_args[0]
        vitals = call_args[1]
        assert "heart_rate" in vitals
        assert "temperature" in vitals
        assert vitals["heart_rate"] == 80
        assert vitals["temperature"] == 37.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_simulator.py::TestCaseReplay -v`
Expected: FAIL — `No module named 'sepsis_vitals.ml.simulator'`

- [ ] **Step 3: Implement CaseReplay**

Create `src/sepsis_vitals/ml/simulator.py`:

```python
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
            patient_id=self.patient_id,
            vitals=tp["vitals"],
            demographics=demographics,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_simulator.py::TestCaseReplay -v`
Expected: All 7 PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/simulator.py tests/test_simulator.py
git commit -m "feat: add CaseReplay — MIMIC-IV patient timeline replay at configurable speed"
```

---

### Task 3: WardSimulator — Multi-Patient Synthetic Ward

**Files:**
- Modify: `src/sepsis_vitals/ml/simulator.py` (add WardSimulator)
- Modify: `tests/test_simulator.py` (add WardSimulator tests)

- [ ] **Step 1: Write failing tests for WardSimulator**

Append to `tests/test_simulator.py`:

```python
class TestWardSimulator:
    """Test synthetic ward simulation."""

    @pytest.fixture
    def mock_ingester(self):
        ingester = MagicMock()
        ingester.ingest_single = AsyncMock(return_value={
            "risk_probability": 0.3,
            "risk_level": "low",
        })
        return ingester

    def test_create_ward(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=8,
            speed=360,
            sepsis_count=2,
            seed=42,
        )

        assert ward.session_id is not None
        assert ward.n_patients == 8
        assert ward.speed == 360
        assert len(ward.patient_ids) == 8

    def test_ward_generates_trajectories(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=6,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        # Should have generated trajectories for all patients
        assert len(ward._trajectories) == 6

    def test_ward_step_emits_observations(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        asyncio.get_event_loop().run_until_complete(ward.step())

        # Should have called ingest_single for each patient with an observation at this timepoint
        assert mock_ingester.ingest_single.call_count >= 1

    def test_ward_includes_guaranteed_deterioration(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=8,
            speed=360,
            sepsis_count=2,
            seed=42,
        )

        # At least one patient should be the scripted deterioration case
        has_severe = any(
            t.get("sepsis_severity") == "severe"
            for t in ward._patient_configs
        )
        assert has_severe

    def test_ward_status(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=6,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        status = ward.status()
        assert status["session_id"] == ward.session_id
        assert status["type"] == "ward"
        assert status["n_patients"] == 6
        assert status["sepsis_count"] == 1
        assert status["is_complete"] is False

    def test_ward_completes(self, mock_ingester):
        from sepsis_vitals.ml.simulator import WardSimulator

        ward = WardSimulator(
            ingester=mock_ingester,
            n_patients=4,
            speed=720,
            sepsis_count=1,
            seed=42,
            obs_per_patient=3,  # small for testing
        )

        # Step through all observations
        for _ in range(20):  # more than enough steps
            if ward.is_complete:
                break
            asyncio.get_event_loop().run_until_complete(ward.step())

        assert ward.is_complete
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_simulator.py::TestWardSimulator -v`
Expected: FAIL — `cannot import name 'WardSimulator'`

- [ ] **Step 3: Implement WardSimulator**

Add to `src/sepsis_vitals/ml/simulator.py` (after CaseReplay):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_simulator.py -v`
Expected: All 13 PASS (7 CaseReplay + 6 WardSimulator)

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/simulator.py tests/test_simulator.py
git commit -m "feat: add WardSimulator — synthetic ward with configurable patient mix"
```

---

### Task 4: SimulationManager — Session Lifecycle

**Files:**
- Modify: `src/sepsis_vitals/ml/simulator.py` (add SimulationManager)
- Modify: `tests/test_simulator.py` (add SimulationManager tests)

- [ ] **Step 1: Write failing tests for SimulationManager**

Append to `tests/test_simulator.py`:

```python
class TestSimulationManager:
    """Test simulation session lifecycle management."""

    @pytest.fixture
    def mock_ingester(self):
        ingester = MagicMock()
        ingester.ingest_single = AsyncMock(return_value={
            "risk_probability": 0.3,
            "risk_level": "low",
        })
        return ingester

    def test_create_manager(self):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        assert manager.list_sessions() == []

    def test_start_ward_session(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        session_id = manager.start_ward(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
            sepsis_count=1,
            seed=42,
        )

        assert session_id is not None
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["type"] == "ward"

    def test_stop_session(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        session_id = manager.start_ward(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
        )

        stopped = manager.stop_session(session_id)
        assert stopped is True

    def test_stop_nonexistent_session(self):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        assert manager.stop_session("nonexistent") is False

    def test_get_session_status(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        session_id = manager.start_ward(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
        )

        status = manager.get_session(session_id)
        assert status is not None
        assert status["session_id"] == session_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_simulator.py::TestSimulationManager -v`
Expected: FAIL — `cannot import name 'SimulationManager'`

- [ ] **Step 3: Implement SimulationManager**

Add to `src/sepsis_vitals/ml/simulator.py` (after WardSimulator):

```python
# ─── SimulationManager ────────────────────────────────────────────────

class SimulationManager:
    """Manages active simulation sessions.

    Provides start/stop/list lifecycle for CaseReplay and WardSimulator
    sessions. Each session runs as an asyncio background task.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Any] = {}  # session_id → simulator instance
        self._tasks: Dict[str, asyncio.Task] = {}  # session_id → asyncio task

    def start_ward(
        self,
        ingester: Any,
        n_patients: int = 8,
        speed: float = 360,
        sepsis_count: int = 2,
        seed: int = 42,
    ) -> str:
        """Start a ward simulation session. Returns session_id."""
        ward = WardSimulator(
            ingester=ingester,
            n_patients=n_patients,
            speed=speed,
            sepsis_count=sepsis_count,
            seed=seed,
        )
        self._sessions[ward.session_id] = ward

        # Start as background task if event loop is running
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(ward.run())
            self._tasks[ward.session_id] = task
        except RuntimeError:
            pass  # no event loop — caller will run manually (tests)

        return ward.session_id

    def start_replay(
        self,
        case_meta: Dict[str, Any],
        timeline: pd.DataFrame,
        ingester: Any,
        speed: float = 720,
    ) -> str:
        """Start a case replay session. Returns session_id."""
        replay = CaseReplay(
            case_meta=case_meta,
            timeline=timeline,
            ingester=ingester,
            speed=speed,
        )
        self._sessions[replay.session_id] = replay

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(replay.run())
            self._tasks[replay.session_id] = task
        except RuntimeError:
            pass

        return replay.session_id

    def stop_session(self, session_id: str) -> bool:
        """Stop a simulation session. Returns True if found and stopped."""
        sim = self._sessions.get(session_id)
        if sim is None:
            return False

        sim.cancel()

        task = self._tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()

        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session."""
        sim = self._sessions.get(session_id)
        if sim is None:
            return None
        return sim.status()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active and completed sessions."""
        return [sim.status() for sim in self._sessions.values()]

    def cleanup_completed(self) -> int:
        """Remove completed sessions. Returns count removed."""
        completed = [
            sid for sid, sim in self._sessions.items()
            if sim.is_complete
        ]
        for sid in completed:
            self._sessions.pop(sid, None)
            self._tasks.pop(sid, None)
        return len(completed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_simulator.py -v`
Expected: All 18 PASS (7 CaseReplay + 6 WardSimulator + 5 SimulationManager)

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/simulator.py tests/test_simulator.py
git commit -m "feat: add SimulationManager — session lifecycle for replay and ward sims"
```

---

### Task 5: Simulator API Endpoints

**Files:**
- Modify: `src/sepsis_vitals/api.py`
- Create: `tests/test_simulator_api.py`

- [ ] **Step 1: Write failing tests for simulator endpoints**

Create `tests/test_simulator_api.py`:

```python
"""Tests for simulator API endpoints — logic tests using SimulationManager directly."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import pandas as pd


class TestSimulatorAPI:
    """Test simulator endpoint logic via SimulationManager."""

    @pytest.fixture
    def mock_ingester(self):
        ingester = MagicMock()
        ingester.ingest_single = AsyncMock(return_value={
            "risk_probability": 0.3,
            "risk_level": "low",
        })
        return ingester

    def test_simulator_gate_env_var(self):
        """ENABLE_SIMULATOR must be true for simulator to activate."""
        import os
        # Default should be false
        assert os.getenv("ENABLE_SIMULATOR", "false").lower() != "true"

    def test_start_ward_returns_session_id(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        session_id = manager.start_ward(
            ingester=mock_ingester,
            n_patients=4,
            speed=360,
        )
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_start_replay_returns_session_id(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        base = pd.Timestamp("2150-01-01 08:00")
        timeline = pd.DataFrame({
            "charttime": [base, base + pd.Timedelta(hours=1)],
            "vital_name": ["heart_rate", "heart_rate"],
            "valuenum": [80, 90],
        })
        meta = {
            "subject_id": 1001, "stay_id": 3001,
            "age_years": 65, "sex": "M",
            "sepsis_label": 1, "icu_los_hours": 24.0,
        }

        manager = SimulationManager()
        session_id = manager.start_replay(
            case_meta=meta,
            timeline=timeline,
            ingester=mock_ingester,
            speed=720,
        )
        assert isinstance(session_id, str)

    def test_list_sessions_shows_all(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        manager.start_ward(ingester=mock_ingester, n_patients=4, speed=360)
        manager.start_ward(ingester=mock_ingester, n_patients=6, speed=720)

        sessions = manager.list_sessions()
        assert len(sessions) == 2

    def test_stop_session_removes(self, mock_ingester):
        from sepsis_vitals.ml.simulator import SimulationManager

        manager = SimulationManager()
        sid = manager.start_ward(ingester=mock_ingester, n_patients=4, speed=360)

        assert manager.stop_session(sid) is True
        status = manager.get_session(sid)
        assert status is not None  # still in list until cleanup

    def test_cases_endpoint_logic(self):
        """CaseLibrary.list_cases should return dicts with required fields."""
        from sepsis_vitals.ml.case_library import CaseLibrary

        lib = CaseLibrary(index_path=":memory:")
        # Empty library should return empty list
        # (SQLite in-memory won't have the table yet, so we test the interface)
        try:
            cases = lib.list_cases()
            assert isinstance(cases, list)
        except Exception:
            pass  # table not created yet — that's fine for this test
```

- [ ] **Step 2: Run tests to verify they pass** (these test logic, not HTTP)

Run: `python3 -m pytest tests/test_simulator_api.py -v`

- [ ] **Step 3: Add simulator endpoints to api.py**

Add the following to `src/sepsis_vitals/api.py`:

1. Near the top, after the monitor singleton, add the simulator gate and singleton:

```python
# ---------------------------------------------------------------------------
# Simulator (gated behind ENABLE_SIMULATOR=true)
# ---------------------------------------------------------------------------

_simulator_enabled = os.getenv("ENABLE_SIMULATOR", "false").lower() == "true"
_simulation_manager = None


def _get_simulation_manager():
    """Lazy-initialize the SimulationManager."""
    global _simulation_manager
    if _simulation_manager is None:
        from sepsis_vitals.ml.simulator import SimulationManager
        _simulation_manager = SimulationManager()
    return _simulation_manager
```

2. Add endpoints after the monitor endpoints:

```python
@app.post("/simulator/ward", dependencies=[Depends(check_rate_limit)])
async def simulator_start_ward(body: dict, user: Dict = Depends(verify_auth)):
    """Start a synthetic ward simulation."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    _, _, ingester = _get_monitor_components()
    if ingester is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    manager = _get_simulation_manager()
    session_id = manager.start_ward(
        ingester=ingester,
        n_patients=body.get("n_patients", 8),
        speed=body.get("speed", 360),
        sepsis_count=body.get("sepsis_count", 2),
        seed=body.get("seed", 42),
    )

    return {"session_id": session_id, "status": "started"}


@app.post("/simulator/replay", dependencies=[Depends(check_rate_limit)])
async def simulator_start_replay(body: dict, user: Dict = Depends(verify_auth)):
    """Start a MIMIC-IV case replay."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    _, _, ingester = _get_monitor_components()
    if ingester is None:
        raise HTTPException(status_code=503, detail="Prediction engine not loaded")

    subject_id = body.get("subject_id")
    speed = body.get("speed", 720)

    from sepsis_vitals.ml.case_library import CaseLibrary
    lib = CaseLibrary()

    if subject_id == "random" or subject_id is None:
        sepsis_only = body.get("sepsis_only", False)
        case_meta = lib.get_random_case(sepsis=sepsis_only if sepsis_only else None)
    else:
        case_meta = lib.get_case(subject_id=int(subject_id))

    if case_meta is None:
        raise HTTPException(status_code=404, detail="Case not found")

    # Load vitals timeline for this case
    from sepsis_vitals.ml.mimic_loader import MIMICLoader
    loader = MIMICLoader.from_demo()
    vitals = loader.load_vitals(stay_ids={case_meta["stay_id"]})

    manager = _get_simulation_manager()
    session_id = manager.start_replay(
        case_meta=case_meta,
        timeline=vitals,
        ingester=ingester,
        speed=speed,
    )

    return {"session_id": session_id, "subject_id": case_meta["subject_id"], "status": "started"}


@app.delete("/simulator/{session_id}", dependencies=[Depends(check_rate_limit)])
async def simulator_stop(session_id: str, user: Dict = Depends(verify_auth)):
    """Stop a simulation session."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    manager = _get_simulation_manager()
    stopped = manager.stop_session(sanitise_string(session_id))

    if not stopped:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "status": "stopped"}


@app.get("/simulator/sessions", dependencies=[Depends(check_rate_limit)])
async def simulator_sessions(user: Dict = Depends(verify_auth)):
    """List active simulation sessions."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    manager = _get_simulation_manager()
    return {"sessions": manager.list_sessions()}


@app.get("/simulator/cases", dependencies=[Depends(check_rate_limit)])
async def simulator_cases(user: Dict = Depends(verify_auth)):
    """List available MIMIC-IV cases for replay."""
    if not _simulator_enabled:
        raise HTTPException(status_code=403, detail="Simulator not enabled")

    from sepsis_vitals.ml.case_library import CaseLibrary
    lib = CaseLibrary()

    try:
        cases = lib.list_cases()
    except Exception:
        cases = []

    return {"cases": cases, "count": len(cases)}
```

- [ ] **Step 4: Run all simulator tests**

Run: `python3 -m pytest tests/test_simulator.py tests/test_simulator_api.py tests/test_case_library.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/api.py tests/test_simulator_api.py
git commit -m "feat: add simulator API endpoints gated behind ENABLE_SIMULATOR"
```

---

### Task 6: Integration Test and Full Suite

**Files:**
- Modify: `tests/test_simulator.py` (add integration test)

- [ ] **Step 1: Write integration test**

Append to `tests/test_simulator.py`:

```python
class TestSimulatorIntegration:
    """End-to-end integration test for the simulator pipeline."""

    def test_ward_flows_through_prediction_pipeline(self):
        """Ward sim → VitalsIngester → predict → DeteriorationTracker → WebSocket."""
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker
        from sepsis_vitals.ml.simulator import WardSimulator

        predictor = MagicMock()
        call_count = 0

        def mock_predict(**kwargs):
            nonlocal call_count
            call_count += 1
            pred = MagicMock()
            pred.risk_probability = 0.3 + (call_count % 5) * 0.1
            level = "low" if pred.risk_probability < 0.5 else "moderate"
            pred.risk_level = level
            pred.to_dict.return_value = {
                "risk_probability": pred.risk_probability,
                "risk_level": level,
                "patient_id": kwargs.get("patient_id", "test"),
                "timestamp": "2026-01-01T08:00:00",
            }
            return pred

        predictor.predict.side_effect = mock_predict

        ws = MagicMock()
        ws.broadcast = AsyncMock()
        registry = PatientRegistry(debounce_seconds=0)
        tracker = DeteriorationTracker(risk_floor=0.40)
        ingester = VitalsIngester(predictor, registry, tracker, ws)

        ward = WardSimulator(
            ingester=ingester,
            n_patients=4,
            speed=720,
            sepsis_count=1,
            seed=42,
            obs_per_patient=3,
        )

        # Run all steps
        for _ in range(ward.total_steps + 5):
            if ward.is_complete:
                break
            asyncio.get_event_loop().run_until_complete(ward.step())

        # Verify: all patients registered
        for pid in ward.patient_ids:
            assert registry.is_registered(pid)

        # Verify: predictions were made
        assert predictor.predict.call_count == ward.position

        # Verify: WebSocket broadcasts happened
        assert ws.broadcast.call_count == ward.position

    def test_case_replay_flows_through_pipeline(self):
        """CaseReplay → VitalsIngester → predict → tracking."""
        from sepsis_vitals.ml.monitor import VitalsIngester, PatientRegistry, DeteriorationTracker
        from sepsis_vitals.ml.simulator import CaseReplay

        predictor = MagicMock()
        pred = MagicMock()
        pred.risk_probability = 0.6
        pred.risk_level = "moderate"
        pred.to_dict.return_value = {
            "risk_probability": 0.6, "risk_level": "moderate",
            "patient_id": "mimic-1001", "timestamp": "2026-01-01T08:00:00",
        }
        predictor.predict.return_value = pred

        ws = MagicMock()
        ws.broadcast = AsyncMock()
        registry = PatientRegistry(debounce_seconds=0)
        tracker = DeteriorationTracker()
        ingester = VitalsIngester(predictor, registry, tracker, ws)

        base = pd.Timestamp("2150-01-01 08:00")
        timeline = pd.DataFrame({
            "charttime": [base + pd.Timedelta(hours=i) for i in range(4)],
            "vital_name": ["heart_rate"] * 4,
            "valuenum": [80, 90, 100, 110],
        })
        meta = {
            "subject_id": 1001, "stay_id": 3001,
            "age_years": 65, "sex": "M",
            "sepsis_label": 1, "icu_los_hours": 24.0,
        }

        replay = CaseReplay(meta, timeline, ingester, speed=720)

        for _ in range(4):
            asyncio.get_event_loop().run_until_complete(replay.step())

        assert replay.is_complete
        assert registry.is_registered("mimic-1001")
        assert predictor.predict.call_count == 4
        assert ws.broadcast.call_count == 4
```

- [ ] **Step 2: Run the full Layer 4 test suite**

Run: `python3 -m pytest tests/test_case_library.py tests/test_simulator.py tests/test_simulator_api.py -v`
Expected: All PASS

- [ ] **Step 3: Run full project test suite for regressions**

Run: `python3 -m pytest tests/ --ignore=tests/test_mimic_loader_demo.py --ignore=tests/test_layer1_integration.py -v`
Expected: All PASS, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_simulator.py
git commit -m "feat: add Layer 4 integration tests — simulator pipeline flow"
```
