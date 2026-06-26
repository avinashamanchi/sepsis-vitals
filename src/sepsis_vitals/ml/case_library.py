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
