<<<<<<< HEAD
"""Data quality checks for partner-site vitals extracts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sepsis_vitals.features import GCS, HR, RR, SBP, SPO2, TEMP, VITALS

PLAUSIBLE_RANGES = {
    TEMP: (25, 45),
    HR: (20, 250),
    RR: (4, 80),
    SBP: (40, 260),
    SPO2: (40, 100),
    GCS: (3, 15),
}

=======
"""
data_quality.py – Site data-quality auditing for partner extracts.

Functions
---------
summarize_vitals_quality(df)       Per-column completeness, range, outlier stats
check_data_contract(df)            Validate against the partner data contract
temporal_quality(df)               Gap analysis and observation frequency
generate_quality_report(df)        Full report dict suitable for JSON export
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .features import CORE_VITALS

# ──────────────────────────────────────────────────────────────────────────────
# Expected ranges (hard physiological limits for outlier detection)
# ──────────────────────────────────────────────────────────────────────────────

_RANGE_HARD: dict[str, tuple[float, float]] = {
    "temperature": (25.0, 45.0),
    "heart_rate":  (0.0,  350.0),
    "resp_rate":   (0.0,  80.0),
    "sbp":         (30.0, 300.0),
    "spo2":        (0.0,  100.0),
    "gcs":         (3.0,  15.0),
}

# Clinically plausible (soft) ranges — values outside are flagged but not removed
_RANGE_SOFT: dict[str, tuple[float, float]] = {
    "temperature": (33.0, 42.0),
    "heart_rate":  (30.0, 220.0),
    "resp_rate":   (4.0,  60.0),
    "sbp":         (50.0, 250.0),
    "spo2":        (50.0, 100.0),
    "gcs":         (3.0,  15.0),
}


# ──────────────────────────────────────────────────────────────────────────────
# Per-vital completeness & range stats
# ──────────────────────────────────────────────────────────────────────────────
>>>>>>> phase-0-expansion

def summarize_vitals_quality(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
<<<<<<< HEAD
    time_col: str = "timestamp",
) -> dict[str, Any]:
    """Return a JSON-serializable quality summary for a vitals extract."""
    n_rows = int(len(df))
    report: dict[str, Any] = {
        "n_rows": n_rows,
        "n_patients": int(df[patient_col].nunique()) if patient_col in df.columns else None,
        "has_time_col": time_col in df.columns,
        "vitals": {},
    }

    present_vitals = [vital for vital in VITALS if vital in df.columns]
    for vital in present_vitals:
        series = df[vital]
        low, high = PLAUSIBLE_RANGES[vital]
        non_missing = series.dropna()
        implausible = non_missing[(non_missing < low) | (non_missing > high)]

        report["vitals"][vital] = {
            "present": True,
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()) if n_rows else 0.0,
            "implausible_count": int(len(implausible)),
            "min": _maybe_float(non_missing.min()),
            "max": _maybe_float(non_missing.max()),
        }

    for vital in set(VITALS) - set(present_vitals):
        report["vitals"][vital] = {
            "present": False,
            "missing_count": n_rows,
            "missing_rate": 1.0 if n_rows else 0.0,
            "implausible_count": 0,
            "min": None,
            "max": None,
        }

    if all(vital in df.columns for vital in VITALS):
        complete_rows = df[VITALS].notna().all(axis=1)
        report["rows_with_all_six_vitals_rate"] = float(complete_rows.mean()) if n_rows else 0.0

        plausible_rows = complete_rows.copy()
        for vital in VITALS:
            low, high = PLAUSIBLE_RANGES[vital]
            plausible_rows &= df[vital].between(low, high)
        report["rows_with_all_six_plausible_vitals_rate"] = (
            float(plausible_rows.mean()) if n_rows else 0.0
        )
    else:
        report["rows_with_all_six_vitals_rate"] = 0.0
        report["rows_with_all_six_plausible_vitals_rate"] = 0.0

    return report


def _maybe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
=======
) -> pd.DataFrame:
    """
    Return a per-vital summary DataFrame with completeness and outlier stats.

    Columns returned:
        vital, n_total, n_present, completeness_pct,
        n_hard_outliers, n_soft_outliers,
        mean, std, p25, median, p75, p5, p95
    """
    records = []

    for vital in CORE_VITALS:
        if vital not in df.columns:
            records.append({
                "vital": vital,
                "n_total": len(df),
                "n_present": 0,
                "completeness_pct": 0.0,
                "n_hard_outliers": 0,
                "n_soft_outliers": 0,
                "mean": None, "std": None,
                "p5": None, "p25": None, "median": None, "p75": None, "p95": None,
            })
            continue

        col = pd.to_numeric(df[vital], errors="coerce")
        n_total   = len(col)
        n_present = int(col.notna().sum())
        present   = col.dropna()

        hard_lo, hard_hi = _RANGE_HARD.get(vital, (-np.inf, np.inf))
        soft_lo, soft_hi = _RANGE_SOFT.get(vital, (-np.inf, np.inf))

        n_hard = int(((present < hard_lo) | (present > hard_hi)).sum())
        n_soft = int(((present < soft_lo) | (present > soft_hi)).sum()) - n_hard

        records.append({
            "vital":             vital,
            "n_total":           n_total,
            "n_present":         n_present,
            "completeness_pct":  round(n_present / n_total * 100, 1) if n_total else 0.0,
            "n_hard_outliers":   n_hard,
            "n_soft_outliers":   max(n_soft, 0),
            "mean":              round(float(present.mean()), 2) if len(present) else None,
            "std":               round(float(present.std()), 2)  if len(present) else None,
            "p5":                round(float(present.quantile(0.05)), 2) if len(present) else None,
            "p25":               round(float(present.quantile(0.25)), 2) if len(present) else None,
            "median":            round(float(present.quantile(0.50)), 2) if len(present) else None,
            "p75":               round(float(present.quantile(0.75)), 2) if len(present) else None,
            "p95":               round(float(present.quantile(0.95)), 2) if len(present) else None,
        })

    return pd.DataFrame(records).set_index("vital")


# ──────────────────────────────────────────────────────────────────────────────
# Data contract validation
# ──────────────────────────────────────────────────────────────────────────────

_CONTRACT_REQUIRED_COLS = {"patient_id", "timestamp"}
_CONTRACT_MIN_VITALS = 3


def check_data_contract(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    min_vitals: int = _CONTRACT_MIN_VITALS,
) -> dict:
    """
    Validate a partner extract against the Sepsis Vitals data contract.

    Returns a dict with keys:
        passed (bool), errors (list[str]), warnings (list[str])
    """
    errors: list[str] = []
    warns:  list[str] = []

    # Required columns
    for col in [patient_col, time_col]:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    # Minimum vitals
    present = [v for v in CORE_VITALS if v in df.columns]
    if len(present) < min_vitals:
        errors.append(
            f"Only {len(present)} core vitals present; minimum is {min_vitals}. "
            f"Present: {present}. Full set: {CORE_VITALS}."
        )

    # Timestamp parseable
    if time_col in df.columns:
        try:
            pd.to_datetime(df[time_col])
        except Exception:
            errors.append(f"Column '{time_col}' cannot be parsed as datetime.")

    # Duplicate rows
    if patient_col in df.columns and time_col in df.columns:
        n_dupes = df.duplicated(subset=[patient_col, time_col]).sum()
        if n_dupes > 0:
            warns.append(f"{n_dupes} duplicate (patient_id, timestamp) rows found.")

    # Missingness check per vital
    for v in present:
        completeness = df[v].notna().mean() * 100
        if completeness < 20:
            errors.append(
                f"Vital '{v}' is {completeness:.0f}% complete — below 20% minimum."
            )
        elif completeness < 50:
            warns.append(
                f"Vital '{v}' is {completeness:.0f}% complete — consider excluding from Phase 1."
            )

    # Hard outliers
    for v in present:
        col = pd.to_numeric(df[v], errors="coerce")
        lo, hi = _RANGE_HARD[v]
        n_out = int(((col < lo) | (col > hi)).sum())
        if n_out > 0:
            warns.append(
                f"Vital '{v}' has {n_out} values outside hard range [{lo}, {hi}]."
            )

    return {
        "passed":   len(errors) == 0,
        "errors":   errors,
        "warnings": warns,
        "vitals_present": present,
        "n_rows":   len(df),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Temporal quality
# ──────────────────────────────────────────────────────────────────────────────

def temporal_quality(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> dict:
    """
    Analyse observation frequency and gap distribution.

    Returns summary statistics on inter-observation gaps per patient.
    """
    if patient_col not in df.columns or time_col not in df.columns:
        return {"error": "patient_col or time_col not found"}

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([patient_col, time_col])

    df["_gap_min"] = (
        df.groupby(patient_col)[time_col]
        .diff()
        .dt.total_seconds()
        / 60
    )

    gaps = df["_gap_min"].dropna()
    obs_per_pt = df.groupby(patient_col).size()

    return {
        "n_patients":            int(df[patient_col].nunique()),
        "n_observations":        int(len(df)),
        "obs_per_patient_mean":  round(float(obs_per_pt.mean()), 1),
        "obs_per_patient_median":round(float(obs_per_pt.median()), 1),
        "obs_per_patient_min":   int(obs_per_pt.min()),
        "obs_per_patient_max":   int(obs_per_pt.max()),
        "gap_min_mean_min":      round(float(gaps.mean()), 1)  if len(gaps) else None,
        "gap_min_median_min":    round(float(gaps.median()), 1) if len(gaps) else None,
        "gap_min_p90_min":       round(float(gaps.quantile(0.9)), 1) if len(gaps) else None,
        "gap_min_max_min":       round(float(gaps.max()), 1)   if len(gaps) else None,
        "pct_gaps_over_4h":      round(float((gaps > 240).mean() * 100), 1) if len(gaps) else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full report
# ──────────────────────────────────────────────────────────────────────────────

def generate_quality_report(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    site_id: Optional[str] = None,
) -> dict:
    """
    Produce a full data-quality report dict, suitable for JSON serialisation.

    Combines contract check + vital summaries + temporal analysis.
    """
    contract  = check_data_contract(df, patient_col=patient_col, time_col=time_col)
    temporal  = temporal_quality(df, patient_col=patient_col, time_col=time_col)
    vitals_df = summarize_vitals_quality(df, patient_col=patient_col)

    return {
        "site_id":       site_id,
        "contract":      contract,
        "temporal":      temporal,
        "vitals":        vitals_df.reset_index().to_dict(orient="records"),
        "overall_status": "PASS" if contract["passed"] else "FAIL",
    }
>>>>>>> phase-0-expansion
