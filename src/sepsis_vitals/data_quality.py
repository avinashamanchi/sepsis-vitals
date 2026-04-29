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


def summarize_vitals_quality(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
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
