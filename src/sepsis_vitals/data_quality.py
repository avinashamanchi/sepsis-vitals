"""
sepsis_vitals/data_quality.py – Data quality auditing for vitals datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CORE_VITALS = ["temperature", "heart_rate", "resp_rate", "sbp", "spo2", "gcs"]

# Hard outlier ranges: values outside these are physically impossible.
HARD_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (25, 45),
    "heart_rate": (0, 350),
    "resp_rate": (0, 80),
    "sbp": (20, 350),
    "spo2": (0, 100),
    "gcs": (3, 15),
}


def _count_hard_outliers(series: pd.Series, vital: str) -> int:
    """Count values outside the hard range for a given vital."""
    lo, hi = HARD_RANGES[vital]
    valid = series.dropna()
    return int(((valid < lo) | (valid > hi)).sum())


def summarize_vitals_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize data quality for each core vital.

    Parameters
    ----------
    df : pd.DataFrame
        Vitals DataFrame.

    Returns
    -------
    pd.DataFrame
        Index = vital names, columns = n_present, completeness_pct, median,
        n_hard_outliers.
    """
    n_rows = len(df)
    records = []

    for v in CORE_VITALS:
        if v not in df.columns:
            records.append({
                "vital": v,
                "n_present": 0,
                "completeness_pct": 0.0,
                "median": np.nan,
                "n_hard_outliers": 0,
            })
        else:
            n_present = int(df[v].notna().sum())
            completeness_pct = 100.0 * n_present / n_rows if n_rows > 0 else 0.0
            median = float(df[v].median()) if n_present > 0 else np.nan
            n_hard_outliers = _count_hard_outliers(df[v], v)
            records.append({
                "vital": v,
                "n_present": n_present,
                "completeness_pct": completeness_pct,
                "median": median,
                "n_hard_outliers": n_hard_outliers,
            })

    result = pd.DataFrame(records).set_index("vital")
    result.index.name = None
    return result


def check_data_contract(df: pd.DataFrame) -> dict:
    """Check whether a DataFrame meets the data contract.

    Parameters
    ----------
    df : pd.DataFrame
        Vitals DataFrame.

    Returns
    -------
    dict
        Keys: passed (bool), errors (list), warnings (list),
        vitals_present (list).
    """
    errors: list[str] = []
    warnings_list: list[str] = []

    # Required columns
    if "patient_id" not in df.columns:
        errors.append("Required column 'patient_id' is missing")
    if "timestamp" not in df.columns:
        errors.append("Required column 'timestamp' is missing")

    # Vital columns present
    vitals_present = [v for v in CORE_VITALS if v in df.columns]
    if len(vitals_present) < 3:
        errors.append(
            f"Fewer than 3 vital columns present ({len(vitals_present)}): {vitals_present}"
        )

    # Duplicate rows
    if df.duplicated().any():
        n_dups = int(df.duplicated().sum())
        warnings_list.append(f"{n_dups} duplicate row(s) detected")

    # Per-vital checks
    n_rows = len(df)
    for v in vitals_present:
        if v in df.columns:
            n_present = int(df[v].notna().sum())
            completeness = 100.0 * n_present / n_rows if n_rows > 0 else 0.0
            if completeness < 50.0:
                warnings_list.append(
                    f"Low completeness for '{v}': {completeness:.1f}%"
                )
            n_outliers = _count_hard_outliers(df[v], v)
            if n_outliers > 0:
                warnings_list.append(
                    f"{n_outliers} hard outlier(s) in '{v}'"
                )

    passed = len(errors) == 0

    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings_list,
        "vitals_present": vitals_present,
    }


def temporal_quality(df: pd.DataFrame) -> dict:
    """Assess temporal quality of observations.

    Parameters
    ----------
    df : pd.DataFrame
        Vitals DataFrame with patient_id and timestamp columns.

    Returns
    -------
    dict
        Keys: n_patients, n_observations, gap_min_median_min.
        If timestamp column is missing, returns {"error": "..."}.
    """
    if "timestamp" not in df.columns:
        return {"error": "Required column 'timestamp' is missing"}
    if "patient_id" not in df.columns:
        return {"error": "Required column 'patient_id' is missing"}

    n_patients = df["patient_id"].nunique()
    n_observations = len(df)

    # Compute inter-observation gaps per patient
    sorted_df = df.sort_values(["patient_id", "timestamp"])
    gaps = (
        sorted_df.groupby("patient_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .dropna()
    )

    gap_min_median_min = float(gaps.median()) if len(gaps) > 0 else 0.0

    return {
        "n_patients": n_patients,
        "n_observations": n_observations,
        "gap_min_median_min": gap_min_median_min,
    }


def generate_quality_report(
    df: pd.DataFrame,
    site_id: str | None = None,
) -> dict:
    """Generate a comprehensive quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Vitals DataFrame.
    site_id : str | None
        Optional site identifier.

    Returns
    -------
    dict
        Keys: contract, temporal, vitals, overall_status, site_id.
    """
    contract = check_data_contract(df)
    temporal = temporal_quality(df)
    vitals_summary = summarize_vitals_quality(df)

    overall_status = "PASS" if contract["passed"] else "FAIL"

    return {
        "contract": contract,
        "temporal": temporal,
        "vitals": vitals_summary.to_dict(orient="index"),
        "overall_status": overall_status,
        "site_id": site_id,
    }
