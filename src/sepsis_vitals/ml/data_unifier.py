"""Time-window binning and multi-source data unification.

Clinical vitals are charted at irregular intervals. A heart rate logged at
10:01 and a blood pressure at 10:45 often belong to the same clinical
assessment window. This module bins observations into fixed-width epochs
and unifies data from multiple sources (CSV, FHIR NDJSON, etc.) into a
single training-ready DataFrame.

Aggregation rules follow clinical conventions:
- Most vitals use **median** (robust to outlier charting errors).
- GCS uses **max** (represents best neurological state in the window).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aggregation column lists
# ---------------------------------------------------------------------------

MEDIAN_VITALS = [
    "temperature",
    "heart_rate",
    "resp_rate",
    "sbp",
    "dbp",
    "spo2",
    "map",
    "lactate",
    "wbc",
    "procalcitonin",
]

MAX_VITALS = ["gcs"]  # GCS = best neurological state, use max


# ---------------------------------------------------------------------------
# bin_to_epochs
# ---------------------------------------------------------------------------


def bin_to_epochs(
    df: pd.DataFrame,
    epoch_minutes: int = 60,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Bin irregular observations into fixed-width time epochs.

    Parameters
    ----------
    df : pd.DataFrame
        Raw observations with at least *patient_col* and *time_col*.
    epoch_minutes : int
        Width of each epoch in minutes. Default 60.
    patient_col : str
        Column identifying the patient. Default ``"patient_id"``.
    time_col : str
        Column containing observation timestamps. Default ``"timestamp"``.
    forward_fill : bool
        If True, forward-fill missing vitals from the previous epoch
        within each patient. Default True.

    Returns
    -------
    pd.DataFrame
        One row per (patient, epoch) with aggregated vitals and an
        ``"epoch"`` column containing the floored timestamp.
    """
    if df.empty:
        result = df.copy()
        result["epoch"] = pd.Series(dtype="datetime64[ns]")
        return result

    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])

    # Floor timestamps to epoch boundaries
    freq = f"{epoch_minutes}min"
    data["epoch"] = data[time_col].dt.floor(freq)

    # Build aggregation rules per column
    all_cols = set(data.columns) - {patient_col, time_col, "epoch"}
    agg_dict: Dict[str, str] = {}

    median_present = [c for c in MEDIAN_VITALS if c in all_cols]
    max_present = [c for c in MAX_VITALS if c in all_cols]
    other_cols = all_cols - set(median_present) - set(max_present)

    for col in median_present:
        agg_dict[col] = "median"
    for col in max_present:
        agg_dict[col] = "max"
    for col in other_cols:
        agg_dict[col] = "last"

    if not agg_dict:
        # Only patient_col and time_col present -- just deduplicate
        result = data.drop_duplicates(subset=[patient_col, "epoch"])
        result = result.drop(columns=[time_col]).reset_index(drop=True)
        return result

    grouped = data.groupby([patient_col, "epoch"], as_index=False).agg(agg_dict)

    # Forward-fill missing vitals per patient
    if forward_fill:
        vitals_to_fill = median_present + max_present
        grouped = grouped.sort_values([patient_col, "epoch"])
        grouped[vitals_to_fill] = grouped.groupby(patient_col)[vitals_to_fill].ffill()

    grouped = grouped.reset_index(drop=True)
    return grouped


# ---------------------------------------------------------------------------
# unify_datasets
# ---------------------------------------------------------------------------


def unify_datasets(
    datasets: List[pd.DataFrame],
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    epoch_minutes: int = 60,
) -> pd.DataFrame:
    """Concatenate, bin, and deduplicate data from multiple sources.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        DataFrames from different sources (CSV, FHIR, etc.).
    patient_col : str
        Column identifying the patient. Default ``"patient_id"``.
    time_col : str
        Column containing observation timestamps. Default ``"timestamp"``.
    epoch_minutes : int
        Width of each epoch in minutes. Default 60.

    Returns
    -------
    pd.DataFrame
        Unified, deduplicated, epoch-binned DataFrame sorted by
        (*patient_col*, ``"epoch"``).
    """
    # Concatenate all sources (align columns automatically via concat)
    combined = pd.concat(datasets, ignore_index=True)

    # Bin to epochs
    binned = bin_to_epochs(
        combined,
        epoch_minutes=epoch_minutes,
        patient_col=patient_col,
        time_col=time_col,
        forward_fill=True,
    )

    # Deduplicate by (patient_col, epoch): keep row with most non-null vitals
    all_vitals = [c for c in MEDIAN_VITALS + MAX_VITALS if c in binned.columns]

    if all_vitals:
        binned["_vital_count"] = binned[all_vitals].notna().sum(axis=1)
        binned = binned.sort_values("_vital_count", ascending=False)
        binned = binned.drop_duplicates(subset=[patient_col, "epoch"], keep="first")
        binned = binned.drop(columns=["_vital_count"])

    # Sort by (patient_col, epoch)
    binned = binned.sort_values([patient_col, "epoch"]).reset_index(drop=True)

    n_obs = len(binned)
    n_patients = binned[patient_col].nunique()
    logger.info("Unified dataset: %d observations, %d patients", n_obs, n_patients)

    return binned
