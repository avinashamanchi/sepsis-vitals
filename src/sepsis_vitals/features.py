"""
sepsis_vitals/features.py -- Feature engineering pipeline for vitals-based sepsis prediction.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


CORE_VITALS = ["temperature", "heart_rate", "resp_rate", "sbp", "spo2", "gcs"]
LAB_VALUES = ["lactate", "wbc", "procalcitonin"]

# Pediatric reference ranges (mean, std) for heart rate by rough age buckets.
# Simplified: single reference for ages < 18.
_PEDS_HR_MEAN = 100.0
_PEDS_HR_STD = 18.0


def build_feature_set(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    rolling_window: int = 3,
    score_cols: bool = True,
    age_col: Optional[str] = "age_years",
    include_episode_aggregates: bool = False,
) -> pd.DataFrame:
    """Build a feature set from a raw vitals DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw vitals data. Must contain *patient_col*, *time_col*, and at least
        three columns from :data:`CORE_VITALS`.
    patient_col : str
        Column identifying patient.
    time_col : str
        Column with observation timestamps.
    rolling_window : int
        Window size for rolling statistics; silently clipped to [1, 12].
    score_cols : bool
        If True, compute qSOFA and risk_level via ``sepsis_vitals.scores``.
    age_col : str | None
        Column with patient age in years. If provided and age < 18,
        pediatric z-scores are added.
    include_episode_aggregates : bool
        If True, add per-patient episode-level aggregates (leaks future
        information -- use only for retrospective analysis).

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with the same number of rows as *df*.
    """
    # ── validation ───────────────────────────────────────────────────────
    if patient_col not in df.columns:
        raise ValueError(
            f"patient_col '{patient_col}' not found in DataFrame columns"
        )
    if time_col not in df.columns:
        raise ValueError(
            f"time_col '{time_col}' not found in DataFrame columns"
        )

    vitals_present = [v for v in CORE_VITALS if v in df.columns]
    if len(vitals_present) < 3:
        raise ValueError(
            f"At least 3 core vital columns must be present, found {len(vitals_present)}: "
            f"{vitals_present}"
        )

    # ── copy & sort ──────────────────────────────────────────────────────
    out = df.copy()
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    # clip rolling window
    rolling_window = max(1, min(rolling_window, 12))

    # ── missingness indicators ───────────────────────────────────────────
    for v in CORE_VITALS:
        if v in out.columns:
            out[f"{v}_missing"] = out[v].isna()
        else:
            out[f"{v}_missing"] = True

    missing_cols = [f"{v}_missing" for v in CORE_VITALS]
    out["n_vitals_missing"] = out[missing_cols].sum(axis=1).astype(int)

    # ── per-patient grouped features ─────────────────────────────────────
    grouped = out.groupby(patient_col, sort=False)

    # delta features (diff from previous observation per patient)
    for v in vitals_present:
        out[f"{v}_delta"] = grouped[v].diff()

    # rolling features
    for v in vitals_present:
        roll = grouped[v].rolling(window=rolling_window, min_periods=1)
        out[f"{v}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        out[f"{v}_roll_std"] = roll.std().reset_index(level=0, drop=True)

    # observation gap in minutes
    out["obs_gap_min"] = (
        grouped[time_col]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )

    # ── lab value features ─────────────────────────────────────────────
    labs_present = [v for v in LAB_VALUES if v in out.columns]
    for v in labs_present:
        out[f"{v}_missing"] = out[v].isna()
    if labs_present:
        out["n_labs_missing"] = out[[f"{v}_missing" for v in labs_present]].sum(axis=1).astype(int)
        # Lab deltas and rolling stats
        for v in labs_present:
            out[f"{v}_delta"] = grouped[v].diff()
            roll = grouped[v].rolling(window=rolling_window, min_periods=1)
            out[f"{v}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)

    # ── score columns ────────────────────────────────────────────────────
    if score_cols:
        try:
            from sepsis_vitals.scores import compute_scores  # noqa: WPS433

            qsofa_vals = []
            risk_vals = []
            for _, row in out.iterrows():
                vitals_dict = {
                    v: row[v] for v in vitals_present if pd.notna(row.get(v))
                }
                result = compute_scores(vitals_dict)
                qsofa_vals.append(result.qsofa)
                risk_vals.append(result.risk_level)
            out["qsofa"] = qsofa_vals
            out["risk_level"] = risk_vals
        except ImportError:
            # scores module not available yet -- skip gracefully
            out["qsofa"] = np.nan
            out["risk_level"] = np.nan

    # ── pediatric z-scores ───────────────────────────────────────────────
    if age_col is not None and age_col in out.columns:
        if "heart_rate" in out.columns:
            peds_mask = out[age_col] < 18
            out["heart_rate_peds_z"] = np.nan
            if peds_mask.any():
                out.loc[peds_mask, "heart_rate_peds_z"] = (
                    (out.loc[peds_mask, "heart_rate"] - _PEDS_HR_MEAN) / _PEDS_HR_STD
                )

    # ── episode aggregates ───────────────────────────────────────────────
    if include_episode_aggregates:
        warnings.warn(
            "Episode aggregates leak future information -- use only for "
            "retrospective analysis.",
            UserWarning,
            stacklevel=2,
        )
        if "heart_rate" in out.columns:
            out["heart_rate_ep_mean"] = grouped["heart_rate"].transform("mean")

    return out
