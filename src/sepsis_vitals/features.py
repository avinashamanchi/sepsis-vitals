"""
features.py – Time-series feature engineering for sepsis vitals.

Design principles (from README):
  • Missingness is a signal, not just a nuisance.
  • Whole-episode aggregates are off by default (they leak future data).
  • qSOFA and partial SIRS are features *and* comparators.
  • Pediatric normalisation is applied when age_col is present.

Feature groups produced
-----------------------
1. Raw vitals (pass-through)
2. Missingness indicators (vital_missing_*)
3. Delta features (change from previous row per patient)
4. Rolling statistics (mean, std, min, max over a window)
5. Clinical scores per row (qSOFA, partial SIRS, shock index, NEWS2, UVA)
6. Pediatric z-score proxies when age_years < 18
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .scores import compute_scores

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CORE_VITALS = [
    "temperature",
    "heart_rate",
    "resp_rate",
    "sbp",
    "spo2",
    "gcs",
]

# Pediatric approximate normal ranges (age-banded medians, simplified)
# Format: (age_min_inclusive, age_max_exclusive): {vital: (median, sd)}
_PEDS_NORMS: dict[tuple[float, float], dict[str, tuple[float, float]]] = {
    (0,   1):  {"heart_rate": (130, 20), "resp_rate": (44, 8),  "sbp": (70, 10)},
    (1,   3):  {"heart_rate": (120, 15), "resp_rate": (30, 6),  "sbp": (80, 10)},
    (3,   6):  {"heart_rate": (110, 15), "resp_rate": (26, 5),  "sbp": (90, 10)},
    (6,  12):  {"heart_rate": (100, 15), "resp_rate": (22, 4),  "sbp": (95, 10)},
    (12, 18):  {"heart_rate": (90,  12), "resp_rate": (18, 3),  "sbp": (105, 12)},
}


def _peds_zscore(vital: str, value: float, age: float) -> Optional[float]:
    """Return a z-score relative to paediatric norms, or None if not applicable."""
    for (age_min, age_max), norms in _PEDS_NORMS.items():
        if age_min <= age < age_max and vital in norms:
            median, sd = norms[vital]
            return round((value - median) / sd, 3) if sd > 0 else None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_set(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    age_col: Optional[str] = "age_years",
    rolling_window: int = 3,
    include_episode_aggregates: bool = False,
    score_cols: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature matrix from a raw vitals DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `patient_col`, `time_col`, and at least 3 of the 6 core
        vitals. All vitals columns must be numeric.
    patient_col : str
        Column identifying each patient / encounter.
    time_col : str
        Datetime column for temporal ordering.
    age_col : str | None
        Column with patient age in years. Enables pediatric z-score features.
        Pass None to skip pediatric features.
    rolling_window : int
        Number of prior observations (within-patient) for rolling stats.
        Minimum 2, maximum 12.
    include_episode_aggregates : bool
        Off by default — whole-episode stats can leak future information.
        Set True only for retrospective analysis after labelling.
    score_cols : bool
        Whether to compute qSOFA, partial SIRS, NEWS2, UVA, shock index
        for each row.

    Returns
    -------
    pd.DataFrame
        Original columns plus all derived features. Index is unchanged.
    """
    df = df.copy()

    # ── Validate inputs ──────────────────────────────────────────────────────
    if patient_col not in df.columns:
        raise ValueError(f"patient_col='{patient_col}' not found in DataFrame.")
    if time_col not in df.columns:
        raise ValueError(f"time_col='{time_col}' not found in DataFrame.")

    present_vitals = [v for v in CORE_VITALS if v in df.columns]
    if len(present_vitals) < 3:
        raise ValueError(
            f"At least 3 core vitals required; found {len(present_vitals)}: "
            f"{present_vitals}. Full set: {CORE_VITALS}."
        )

    rolling_window = int(np.clip(rolling_window, 2, 12))

    # ── Sort within patient by time ──────────────────────────────────────────
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([patient_col, time_col]).reset_index(drop=True)

    # ── 1. Missingness indicators ────────────────────────────────────────────
    for v in CORE_VITALS:
        df[f"{v}_missing"] = (df[v].isna().astype(int)
                              if v in df.columns else 1)

    df["n_vitals_missing"] = sum(
        df[f"{v}_missing"] for v in CORE_VITALS
    )
    df["n_vitals_present"] = 6 - df["n_vitals_missing"]

    # ── 2. Delta features (change from previous row within patient) ──────────
    grp = df.groupby(patient_col, sort=False)

    for v in present_vitals:
        df[f"{v}_delta"] = grp[v].diff()

        # Absolute change and direction
        df[f"{v}_delta_abs"] = df[f"{v}_delta"].abs()
        df[f"{v}_delta_dir"] = np.sign(df[f"{v}_delta"].fillna(0)).astype(int)

    # Time gap between observations (minutes)
    df["_time_prev"] = grp[time_col].shift(1)
    df["obs_gap_min"] = (
        (df[time_col] - df["_time_prev"]).dt.total_seconds() / 60
    ).round(1)
    df.drop(columns=["_time_prev"], inplace=True)

    # ── 3. Rolling statistics ────────────────────────────────────────────────
    for v in present_vitals:
        rolling = grp[v].rolling(rolling_window, min_periods=1)
        df[f"{v}_roll_mean"] = rolling.mean().reset_index(level=0, drop=True).round(2)
        df[f"{v}_roll_std"]  = rolling.std().reset_index(level=0, drop=True).round(3)
        df[f"{v}_roll_min"]  = rolling.min().reset_index(level=0, drop=True)
        df[f"{v}_roll_max"]  = rolling.max().reset_index(level=0, drop=True)

    # Rolling missingness count
    for v in CORE_VITALS:
        if f"{v}_missing" in df.columns:
            df[f"{v}_roll_missing_count"] = (
                grp[f"{v}_missing"]
                .rolling(rolling_window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .astype(int)
            )

    # ── 4. Pediatric z-scores ────────────────────────────────────────────────
    if age_col and age_col in df.columns:
        peds_mask = df[age_col] < 18
        for v in ["heart_rate", "resp_rate", "sbp"]:
            if v in df.columns:
                col = f"{v}_peds_z"
                df[col] = np.nan
                for idx in df[peds_mask].index:
                    age = df.at[idx, age_col]
                    val = df.at[idx, v]
                    if pd.notna(age) and pd.notna(val):
                        z = _peds_zscore(v, float(val), float(age))
                        if z is not None:
                            df.at[idx, col] = z

    # ── 5. Clinical scores per row ───────────────────────────────────────────
    if score_cols:
        score_records = []
        vital_keys = present_vitals

        for _, row in df.iterrows():
            vitals_dict = {
                k: float(row[k]) for k in vital_keys if pd.notna(row.get(k))
            }
            bundle = compute_scores(vitals_dict)
            score_records.append(bundle.as_dict())

        score_df = pd.DataFrame(score_records, index=df.index)
        df = pd.concat([df, score_df], axis=1)

    # ── 6. Whole-episode aggregates (opt-in, future-leaking) ─────────────────
    if include_episode_aggregates:
        warnings.warn(
            "include_episode_aggregates=True: these features summarise the "
            "entire episode and WILL leak future information into training. "
            "Use only for retrospective analysis.",
            UserWarning,
            stacklevel=2,
        )
        for v in present_vitals:
            agg = grp[v].agg(["mean", "std", "min", "max"])
            agg.columns = [f"{v}_ep_{s}" for s in agg.columns]
            df = df.join(agg, on=patient_col)

    return df
