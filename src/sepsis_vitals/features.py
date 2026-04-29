"""Feature engineering for vitals-only sepsis prediction.

The feature set is intentionally constrained to six bedside variables:
temperature, heart rate, respiratory rate, systolic blood pressure, SpO2,
and GCS. That constraint is the deployment hypothesis for district hospitals
where labs, imaging, dense EHR history, and continuous monitoring may be absent.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

TEMP = "temperature"
HR = "heart_rate"
RR = "resp_rate"
SBP = "sbp"
SPO2 = "spo2"
GCS = "gcs"

VITALS = [TEMP, HR, RR, SBP, SPO2, GCS]

QSOFA_RR_THRESH = 22
QSOFA_SBP_THRESH = 100
QSOFA_GCS_THRESH = 15

SIRS_TEMP_HIGH = 38.3
SIRS_TEMP_LOW = 36.0
SIRS_HR_HIGH = 90
SIRS_RR_HIGH = 20

SHOCK_SBP_THRESH = 90
SHOCK_INDEX_WARN = 0.7
SHOCK_INDEX_CRIT = 1.0

SPO2_NORMAL = 95
SPO2_LOW = 90
SPO2_CRIT = 85

TEMP_FEVER_HIGH = 39.0
TEMP_HYPO = 36.0

HR_BRADY = 60
HR_SEVERE = 130

RR_HIGH = 25
RR_CRIT = 30

HR_RAPID_RISE = 20
SBP_RAPID_DROP = -20
RR_RAPID_RISE = 5
SPO2_RAPID_DROP = -5

PEDIATRIC_THRESHOLDS: dict[int, dict[str, tuple[Optional[int], Optional[int]]]] = {
    1: {HR: (100, 180), RR: (25, 60), SBP: (70, None)},
    5: {HR: (80, 140), RR: (20, 40), SBP: (80, None)},
    12: {HR: (70, 120), RR: (18, 30), SBP: (90, None)},
    18: {HR: (60, 110), RR: (16, 25), SBP: (95, None)},
}


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add one binary missingness flag per available vital."""
    out = df.copy()
    for col in VITALS:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(np.int8)
    return out


def add_abnormality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinical threshold flags for adult vitals."""
    out = df.copy()

    if TEMP in out.columns:
        out["flag_fever"] = (out[TEMP] >= SIRS_TEMP_HIGH).astype(np.int8)
        out["flag_high_fever"] = (out[TEMP] >= TEMP_FEVER_HIGH).astype(np.int8)
        out["flag_hypothermia"] = (out[TEMP] < TEMP_HYPO).astype(np.int8)
        out["flag_temp_abnormal"] = (
            out["flag_fever"] | out["flag_hypothermia"]
        ).astype(np.int8)

    if HR in out.columns:
        out["flag_tachycardia"] = (out[HR] >= SIRS_HR_HIGH).astype(np.int8)
        out["flag_severe_tachycardia"] = (out[HR] >= HR_SEVERE).astype(np.int8)
        out["flag_bradycardia"] = (out[HR] < HR_BRADY).astype(np.int8)

    if RR in out.columns:
        out["flag_tachypnea_sirs"] = (out[RR] >= SIRS_RR_HIGH).astype(np.int8)
        out["flag_tachypnea_qsofa"] = (out[RR] >= QSOFA_RR_THRESH).astype(np.int8)
        out["flag_tachypnea_severe"] = (out[RR] >= RR_HIGH).astype(np.int8)
        out["flag_tachypnea_crit"] = (out[RR] >= RR_CRIT).astype(np.int8)

    if SBP in out.columns:
        out["flag_hypotension_qsofa"] = (out[SBP] <= QSOFA_SBP_THRESH).astype(np.int8)
        out["flag_hypotension_severe"] = (out[SBP] <= SHOCK_SBP_THRESH).astype(np.int8)

    if SPO2 in out.columns:
        out["flag_hypoxia"] = (out[SPO2] < SPO2_NORMAL).astype(np.int8)
        out["flag_hypoxia_mod"] = (out[SPO2] < SPO2_LOW).astype(np.int8)
        out["flag_hypoxia_crit"] = (out[SPO2] < SPO2_CRIT).astype(np.int8)

    if GCS in out.columns:
        out["flag_ams"] = (out[GCS] < QSOFA_GCS_THRESH).astype(np.int8)
        out["flag_ams_mod"] = (out[GCS] <= 13).astype(np.int8)
        out["flag_ams_severe"] = (out[GCS] <= 8).astype(np.int8)

    return out


def add_clinical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add qSOFA, partial SIRS, and a six-vital abnormality count."""
    out = df.copy()

    qsofa_parts = [
        c
        for c in ["flag_ams", "flag_tachypnea_qsofa", "flag_hypotension_qsofa"]
        if c in out.columns
    ]
    if qsofa_parts:
        out["qsofa_score"] = out[qsofa_parts].sum(axis=1).astype(np.int8)
        out["qsofa_positive"] = (out["qsofa_score"] >= 2).astype(np.int8)

    sirs_parts = [
        c
        for c in ["flag_temp_abnormal", "flag_tachycardia", "flag_tachypnea_sirs"]
        if c in out.columns
    ]
    if sirs_parts:
        out["sirs_partial"] = out[sirs_parts].sum(axis=1).astype(np.int8)
        out["sirs_partial_positive"] = (out["sirs_partial"] >= 2).astype(np.int8)

    primary_flags = [
        "flag_temp_abnormal",
        "flag_tachycardia",
        "flag_tachypnea_sirs",
        "flag_hypotension_qsofa",
        "flag_hypoxia",
        "flag_ams",
    ]
    present = [f for f in primary_flags if f in out.columns]
    if present:
        out["n_vitals_abnormal"] = out[present].sum(axis=1).astype(np.int8)

    return out


def add_shock_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add shock index and simple physiology interaction features."""
    out = df.copy()

    if HR in out.columns and SBP in out.columns:
        sbp_safe = out[SBP].replace(0, np.nan)
        out["shock_index"] = (out[HR] / sbp_safe).round(4)
        out["flag_si_elevated"] = (out["shock_index"] >= SHOCK_INDEX_WARN).astype(np.int8)
        out["flag_si_critical"] = (out["shock_index"] >= SHOCK_INDEX_CRIT).astype(np.int8)
        out["hr_sbp_diff"] = (out[HR] - out[SBP]).round(2)
        out["flag_hr_exceeds_sbp"] = (out["hr_sbp_diff"] > 0).astype(np.int8)

    if SPO2 in out.columns and HR in out.columns:
        out["spo2_hr_product"] = (out[SPO2] * out[HR] / 100).round(2)

    if RR in out.columns and SBP in out.columns:
        sbp_safe = out[SBP].replace(0, np.nan)
        out["rr_sbp_ratio"] = (out[RR] / sbp_safe).round(4)

    if TEMP in out.columns and HR in out.columns:
        out["temp_hr_product"] = (out[TEMP] * out[HR]).round(2)

    return out


def add_vital_deltas(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Add within-patient first differences and deterioration flags."""
    out = df.copy().sort_values([patient_col, time_col])

    for col in VITALS:
        if col not in out.columns:
            continue
        grouped = out.groupby(patient_col)[col]
        out[f"{col}_delta"] = grouped.diff().round(3)
        out[f"{col}_delta_pct"] = (
            grouped.pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .round(4)
        )

    if pd.api.types.is_datetime64_any_dtype(out[time_col]):
        out["hours_since_last"] = (
            out.groupby(patient_col)[time_col]
            .diff()
            .dt.total_seconds()
            .div(3600)
            .round(3)
        )
    else:
        out["hours_since_last"] = out.groupby(patient_col)[time_col].diff().round(3)

    if f"{HR}_delta" in out.columns:
        out["flag_hr_rapid_rise"] = (out[f"{HR}_delta"] > HR_RAPID_RISE).astype(np.int8)
    if f"{SBP}_delta" in out.columns:
        out["flag_sbp_rapid_drop"] = (out[f"{SBP}_delta"] < SBP_RAPID_DROP).astype(np.int8)
    if f"{RR}_delta" in out.columns:
        out["flag_rr_rapid_rise"] = (out[f"{RR}_delta"] > RR_RAPID_RISE).astype(np.int8)
    if f"{SPO2}_delta" in out.columns:
        out["flag_spo2_rapid_drop"] = (
            out[f"{SPO2}_delta"] < SPO2_RAPID_DROP
        ).astype(np.int8)

    det_flags = [
        c
        for c in [
            "flag_hr_rapid_rise",
            "flag_sbp_rapid_drop",
            "flag_rr_rapid_rise",
            "flag_spo2_rapid_drop",
        ]
        if c in out.columns
    ]
    if det_flags:
        out["n_vitals_deteriorating"] = out[det_flags].sum(axis=1).astype(np.int8)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    window: int = 3,
    min_periods: int = 2,
) -> pd.DataFrame:
    """Add rolling statistics over the last n readings per patient."""
    out = df.copy().sort_values([patient_col, time_col])

    for col in VITALS:
        if col not in out.columns:
            continue
        grouped_roll = out.groupby(patient_col)[col].rolling(
            window=window, min_periods=min_periods
        )

        for stat_name in ("mean", "std", "min", "max"):
            out[f"{col}_roll{window}_{stat_name}"] = (
                getattr(grouped_roll, stat_name)()
                .reset_index(level=0, drop=True)
                .round(3)
            )

        if col != GCS:
            mean_col = f"{col}_roll{window}_mean"
            std_col = f"{col}_roll{window}_std"
            out[f"{col}_roll{window}_cv"] = (
                out[std_col] / out[mean_col].replace(0, np.nan)
            ).round(4)

    return out


def add_missingness_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add structural missingness features across all available vitals."""
    out = df.copy()
    missing_cols = [f"{v}_missing" for v in VITALS if f"{v}_missing" in out.columns]
    if not missing_cols:
        return out

    out["n_vitals_missing"] = out[missing_cols].sum(axis=1).astype(np.int8)
    out["any_vital_missing"] = (out["n_vitals_missing"] > 0).astype(np.int8)
    out["all_vitals_present"] = (out["n_vitals_missing"] == 0).astype(np.int8)

    high_signal_missing = [
        f"{v}_missing" for v in [SBP, GCS, SPO2] if f"{v}_missing" in out.columns
    ]
    if high_signal_missing:
        out["flag_high_signal_miss"] = (
            out[high_signal_missing].sum(axis=1) > 0
        ).astype(np.int8)

    out["missingness_tier"] = pd.cut(
        out["n_vitals_missing"],
        bins=[-1, 0, 1, 2, 6],
        labels=[0, 1, 2, 3],
    ).astype(np.int8)

    return out


def _pediatric_flag(row: pd.Series, vital: str, age_col: str) -> float:
    age = row.get(age_col, np.nan)
    value = row.get(vital, np.nan)
    if pd.isna(age) or pd.isna(value):
        return np.nan

    for age_max, ranges in sorted(PEDIATRIC_THRESHOLDS.items()):
        if age <= age_max:
            low, high = ranges.get(vital, (None, None))
            outside = (high is not None and value > high) or (
                low is not None and value < low
            )
            return float(outside)

    if vital == HR:
        return float(value >= SIRS_HR_HIGH or value < HR_BRADY)
    if vital == RR:
        return float(value >= QSOFA_RR_THRESH)
    if vital == SBP:
        return float(value <= QSOFA_SBP_THRESH)
    return np.nan


def add_pediatric_flags(df: pd.DataFrame, age_col: str = "age_years") -> pd.DataFrame:
    """Add age-adjusted pediatric abnormality flags for HR, RR, and SBP."""
    if age_col not in df.columns:
        warnings.warn(
            f"age_col='{age_col}' not found; pediatric flags skipped. "
            "Pass age_col=None to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
        return df

    out = df.copy()
    for vital in [HR, RR, SBP]:
        if vital in out.columns:
            out[f"flag_{vital}_peds"] = (
                out.apply(lambda row: _pediatric_flag(row, vital, age_col), axis=1)
                .astype("Float64")
                .round(0)
                .astype("Int8")
            )
    return out


def add_episode_aggregates(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    """Add whole-episode summaries for one-row-per-episode models."""
    out = df.copy()
    for col in VITALS:
        if col not in out.columns:
            continue
        grouped = out.groupby(patient_col)[col]
        out[f"{col}_ep_mean"] = grouped.transform("mean").round(3)
        out[f"{col}_ep_std"] = grouped.transform("std").round(3)
        out[f"{col}_ep_min"] = grouped.transform("min").round(3)
        out[f"{col}_ep_max"] = grouped.transform("max").round(3)
        out[f"{col}_ep_range"] = (
            grouped.transform("max") - grouped.transform("min")
        ).round(3)
        out[f"{col}_ep_nreadings"] = grouped.transform("count").astype(np.int16)
    return out


def _validate_input(df: pd.DataFrame, patient_col: str, time_col: str) -> None:
    errors = []
    if patient_col not in df.columns:
        errors.append(f"patient_col '{patient_col}' not found in DataFrame columns.")
    if time_col not in df.columns:
        errors.append(f"time_col '{time_col}' not found in DataFrame columns.")
    if errors:
        raise ValueError("\n".join(errors))

    vital_cols_present = [v for v in VITALS if v in df.columns]
    if len(vital_cols_present) < 3:
        raise ValueError(
            "Fewer than 3 vital sign columns found. "
            f"Expected any of {VITALS}; got {vital_cols_present}."
        )

    missing_vitals = sorted(set(VITALS) - set(vital_cols_present))
    if missing_vitals:
        warnings.warn(
            f"Vital columns not found and will be skipped: {missing_vitals}",
            UserWarning,
            stacklevel=3,
        )


def build_feature_set(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    age_col: Optional[str] = "age_years",
    rolling_window: int = 3,
    include_episode_agg: bool = False,
) -> pd.DataFrame:
    """Run the full vitals-only feature engineering pipeline."""
    _validate_input(df, patient_col, time_col)

    result = df.copy()
    for step in [
        add_missingness_flags,
        add_abnormality_flags,
        add_clinical_scores,
        add_shock_indices,
    ]:
        result = step(result)

    result = add_vital_deltas(result, patient_col=patient_col, time_col=time_col)
    result = add_rolling_features(
        result,
        patient_col=patient_col,
        time_col=time_col,
        window=rolling_window,
    )
    result = add_missingness_patterns(result)

    if age_col is not None:
        result = add_pediatric_flags(result, age_col=age_col)

    if include_episode_agg:
        result = add_episode_aggregates(result, patient_col=patient_col)

    return result


def get_feature_inventory(df_engineered: pd.DataFrame) -> pd.DataFrame:
    """Return a compact model-card-ready inventory of engineered features."""
    meta: dict[str, tuple[str, str, str]] = {
        **{
            f"{v}_missing": ("binary", f"1 if {v} not recorded this reading", "0/1")
            for v in VITALS
        },
        "flag_fever": ("binary", "Temperature >= 38.3 C", "0/1"),
        "flag_high_fever": ("binary", "Temperature >= 39.0 C", "0/1"),
        "flag_hypothermia": ("binary", "Temperature < 36.0 C", "0/1"),
        "flag_temp_abnormal": ("binary", "Fever or hypothermia", "0/1"),
        "flag_tachycardia": ("binary", "HR >= 90 bpm", "0/1"),
        "flag_severe_tachycardia": ("binary", "HR >= 130 bpm", "0/1"),
        "flag_bradycardia": ("binary", "HR < 60 bpm", "0/1"),
        "flag_tachypnea_sirs": ("binary", "RR >= 20", "0/1"),
        "flag_tachypnea_qsofa": ("binary", "RR >= 22", "0/1"),
        "flag_tachypnea_severe": ("binary", "RR >= 25", "0/1"),
        "flag_tachypnea_crit": ("binary", "RR >= 30", "0/1"),
        "flag_hypotension_qsofa": ("binary", "SBP <= 100 mmHg", "0/1"),
        "flag_hypotension_severe": ("binary", "SBP <= 90 mmHg", "0/1"),
        "flag_hypoxia": ("binary", "SpO2 < 95%", "0/1"),
        "flag_hypoxia_mod": ("binary", "SpO2 < 90%", "0/1"),
        "flag_hypoxia_crit": ("binary", "SpO2 < 85%", "0/1"),
        "flag_ams": ("binary", "GCS < 15", "0/1"),
        "flag_ams_mod": ("binary", "GCS <= 13", "0/1"),
        "flag_ams_severe": ("binary", "GCS <= 8", "0/1"),
        "qsofa_score": ("ordinal", "qSOFA total score", "0-3"),
        "qsofa_positive": ("binary", "qSOFA >= 2", "0/1"),
        "sirs_partial": ("ordinal", "SIRS count without WBC", "0-3"),
        "sirs_partial_positive": ("binary", "Partial SIRS >= 2", "0/1"),
        "n_vitals_abnormal": ("ordinal", "Count of abnormal primary vitals", "0-6"),
        "shock_index": ("continuous", "HR / SBP", "0.3-3.0"),
        "flag_si_elevated": ("binary", "Shock index >= 0.7", "0/1"),
        "flag_si_critical": ("binary", "Shock index >= 1.0", "0/1"),
        "hr_sbp_diff": ("continuous", "HR minus SBP", "varies"),
        "flag_hr_exceeds_sbp": ("binary", "HR numerically exceeds SBP", "0/1"),
        "spo2_hr_product": ("continuous", "SpO2 * HR / 100", "varies"),
        "rr_sbp_ratio": ("continuous", "RR / SBP", "varies"),
        "temp_hr_product": ("continuous", "Temperature * HR", "varies"),
        **{
            f"{v}_delta": ("continuous", f"Absolute change in {v}", "varies")
            for v in VITALS
        },
        **{
            f"{v}_delta_pct": ("continuous", f"Percent change in {v}", "varies")
            for v in VITALS
        },
        "hours_since_last": ("continuous", "Hours since prior reading", "0-24"),
        "flag_hr_rapid_rise": ("binary", "HR rose >20 bpm", "0/1"),
        "flag_sbp_rapid_drop": ("binary", "SBP dropped >20 mmHg", "0/1"),
        "flag_rr_rapid_rise": ("binary", "RR rose >5 breaths/min", "0/1"),
        "flag_spo2_rapid_drop": ("binary", "SpO2 dropped >5 points", "0/1"),
        "n_vitals_deteriorating": ("ordinal", "Count of rapidly worsening vitals", "0-4"),
        "n_vitals_missing": ("ordinal", "Count of missing vitals", "0-6"),
        "any_vital_missing": ("binary", "At least one vital missing", "0/1"),
        "all_vitals_present": ("binary", "All six vitals present", "0/1"),
        "flag_high_signal_miss": ("binary", "SBP, GCS, or SpO2 missing", "0/1"),
        "missingness_tier": ("ordinal", "0 complete, 3 severe missingness", "0-3"),
        "flag_heart_rate_peds": ("binary", "HR outside age-adjusted range", "0/1"),
        "flag_resp_rate_peds": ("binary", "RR outside age-adjusted range", "0/1"),
        "flag_sbp_peds": ("binary", "SBP outside age-adjusted range", "0/1"),
    }

    for col in df_engineered.columns:
        if "_roll" in col:
            meta[col] = ("continuous", "Rolling within-patient vital statistic", "varies")
        elif "_ep_" in col:
            meta[col] = ("continuous", "Whole-episode vital summary", "varies")

    rows = [
        {
            "feature": feature,
            "type": values[0],
            "clinical_meaning": values[1],
            "range": values[2],
        }
        for feature, values in meta.items()
        if feature in df_engineered.columns
    ]
    return pd.DataFrame(rows).sort_values(["type", "feature"]).reset_index(drop=True)
