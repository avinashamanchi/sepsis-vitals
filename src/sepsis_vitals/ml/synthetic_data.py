"""
sepsis_vitals.ml.synthetic_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clinically-grounded synthetic patient data generator for sepsis model training.

Generates realistic vital sign trajectories based on published clinical distributions:
- MIMIC-III vital sign distributions
- Surviving Sepsis Campaign 2021 guidelines
- NEWS2 reference ranges (Royal College of Physicians)
- qSOFA validation studies (Seymour et al., JAMA 2016)

Produces temporal patient encounters with realistic septic deterioration patterns,
including early/late sepsis phases, septic shock, and recovery trajectories.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Clinical reference distributions (based on MIMIC-III / literature)
# ---------------------------------------------------------------------------

# Normal adult vital sign distributions (mean, std)
NORMAL_VITALS = {
    "temperature": (37.0, 0.7),
    "heart_rate": (82.0, 16.0),
    "resp_rate": (17.0, 4.0),
    "sbp": (125.0, 22.0),
    "dbp": (75.0, 13.0),
    "spo2": (96.5, 2.0),
    "gcs": (15.0, 0.0),
    "map": (90.0, 15.0),
}

# Early sepsis (SIRS-positive, pre-organ dysfunction)
EARLY_SEPSIS_VITALS = {
    "temperature": (38.6, 0.8),
    "heart_rate": (105.0, 15.0),
    "resp_rate": (22.0, 4.0),
    "sbp": (105.0, 18.0),
    "dbp": (62.0, 12.0),
    "spo2": (94.5, 2.5),
    "gcs": (14.5, 0.8),
    "map": (76.0, 14.0),
}

# Severe sepsis / septic shock
SEVERE_SEPSIS_VITALS = {
    "temperature": (39.2, 1.0),
    "heart_rate": (125.0, 18.0),
    "resp_rate": (28.0, 5.0),
    "sbp": (82.0, 15.0),
    "dbp": (48.0, 10.0),
    "spo2": (90.0, 4.0),
    "gcs": (12.0, 2.5),
    "map": (60.0, 12.0),
}

# Hypothermic sepsis (subset — cold sepsis, ~15% of septic patients)
HYPOTHERMIC_SEPSIS_VITALS = {
    "temperature": (35.2, 0.6),
    "heart_rate": (115.0, 20.0),
    "resp_rate": (26.0, 5.0),
    "sbp": (85.0, 18.0),
    "dbp": (50.0, 12.0),
    "spo2": (91.0, 3.5),
    "gcs": (13.0, 2.0),
    "map": (62.0, 14.0),
}

# Age-dependent adjustments
AGE_ADJUSTMENTS = {
    # (hr_offset, sbp_offset, rr_offset, temp_offset)
    "18-30": (5.0, -5.0, 0.0, 0.0),
    "31-50": (0.0, 0.0, 0.0, 0.0),
    "51-65": (-3.0, 8.0, 1.0, -0.1),
    "66-80": (-5.0, 15.0, 2.0, -0.2),
    "80+": (-8.0, 10.0, 2.0, -0.3),
}

# Comorbidity vital sign modifiers
COMORBIDITY_MODIFIERS = {
    "hypertension": {"sbp": (15.0, 5.0), "dbp": (8.0, 3.0)},
    "diabetes": {"heart_rate": (5.0, 3.0)},
    "copd": {"resp_rate": (3.0, 2.0), "spo2": (-3.0, 1.5)},
    "heart_failure": {"heart_rate": (10.0, 5.0), "sbp": (-8.0, 5.0)},
}

# Vital sign physiological limits
VITAL_LIMITS = {
    "temperature": (30.0, 43.0),
    "heart_rate": (30, 220),
    "resp_rate": (4, 60),
    "sbp": (40, 260),
    "dbp": (20, 160),
    "spo2": (50, 100),
    "gcs": (3, 15),
    "map": (30, 180),
}

# Demographics
SEXES = ["M", "F"]
ETHNICITIES = [
    "non_hispanic_white",
    "non_hispanic_black",
    "hispanic",
    "asian",
    "other",
]
ETHNICITY_WEIGHTS = [0.40, 0.22, 0.20, 0.10, 0.08]

COMORBIDITY_LIST = ["hypertension", "diabetes", "copd", "heart_failure", "none"]
COMORBIDITY_PREVALENCE = {
    "hypertension": 0.35,
    "diabetes": 0.20,
    "copd": 0.12,
    "heart_failure": 0.08,
}


def _get_age_bucket(age: int) -> str:
    if age <= 30:
        return "18-30"
    elif age <= 50:
        return "31-50"
    elif age <= 65:
        return "51-65"
    elif age <= 80:
        return "66-80"
    else:
        return "80+"


def _clamp_vital(name: str, value: float) -> float:
    lo, hi = VITAL_LIMITS.get(name, (-1e9, 1e9))
    return float(np.clip(value, lo, hi))


def _round_vital(name: str, value: float) -> float:
    if name in ("temperature",):
        return round(value, 1)
    elif name in ("spo2",):
        return min(round(value, 0), 100.0)
    elif name == "gcs":
        return float(int(round(value)))
    else:
        return round(value, 0)


def generate_patient_trajectory(
    rng: np.random.Generator,
    patient_id: str,
    is_septic: bool,
    n_observations: int,
    age: int,
    sex: str,
    ethnicity: str,
    comorbidities: list,
    base_time: pd.Timestamp,
    sepsis_severity: str = "early",
) -> list:
    """Generate a temporal trajectory of vital signs for a single patient.

    For septic patients, simulates deterioration over time:
    - Phase 1 (0-30%): Normal-ish vitals with subtle early signs
    - Phase 2 (30-60%): Developing sepsis with worsening vitals
    - Phase 3 (60-100%): Full sepsis presentation

    For non-septic patients, generates stable vitals with normal variation.
    """
    rows = []
    age_bucket = _get_age_bucket(age)
    age_adj = AGE_ADJUSTMENTS[age_bucket]

    # Select target vitals based on sepsis state
    if not is_septic:
        target_vitals = NORMAL_VITALS
    elif sepsis_severity == "severe":
        target_vitals = SEVERE_SEPSIS_VITALS
    elif sepsis_severity == "hypothermic":
        target_vitals = HYPOTHERMIC_SEPSIS_VITALS
    else:
        target_vitals = EARLY_SEPSIS_VITALS

    # Compute comorbidity adjustments
    comorbidity_adj = {}
    for vital in NORMAL_VITALS:
        comorbidity_adj[vital] = (0.0, 0.0)

    for comorb in comorbidities:
        if comorb in COMORBIDITY_MODIFIERS:
            for vital, (mean_adj, std_adj) in COMORBIDITY_MODIFIERS[comorb].items():
                old = comorbidity_adj[vital]
                comorbidity_adj[vital] = (old[0] + mean_adj, old[1] + std_adj)

    for i in range(n_observations):
        # Time progression: observations every 2-6 hours with jitter
        hours_offset = i * rng.uniform(2.0, 6.0)
        timestamp = base_time + pd.Timedelta(hours=hours_offset)

        # Phase-based interpolation for septic patients
        progress = i / max(n_observations - 1, 1)

        if is_septic:
            # Interpolate from normal → septic over the trajectory
            if progress < 0.3:
                # Early phase: mostly normal with subtle signs
                blend = progress / 0.3 * 0.3
            elif progress < 0.6:
                # Developing phase
                blend = 0.3 + (progress - 0.3) / 0.3 * 0.4
            else:
                # Full sepsis
                blend = 0.7 + (progress - 0.6) / 0.4 * 0.3

            current_vitals = {}
            for vital in NORMAL_VITALS:
                normal_mean, normal_std = NORMAL_VITALS[vital]
                target_mean, target_std = target_vitals[vital]
                mean = normal_mean + blend * (target_mean - normal_mean)
                std = normal_std + blend * (target_std - normal_std)
                current_vitals[vital] = (mean, std)
        else:
            current_vitals = dict(NORMAL_VITALS)

        # Generate vital values
        row = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "age_years": age,
            "sex": sex,
            "ethnicity": ethnicity,
        }

        for vital in ["temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "gcs", "map"]:
            mean, std = current_vitals[vital]

            # Apply age adjustments
            if vital == "heart_rate":
                mean += age_adj[0]
            elif vital == "sbp":
                mean += age_adj[1]
            elif vital == "resp_rate":
                mean += age_adj[2]
            elif vital == "temperature":
                mean += age_adj[3]

            # Apply comorbidity adjustments
            c_mean, c_std = comorbidity_adj.get(vital, (0.0, 0.0))
            mean += c_mean
            std = max(std + c_std, 0.5)

            # Add temporal autocorrelation (vitals don't jump randomly)
            if i > 0 and vital in rows[-1] and not np.isnan(rows[-1].get(vital, np.nan)):
                prev_val = rows[-1][vital]
                # 60% previous value influence for continuity
                raw = rng.normal(mean, std)
                value = 0.4 * raw + 0.6 * prev_val
            else:
                value = rng.normal(mean, std)

            # Add measurement noise
            noise = rng.normal(0, std * 0.05)
            value += noise

            # Clamp and round
            value = _clamp_vital(vital, value)
            value = _round_vital(vital, value)

            if np.isnan(value):
                value = _round_vital(vital, mean)
            row[vital] = value

        # Introduce realistic missing data (~5% per vital for non-GCS)
        for vital in ["temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "map"]:
            if rng.random() < 0.05:
                row[vital] = np.nan

        # GCS rarely missing
        if rng.random() < 0.02:
            row["gcs"] = np.nan

        # Label
        if is_septic:
            # Label becomes positive when patient is in active sepsis phase
            if progress >= 0.25:
                row["sepsis_label"] = 1
            else:
                row["sepsis_label"] = 0
        else:
            row["sepsis_label"] = 0

        # Comorbidities as binary columns
        for comorb in ["hypertension", "diabetes", "copd", "heart_failure"]:
            row[f"has_{comorb}"] = 1 if comorb in comorbidities else 0

        rows.append(row)

    return rows


def generate_dataset(
    n_patients: int = 10000,
    sepsis_prevalence: float = 0.15,
    obs_per_patient: Tuple[int, int] = (6, 24),
    seed: int = 42,
    include_demographics: bool = True,
) -> pd.DataFrame:
    """Generate a full synthetic dataset for sepsis model training.

    Parameters
    ----------
    n_patients : int
        Number of unique patients to generate.
    sepsis_prevalence : float
        Fraction of patients who are septic (0.0 to 1.0).
    obs_per_patient : tuple of (min, max)
        Range of observations per patient.
    seed : int
        Random seed for reproducibility.
    include_demographics : bool
        Whether to include demographic columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: patient_id, timestamp, age_years, sex,
        ethnicity, temperature, heart_rate, resp_rate, sbp, dbp, spo2,
        gcs, map, sepsis_label, has_hypertension, has_diabetes, has_copd,
        has_heart_failure
    """
    rng = np.random.default_rng(seed)
    all_rows = []

    base_time = pd.Timestamp("2024-01-01 00:00:00")

    for pid in range(n_patients):
        patient_id = f"PT-{pid:06d}"

        # Demographics
        age = int(rng.integers(18, 95))
        sex = rng.choice(SEXES)
        ethnicity = rng.choice(ETHNICITIES, p=ETHNICITY_WEIGHTS)

        # Comorbidities (age-dependent prevalence)
        comorbidities = []
        age_factor = max(0.5, min(age / 60.0, 2.0))
        for comorb, base_prev in COMORBIDITY_PREVALENCE.items():
            if rng.random() < base_prev * age_factor:
                comorbidities.append(comorb)

        # Sepsis status
        # Older patients and those with comorbidities have higher sepsis risk
        sepsis_risk = sepsis_prevalence
        if age > 65:
            sepsis_risk *= 1.5
        if len(comorbidities) >= 2:
            sepsis_risk *= 1.3
        is_septic = rng.random() < min(sepsis_risk, 0.6)

        # Number of observations
        min_obs, max_obs = obs_per_patient
        n_obs = int(rng.integers(min_obs, max_obs + 1))

        # Sepsis severity distribution
        if is_septic:
            severity_roll = rng.random()
            if severity_roll < 0.5:
                severity = "early"
            elif severity_roll < 0.85:
                severity = "severe"
            else:
                severity = "hypothermic"
        else:
            severity = "early"  # won't be used

        # Patient-specific time offset (different admission times)
        patient_base_time = base_time + pd.Timedelta(hours=int(rng.integers(0, 8760)))

        rows = generate_patient_trajectory(
            rng=rng,
            patient_id=patient_id,
            is_septic=is_septic,
            n_observations=n_obs,
            age=age,
            sex=sex,
            ethnicity=ethnicity,
            comorbidities=comorbidities,
            base_time=patient_base_time,
            sepsis_severity=severity,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Ensure correct dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sepsis_label"] = df["sepsis_label"].astype(int)

    if not include_demographics:
        df = df.drop(columns=["sex", "ethnicity"], errors="ignore")

    return df


def generate_train_val_test(
    n_patients: int = 10000,
    sepsis_prevalence: float = 0.15,
    obs_per_patient: Tuple[int, int] = (6, 24),
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate train/validation/test splits at the patient level.

    Splits are done by patient_id to prevent data leakage.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    df = generate_dataset(
        n_patients=n_patients,
        sepsis_prevalence=sepsis_prevalence,
        obs_per_patient=obs_per_patient,
        seed=seed,
    )

    # Patient-level split to prevent leakage
    patient_ids = df["patient_id"].unique()
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(patient_ids)

    n_train = int(len(patient_ids) * train_frac)
    n_val = int(len(patient_ids) * val_frac)

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df
