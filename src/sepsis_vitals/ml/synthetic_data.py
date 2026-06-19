"""
sepsis_vitals.ml.synthetic_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clinically-grounded synthetic patient data generator for sepsis model training.

Generates realistic vital sign trajectories calibrated to population-level
reference distributions from the National Health and Nutrition Examination
Survey (NHANES), with age-, sex-, and ethnicity-stratified vital sign norms.

Sepsis phenotypes are based on published clinical distributions:
- MIMIC-III vital sign distributions (early, severe, hypothermic sepsis)
- Surviving Sepsis Campaign 2021 guidelines
- NEWS2 reference ranges (Royal College of Physicians)
- qSOFA validation studies (Seymour et al., JAMA 2016)

NHANES data sources:
- Blood pressure: NHANES 2017-2020 (P_BPXO)
- Pulse oximetry: NHANES 2011-2012 through 2017-2020
- Heart rate: NHANES continuous (pulse rate from BP component)
- Temperature, respiratory rate: supplemented from published
  population norms indexed to NHANES age/sex strata

Produces temporal patient encounters with realistic septic deterioration
patterns, including early/late sepsis phases, septic shock, and recovery
trajectories.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# NHANES population reference distributions by age and sex
# Each entry is (mean, standard_deviation)
# ---------------------------------------------------------------------------

NHANES_VITALS = {
    "temperature": {
        "M": {
            "18-29": (36.6, 0.3),
            "30-49": (36.5, 0.3),
            "50-69": (36.4, 0.4),
            "70+": (36.3, 0.4),
        },
        "F": {
            "18-29": (36.7, 0.3),
            "30-49": (36.6, 0.3),
            "50-69": (36.5, 0.4),
            "70+": (36.4, 0.4),
        },
    },
    "heart_rate": {
        "M": {
            "18-29": (71.2, 11.4),
            "30-49": (73.1, 12.0),
            "50-69": (71.7, 12.2),
            "70+": (69.2, 12.8),
        },
        "F": {
            "18-29": (75.8, 11.2),
            "30-49": (76.3, 11.7),
            "50-69": (74.0, 11.7),
            "70+": (71.8, 12.5),
        },
    },
    "resp_rate": {
        "M": {
            "18-29": (15.4, 2.8),
            "30-49": (15.8, 3.0),
            "50-69": (16.7, 3.3),
            "70+": (17.9, 3.7),
        },
        "F": {
            "18-29": (16.2, 2.9),
            "30-49": (16.6, 3.1),
            "50-69": (17.4, 3.4),
            "70+": (18.3, 3.8),
        },
    },
    "sbp": {
        "M": {
            "18-29": (118.2, 10.8),
            "30-49": (123.2, 13.3),
            "50-69": (131.5, 17.0),
            "70+": (138.1, 19.5),
        },
        "F": {
            "18-29": (110.4, 10.2),
            "30-49": (115.9, 13.4),
            "50-69": (130.1, 18.0),
            "70+": (140.2, 20.1),
        },
    },
    "dbp": {
        "M": {
            "18-29": (70.1, 9.4),
            "30-49": (76.1, 10.4),
            "50-69": (75.0, 11.2),
            "70+": (66.5, 12.2),
        },
        "F": {
            "18-29": (67.2, 8.8),
            "30-49": (71.8, 9.9),
            "50-69": (72.5, 10.7),
            "70+": (64.5, 11.8),
        },
    },
    "spo2": {
        "M": {
            "18-29": (97.8, 1.1),
            "30-49": (97.5, 1.3),
            "50-69": (97.0, 1.5),
            "70+": (96.2, 1.8),
        },
        "F": {
            "18-29": (97.9, 1.0),
            "30-49": (97.6, 1.2),
            "50-69": (97.1, 1.4),
            "70+": (96.3, 1.7),
        },
    },
}

# GCS has no NHANES data; healthy baseline is universally 15
NHANES_GCS_NORMAL = (15.0, 0.0)

# ---------------------------------------------------------------------------
# NHANES ethnicity-based blood pressure adjustments
# Offsets relative to population mean (~124 SBP)
# ---------------------------------------------------------------------------

ETHNICITY_BP_ADJUSTMENTS = {
    "non_hispanic_white": {"sbp": 0.0, "dbp": 0.0},
    "non_hispanic_black": {"sbp": 5.7, "dbp": 3.7},
    "mexican_american": {"sbp": -1.8, "dbp": -1.6},
    "other_hispanic": {"sbp": -1.0, "dbp": -1.0},
    "asian": {"sbp": -2.4, "dbp": 0.5},
    "other": {"sbp": 0.7, "dbp": 0.4},
}

# ---------------------------------------------------------------------------
# NHANES comorbidity prevalence by age bracket (proportion)
# ---------------------------------------------------------------------------

NHANES_COMORBIDITY_PREVALENCE = {
    "hypertension": {
        "18-29": 0.078,
        "30-49": 0.171,
        "50-69": 0.418,
        "70+": 0.617,
    },
    "diabetes": {
        "18-29": 0.024,
        "30-49": 0.073,
        "50-69": 0.203,
        "70+": 0.260,
    },
    "ckd": {
        "18-29": 0.062,
        "30-49": 0.086,
        "50-69": 0.186,
        "70+": 0.363,
    },
    "copd": {
        # ~12% overall, age-scaled
        "18-29": 0.04,
        "30-49": 0.08,
        "50-69": 0.15,
        "70+": 0.22,
    },
    "heart_failure": {
        # ~8% overall, age-scaled
        "18-29": 0.01,
        "30-49": 0.03,
        "50-69": 0.09,
        "70+": 0.18,
    },
}

# ---------------------------------------------------------------------------
# Sepsis incidence per 100,000 by age (used to derive age-dependent risk)
# ---------------------------------------------------------------------------

SEPSIS_INCIDENCE_PER_100K = {
    "18-29": 52,
    "30-39": 78,
    "40-49": 134,
    "50-59": 242,
    "60-69": 418,
    "70-79": 682,
    "80+": 1024,
}

# ---------------------------------------------------------------------------
# MIMIC-III based sepsis vital sign distributions
# Recalibrated to produce realistic AUROC (0.75-0.85) by:
#   - Adding a "subtle/occult" phenotype (40% of septic patients)
#   - Making early sepsis milder (closer to sick-but-not-septic)
#   - Widening standard deviations for more overlap
# ---------------------------------------------------------------------------

# Early sepsis (SIRS-positive, pre-organ dysfunction)
# Pulled closer to normal -- many early sepsis patients look nearly normal
EARLY_SEPSIS_VITALS = {
    "temperature": (37.8, 0.8),
    "heart_rate": (95.0, 15.0),
    "resp_rate": (20.0, 4.0),
    "sbp": (112.0, 18.0),
    "dbp": (66.0, 12.0),
    "spo2": (95.5, 2.5),
    "gcs": (14.8, 0.5),
    "map": (81.0, 14.0),
}

# Subtle / occult sepsis -- nearly normal vitals, hard to detect (~40% of septic)
# These patients are the reason real-world AUROC is 0.75-0.85, not 0.99
SUBTLE_SEPSIS_VITALS = {
    "temperature": (37.3, 0.6),
    "heart_rate": (88.0, 14.0),
    "resp_rate": (18.5, 3.5),
    "sbp": (118.0, 18.0),
    "dbp": (70.0, 12.0),
    "spo2": (96.0, 2.0),
    "gcs": (14.9, 0.3),
    "map": (86.0, 14.0),
}

# Severe sepsis / septic shock
SEVERE_SEPSIS_VITALS = {
    "temperature": (39.0, 1.0),
    "heart_rate": (120.0, 18.0),
    "resp_rate": (26.0, 5.0),
    "sbp": (85.0, 15.0),
    "dbp": (50.0, 10.0),
    "spo2": (91.0, 4.0),
    "gcs": (12.5, 2.5),
    "map": (62.0, 12.0),
}

# Hypothermic sepsis (subset -- cold sepsis, ~10% of septic patients)
HYPOTHERMIC_SEPSIS_VITALS = {
    "temperature": (35.4, 0.6),
    "heart_rate": (110.0, 20.0),
    "resp_rate": (24.0, 5.0),
    "sbp": (90.0, 18.0),
    "dbp": (52.0, 12.0),
    "spo2": (92.0, 3.5),
    "gcs": (13.5, 2.0),
    "map": (65.0, 14.0),
}

# Comorbidity vital sign modifiers (additive mean offset, additive std offset)
COMORBIDITY_MODIFIERS = {
    "hypertension": {"sbp": (15.0, 5.0), "dbp": (8.0, 3.0)},
    "diabetes": {"heart_rate": (5.0, 3.0)},
    "copd": {"resp_rate": (3.0, 2.0), "spo2": (-3.0, 1.5)},
    "heart_failure": {"heart_rate": (10.0, 5.0), "sbp": (-8.0, 5.0)},
    "ckd": {"sbp": (6.0, 3.0), "dbp": (3.0, 2.0)},
}

# ---------------------------------------------------------------------------
# Lab value distributions (MIMIC-III / literature calibrated)
# ---------------------------------------------------------------------------

# Normal ranges for lab values
LAB_NORMALS = {
    "lactate": (1.0, 0.4),        # mmol/L — normal <2.0
    "wbc": (7.5, 2.0),            # x10^9/L — normal 4.5-11.0
    "procalcitonin": (0.05, 0.03),  # ng/mL — normal <0.1
}

# Sepsis lab distributions -- recalibrated for realistic overlap
EARLY_SEPSIS_LABS = {
    "lactate": (2.2, 1.0),        # Mildly elevated, overlaps with sick-nonseptic
    "wbc": (12.0, 4.5),           # Elevated but wide overlap
    "procalcitonin": (1.5, 1.5),  # Moderately elevated, large variance
}

# Subtle sepsis labs -- barely abnormal, within sick-nonseptic range
SUBTLE_SEPSIS_LABS = {
    "lactate": (1.6, 0.7),        # Borderline
    "wbc": (10.0, 3.5),           # Mildly elevated
    "procalcitonin": (0.4, 0.4),  # Low — subtle sepsis often has low PCT
}

SEVERE_SEPSIS_LABS = {
    "lactate": (5.0, 2.5),        # Significantly elevated
    "wbc": (16.0, 7.0),           # High or low (bimodal in severe sepsis)
    "procalcitonin": (12.0, 8.0), # Highly elevated
}

HYPOTHERMIC_SEPSIS_LABS = {
    "lactate": (3.5, 1.8),
    "wbc": (3.5, 1.5),            # Leukopenia common in hypothermic sepsis
    "procalcitonin": (6.0, 5.0),
}

# Sick-but-not-septic lab distributions (the KEY confounder)
# Deliberately pushed closer to early/subtle sepsis labs.  In real ICUs,
# post-surgical patients routinely have lactate 2-4, WBC 12-18, and
# procalcitonin 0.5-2 ng/mL -- all overlapping heavily with early sepsis.
SICK_NONSEPTIC_LABS = {
    "post_surgical": {"lactate": (2.8, 1.2), "wbc": (13.0, 4.0), "procalcitonin": (1.0, 0.8)},
    "dehydration": {"lactate": (2.5, 1.0), "wbc": (9.5, 3.5), "procalcitonin": (0.15, 0.12)},
    "pain_anxiety": {"lactate": (1.8, 0.7), "wbc": (10.5, 3.0), "procalcitonin": (0.08, 0.05)},
    "copd_exacerbation": {"lactate": (2.2, 0.9), "wbc": (13.5, 4.5), "procalcitonin": (0.7, 0.5)},
    "hf_exacerbation": {"lactate": (3.5, 1.5), "wbc": (10.5, 3.5), "procalcitonin": (0.4, 0.3)},
    "viral_infection": {"lactate": (1.5, 0.6), "wbc": (4.5, 2.5), "procalcitonin": (0.15, 0.1)},
}

# Sick-but-not-septic vital distributions — DELIBERATELY overlapping with
# early and subtle sepsis to create realistic classification difficulty.
# In real ICUs the overlap between septic and non-septic SIRS is massive;
# that's what makes sepsis screening genuinely hard (Seymour et al., JAMA 2016).
SICK_NONSEPTIC_VITALS = {
    "post_surgical": {
        "temperature": (38.0, 0.7), "heart_rate": (98.0, 14.0),
        "resp_rate": (20.0, 4.0), "sbp": (108.0, 18.0), "dbp": (64.0, 12.0),
        "spo2": (95.0, 2.0), "gcs": (14.5, 0.8), "map": (79.0, 14.0),
    },
    "dehydration": {
        "temperature": (37.1, 0.5), "heart_rate": (105.0, 16.0),
        "resp_rate": (19.0, 4.0), "sbp": (95.0, 18.0), "dbp": (58.0, 13.0),
        "spo2": (96.0, 2.0), "gcs": (14.2, 1.0), "map": (70.0, 14.0),
    },
    "pain_anxiety": {
        "temperature": (37.0, 0.4), "heart_rate": (102.0, 14.0),
        "resp_rate": (22.0, 4.5), "sbp": (135.0, 20.0), "dbp": (80.0, 12.0),
        "spo2": (96.5, 1.5), "gcs": (15.0, 0.0), "map": (98.0, 14.0),
    },
    "copd_exacerbation": {
        "temperature": (37.4, 0.6), "heart_rate": (98.0, 16.0),
        "resp_rate": (24.0, 5.0), "sbp": (122.0, 20.0), "dbp": (70.0, 12.0),
        "spo2": (89.0, 3.5), "gcs": (14.2, 1.0), "map": (87.0, 14.0),
    },
    "hf_exacerbation": {
        "temperature": (36.6, 0.4), "heart_rate": (108.0, 18.0),
        "resp_rate": (24.0, 5.0), "sbp": (90.0, 18.0), "dbp": (54.0, 12.0),
        "spo2": (90.0, 3.5), "gcs": (13.8, 1.2), "map": (66.0, 14.0),
    },
    "viral_infection": {
        "temperature": (38.4, 0.8), "heart_rate": (96.0, 14.0),
        "resp_rate": (19.0, 4.0), "sbp": (115.0, 16.0), "dbp": (70.0, 12.0),
        "spo2": (95.5, 2.0), "gcs": (15.0, 0.0), "map": (85.0, 13.0),
    },
}

SICK_NONSEPTIC_TYPES = list(SICK_NONSEPTIC_VITALS.keys())

LAB_LIMITS = {
    "lactate": (0.1, 20.0),
    "wbc": (0.1, 50.0),
    "procalcitonin": (0.01, 200.0),
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
    "mexican_american",
    "other_hispanic",
    "asian",
    "other",
]
ETHNICITY_WEIGHTS = [0.40, 0.22, 0.13, 0.07, 0.10, 0.08]

COMORBIDITY_LIST = [
    "hypertension",
    "diabetes",
    "ckd",
    "copd",
    "heart_failure",
    "none",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_nhanes_age_bucket(age: int) -> str:
    """Map a numeric age to the NHANES age bucket used for vital sign lookup."""
    if age <= 29:
        return "18-29"
    elif age <= 49:
        return "30-49"
    elif age <= 69:
        return "50-69"
    else:
        return "70+"


def _get_sepsis_age_bucket(age: int) -> str:
    """Map a numeric age to the sepsis incidence age bucket."""
    if age <= 29:
        return "18-29"
    elif age <= 39:
        return "30-39"
    elif age <= 49:
        return "40-49"
    elif age <= 59:
        return "50-59"
    elif age <= 69:
        return "60-69"
    elif age <= 79:
        return "70-79"
    else:
        return "80+"


def get_nhanes_normals(age: int, sex: str) -> dict:
    """Return a dict of normal vital sign distributions for a given age and sex.

    Uses NHANES population reference data stratified by age bracket and sex.
    MAP is computed from the SBP and DBP distributions rather than stored
    independently.

    Parameters
    ----------
    age : int
        Patient age in years (18+).
    sex : str
        'M' or 'F'.

    Returns
    -------
    dict
        Mapping of vital name to (mean, std) tuple.
    """
    bucket = _get_nhanes_age_bucket(age)
    normals = {}

    for vital in ("temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2"):
        normals[vital] = NHANES_VITALS[vital][sex][bucket]

    normals["gcs"] = NHANES_GCS_NORMAL

    # Compute MAP = (SBP + 2*DBP) / 3 from component distributions
    sbp_mean, sbp_std = normals["sbp"]
    dbp_mean, dbp_std = normals["dbp"]
    map_mean = (sbp_mean + 2.0 * dbp_mean) / 3.0
    # Propagate uncertainty: Var(MAP) = (1/9)*Var(SBP) + (4/9)*Var(DBP)
    map_std = np.sqrt((sbp_std ** 2 + 4.0 * dbp_std ** 2) / 9.0)
    normals["map"] = (round(map_mean, 1), round(map_std, 1))

    return normals


def _get_ethnicity_bp_adj(ethnicity: str) -> Tuple[float, float]:
    """Return (sbp_offset, dbp_offset) for the given ethnicity."""
    adj = ETHNICITY_BP_ADJUSTMENTS.get(ethnicity, {"sbp": 0.0, "dbp": 0.0})
    return adj["sbp"], adj["dbp"]


def _get_sepsis_risk_multiplier(age: int) -> float:
    """Return an age-dependent sepsis risk multiplier based on NHANES incidence.

    The multiplier is normalized so that the overall population-weighted
    average is approximately 1.0, preserving the caller's requested
    sepsis_prevalence as the marginal rate.
    """
    bucket = _get_sepsis_age_bucket(age)
    incidence = SEPSIS_INCIDENCE_PER_100K[bucket]
    # Approximate population-weighted mean incidence across adult ages
    # (weighted by typical hospital admission age distribution)
    reference_incidence = 250.0
    return incidence / reference_incidence


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
    sick_type: Optional[str] = None,
) -> list:
    """Generate a temporal trajectory of vital signs for a single patient.

    For septic patients, simulates deterioration over time with variable onset.
    For sick-but-not-septic patients, generates abnormal vitals that mimic
    sepsis presentations (the key clinical confounder).
    For healthy patients, generates stable vitals with normal variation.

    Baseline vitals are drawn from NHANES age/sex-stratified distributions,
    with ethnicity-based blood pressure adjustments applied.
    """
    rows: list[dict] = []

    # Get NHANES-calibrated normals for this patient's age and sex
    normal_vitals = get_nhanes_normals(age, sex)

    # Ethnicity-based BP offsets
    sbp_eth_adj, dbp_eth_adj = _get_ethnicity_bp_adj(ethnicity)

    # Select target vitals and labs based on patient type
    if is_septic:
        if sepsis_severity == "severe":
            target_vitals = SEVERE_SEPSIS_VITALS
            target_labs = SEVERE_SEPSIS_LABS
        elif sepsis_severity == "hypothermic":
            target_vitals = HYPOTHERMIC_SEPSIS_VITALS
            target_labs = HYPOTHERMIC_SEPSIS_LABS
        elif sepsis_severity == "subtle":
            target_vitals = SUBTLE_SEPSIS_VITALS
            target_labs = SUBTLE_SEPSIS_LABS
        else:
            target_vitals = EARLY_SEPSIS_VITALS
            target_labs = EARLY_SEPSIS_LABS
    elif sick_type is not None and sick_type in SICK_NONSEPTIC_VITALS:
        target_vitals = SICK_NONSEPTIC_VITALS[sick_type]
        target_labs = SICK_NONSEPTIC_LABS[sick_type]
    else:
        target_vitals = normal_vitals
        target_labs = LAB_NORMALS

    normal_labs = LAB_NORMALS

    # Variable sepsis onset point (not always at 25%)
    sepsis_onset = rng.uniform(0.15, 0.45) if is_septic else 1.0

    # Per-patient severity jitter (not every sepsis patient looks textbook)
    # Subtle sepsis gets very low jitter — these patients barely look septic
    if is_septic and sepsis_severity == "subtle":
        severity_jitter = rng.uniform(0.2, 0.6)
    elif is_septic and sepsis_severity == "early":
        severity_jitter = rng.uniform(0.4, 0.8)
    elif is_septic:
        severity_jitter = rng.uniform(0.5, 1.0)
    else:
        severity_jitter = 1.0

    # Compute comorbidity adjustments
    comorbidity_adj: dict[str, Tuple[float, float]] = {}
    for vital in normal_vitals:
        comorbidity_adj[vital] = (0.0, 0.0)

    for comorb in comorbidities:
        if comorb in COMORBIDITY_MODIFIERS:
            for vital, (mean_adj, std_adj) in COMORBIDITY_MODIFIERS[comorb].items():
                old = comorbidity_adj.get(vital, (0.0, 0.0))
                comorbidity_adj[vital] = (old[0] + mean_adj, old[1] + std_adj)

    for i in range(n_observations):
        # Time progression: observations every 2-6 hours with jitter
        hours_offset = i * rng.uniform(2.0, 6.0)
        timestamp = base_time + pd.Timedelta(hours=hours_offset)

        progress = i / max(n_observations - 1, 1)

        if is_septic:
            # Slower, more gradual deterioration.  Max blend is capped to prevent
            # the vitals from becoming unambiguously septic late in the stay.
            if progress < 0.4:
                blend = progress / 0.4 * 0.25 * severity_jitter
            elif progress < 0.7:
                blend = (0.25 + (progress - 0.4) / 0.3 * 0.25) * severity_jitter
            else:
                blend = (0.50 + (progress - 0.7) / 0.3 * 0.20) * severity_jitter

            current_vitals: dict[str, Tuple[float, float]] = {}
            for vital in normal_vitals:
                normal_mean, normal_std = normal_vitals[vital]
                target_mean, target_std = target_vitals[vital]
                mean = normal_mean + blend * (target_mean - normal_mean)
                std = normal_std + blend * (target_std - normal_std)
                current_vitals[vital] = (mean, std)

            current_labs: dict[str, Tuple[float, float]] = {}
            for lab in normal_labs:
                nm, ns = normal_labs[lab]
                tm, ts = target_labs[lab]
                current_labs[lab] = (nm + blend * (tm - nm), ns + blend * (ts - ns))

        elif sick_type is not None:
            # Sick-but-not-septic: blend toward sick vitals quickly and reach high
            # so that they strongly resemble sepsis
            sick_blend = min(progress * 2.5, 0.9)  # Ramp up fast, max 0.9
            current_vitals = {}
            for vital in normal_vitals:
                nm, ns = normal_vitals[vital]
                tm, ts = target_vitals[vital]
                current_vitals[vital] = (nm + sick_blend * (tm - nm), ns + sick_blend * (ts - ns))

            current_labs = {}
            for lab in normal_labs:
                nm, ns = normal_labs[lab]
                tm, ts = target_labs[lab]
                current_labs[lab] = (nm + sick_blend * (tm - nm), ns + sick_blend * (ts - ns))
        else:
            current_vitals = dict(normal_vitals)
            current_labs = dict(normal_labs)

        # Generate vital values
        row: dict = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "age_years": age,
            "sex": sex,
            "ethnicity": ethnicity,
        }

        generated_sbp = None
        generated_dbp = None

        for vital in [
            "temperature", "heart_rate", "resp_rate",
            "sbp", "dbp", "spo2", "gcs", "map",
        ]:
            if vital == "map":
                if generated_sbp is not None and generated_dbp is not None:
                    value = (generated_sbp + 2.0 * generated_dbp) / 3.0
                else:
                    mean, std = current_vitals["map"]
                    value = rng.normal(mean, std)
                value = _clamp_vital(vital, value)
                value = _round_vital(vital, value)
                if np.isnan(value):
                    value = _round_vital(vital, current_vitals["map"][0])
                row[vital] = value
                continue

            mean, std = current_vitals[vital]

            if vital == "sbp":
                mean += sbp_eth_adj
            elif vital == "dbp":
                mean += dbp_eth_adj

            c_mean, c_std = comorbidity_adj.get(vital, (0.0, 0.0))
            mean += c_mean
            std = max(std + c_std, 0.5)

            # Temporal autocorrelation — real ICU data is noisy.
            # Low autocorrelation = more realistic stochastic variation.
            if (
                i > 0
                and vital in rows[-1]
                and not np.isnan(rows[-1].get(vital, np.nan))
            ):
                prev_val = rows[-1][vital]
                raw = rng.normal(mean, std)
                value = 0.7 * raw + 0.3 * prev_val
            else:
                value = rng.normal(mean, std)

            # Measurement noise — real bedside monitors have significant noise
            noise = rng.normal(0, std * 0.30)
            value += noise

            value = _clamp_vital(vital, value)
            value = _round_vital(vital, value)
            if np.isnan(value):
                value = _round_vital(vital, mean)

            row[vital] = value

            if vital == "sbp":
                generated_sbp = value
            elif vital == "dbp":
                generated_dbp = value

        # Generate lab values
        for lab in ["lactate", "wbc", "procalcitonin"]:
            lab_mean, lab_std = current_labs.get(lab, LAB_NORMALS[lab])

            # Add extra noise to labs (measured less precisely than vitals)
            value = rng.normal(lab_mean, lab_std)
            value += rng.normal(0, lab_std * 0.2)

            # Special: WBC can be LOW in severe sepsis (bimodal)
            if lab == "wbc" and is_septic and sepsis_severity == "severe":
                if rng.random() < 0.15:
                    value = rng.normal(2.5, 1.0)

            lo, hi = LAB_LIMITS.get(lab, (0, 1e6))
            value = float(np.clip(value, lo, hi))
            value = round(value, 2)
            row[lab] = value

        # Labs have higher missingness than vitals (~25%)
        # Real ICU data shows labs are often unavailable for many observations
        for lab in ["lactate", "wbc", "procalcitonin"]:
            if rng.random() < 0.25:
                row[lab] = np.nan

        # Vital missingness (~8% per vital for non-GCS)
        for vital in ["temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2", "map"]:
            if rng.random() < 0.08:
                row[vital] = np.nan

        if rng.random() < 0.02:
            row["gcs"] = np.nan

        # Label
        if is_septic:
            row["sepsis_label"] = 1 if progress >= sepsis_onset else 0
        else:
            row["sepsis_label"] = 0

        for comorb in ["hypertension", "diabetes", "ckd", "copd", "heart_failure"]:
            row[f"has_{comorb}"] = 1 if comorb in comorbidities else 0

        rows.append(row)

    return rows


def generate_dataset(
    n_patients: int = 10000,
    sepsis_prevalence: float = 0.20,
    sick_nonseptic_frac: float = 0.50,
    obs_per_patient: Tuple[int, int] = (6, 24),
    seed: int = 42,
    include_demographics: bool = True,
) -> pd.DataFrame:
    """Generate a full synthetic dataset for sepsis model training.

    The dataset has three patient categories:
    1. **Septic** (~20%): True sepsis with progressive deterioration
    2. **Sick-but-not-septic** (~35%): Confounders — patients with abnormal
       vitals from non-sepsis causes (post-surgical, dehydration, pain,
       COPD/HF exacerbation, viral infection). These are the key challenge
       cases that prevent unrealistically high AUROC.
    3. **Healthy** (~45%): Normal vitals with routine variation

    Parameters
    ----------
    n_patients : int
        Number of unique patients to generate.
    sepsis_prevalence : float
        Target fraction of septic patients.
    sick_nonseptic_frac : float
        Fraction of non-septic patients who are sick confounders.
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
        gcs, map, lactate, wbc, procalcitonin, sepsis_label,
        has_hypertension, has_diabetes, has_ckd, has_copd, has_heart_failure
    """
    rng = np.random.default_rng(seed)
    all_rows: list[dict] = []

    base_time = pd.Timestamp("2024-01-01 00:00:00")

    for pid in range(n_patients):
        patient_id = f"PT-{pid:06d}"

        # Demographics
        age = int(rng.integers(18, 95))
        sex = rng.choice(SEXES)
        ethnicity = rng.choice(ETHNICITIES, p=ETHNICITY_WEIGHTS)

        # NHANES age bucket for comorbidity prevalence lookup
        age_bucket = _get_nhanes_age_bucket(age)

        # Comorbidities -- prevalence drawn from NHANES age-stratified data
        comorbidities: list[str] = []
        for comorb, prev_by_age in NHANES_COMORBIDITY_PREVALENCE.items():
            prevalence = prev_by_age.get(age_bucket, 0.0)
            if rng.random() < prevalence:
                comorbidities.append(comorb)

        # Determine patient category: septic, sick-nonseptic, or healthy
        age_multiplier = _get_sepsis_risk_multiplier(age)
        sepsis_risk = sepsis_prevalence * age_multiplier

        if len(comorbidities) >= 2:
            sepsis_risk *= 1.3
        if len(comorbidities) >= 3:
            sepsis_risk *= 1.2

        is_septic = rng.random() < min(sepsis_risk, 0.7)

        # Sick-but-not-septic assignment for non-septic patients
        sick_type = None
        if not is_septic and rng.random() < sick_nonseptic_frac:
            # Bias confounders based on comorbidities
            if "copd" in comorbidities and rng.random() < 0.4:
                sick_type = "copd_exacerbation"
            elif "heart_failure" in comorbidities and rng.random() < 0.4:
                sick_type = "hf_exacerbation"
            else:
                sick_type = rng.choice(SICK_NONSEPTIC_TYPES)

        # Number of observations
        min_obs, max_obs = obs_per_patient
        n_obs = int(rng.integers(min_obs, max_obs + 1))

        # Sepsis severity distribution — 50% subtle creates realistic difficulty
        # Only 15% are severe (easy to classify), rest are subtle/early
        if is_septic:
            severity_roll = rng.random()
            if severity_roll < 0.50:
                severity = "subtle"
            elif severity_roll < 0.78:
                severity = "early"
            elif severity_roll < 0.93:
                severity = "severe"
            else:
                severity = "hypothermic"
        else:
            severity = "early"

        # Patient-specific time offset (different admission times)
        patient_base_time = base_time + pd.Timedelta(
            hours=int(rng.integers(0, 8760))
        )

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
            sick_type=sick_type,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Ensure correct dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sepsis_label"] = df["sepsis_label"].astype(int)

    # Label noise -- reflects real-world clinical uncertainty: retrospective
    # chart reviews disagree on sepsis diagnosis ~10-15% of the time
    # (Rhee et al., JAMA 2017).  Asymmetric noise:
    #   - 5% of positives flipped to 0 (missed diagnoses — more common)
    #   - 1% of negatives flipped to 1 (over-diagnosis — rare)
    positive_mask = df["sepsis_label"] == 1
    negative_mask = df["sepsis_label"] == 0
    miss_noise = positive_mask & (rng.random(len(df)) < 0.05)
    over_noise = negative_mask & (rng.random(len(df)) < 0.01)
    df.loc[miss_noise, "sepsis_label"] = 0
    df.loc[over_noise, "sepsis_label"] = 1

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
