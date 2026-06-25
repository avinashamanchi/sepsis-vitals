"""Sepsis-3 labeling module.

Implements the Sepsis-3 operational definition for clinical data labeling,
following Singer et al., "The Third International Consensus Definitions for
Sepsis and Septic Shock (Sepsis-3)", JAMA 2016;315(8):801-810.

Provides:
- Individual SOFA component scoring functions (pure Python, no pandas)
- Composite SOFA score computation on DataFrames
- Suspected infection detection from antibiotics + cultures
- Sepsis onset derivation (SOFA increase >= 2 near suspected infection)
- Per-observation binary labeling with optional ICD fallback
"""
from __future__ import annotations

from typing import Optional, Set

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MIMIC-IV item ID mappings
# ---------------------------------------------------------------------------
SOFA_LAB_ITEMS = {
    50912: "creatinine",
    52546: "creatinine",
    50885: "bilirubin_total",
    51265: "platelets",
}

PAO2_ITEM = 50821

FIO2_ITEMS = {223835: "fio2_chart", 50816: "fio2_lab"}

VASOPRESSOR_ITEMS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    222315: "vasopressin",
    221749: "phenylephrine",
}

ANTIBIOTIC_KEYWORDS = [
    "cillin",
    "mycin",
    "oxacin",
    "azole",
    "ceph",
    "meropenem",
    "vancomycin",
    "zosyn",
    "flagyl",
    "bactrim",
    "cipro",
    "levo",
    "metro",
    "sulbactam",
    "tazobactam",
    "azithro",
    "doxycycline",
    "trimethoprim",
    "clindamycin",
    "linezolid",
    "daptomycin",
    "ceftri",
    "cefaz",
    "cefep",
    "ampicillin",
    "amoxicillin",
    "ertapenem",
    "imipenem",
    "gentamicin",
    "tobramycin",
]

CULTURE_SPECIMENS = {
    "BLOOD CULTURE",
    "URINE",
    "SPUTUM",
    "BRONCHOALVEOLAR LAVAGE",
    "PERITONEAL FLUID",
    "ABSCESS",
    "TISSUE",
    "CSF;SPINAL FLUID",
    "PLEURAL FLUID",
    "JOINT FLUID",
}


# ---------------------------------------------------------------------------
# SOFA component scoring functions (pure Python)
# ---------------------------------------------------------------------------


def sofa_respiratory(pao2_fio2: Optional[float] = None) -> int:
    """Score the respiratory component of SOFA (PaO2/FiO2 ratio).

    0: >= 400, 1: 300-399, 2: 200-299, 3: 100-199, 4: < 100.
    None -> 0 (missing data assumed normal).
    """
    if pao2_fio2 is None:
        return 0
    if pao2_fio2 >= 400:
        return 0
    if pao2_fio2 >= 300:
        return 1
    if pao2_fio2 >= 200:
        return 2
    if pao2_fio2 >= 100:
        return 3
    return 4


def sofa_coagulation(platelets: Optional[float] = None) -> int:
    """Score the coagulation component of SOFA (platelet count x10^3/uL).

    0: >= 150, 1: 100-149, 2: 50-99, 3: 20-49, 4: < 20.
    None -> 0.
    """
    if platelets is None:
        return 0
    if platelets >= 150:
        return 0
    if platelets >= 100:
        return 1
    if platelets >= 50:
        return 2
    if platelets >= 20:
        return 3
    return 4


def sofa_liver(bilirubin: Optional[float] = None) -> int:
    """Score the liver component of SOFA (bilirubin mg/dL).

    0: < 1.2, 1: 1.2-1.9, 2: 2.0-5.9, 3: 6.0-11.9, 4: >= 12.0.
    None -> 0.
    """
    if bilirubin is None:
        return 0
    if bilirubin < 1.2:
        return 0
    if bilirubin < 2.0:
        return 1
    if bilirubin < 6.0:
        return 2
    if bilirubin < 12.0:
        return 3
    return 4


def sofa_cardiovascular(
    map_mmhg: Optional[float] = None,
    dopamine_rate: Optional[float] = None,
    norepinephrine_rate: Optional[float] = None,
    epinephrine_rate: Optional[float] = None,
    dobutamine_rate: Optional[float] = None,
) -> int:
    """Score the cardiovascular component of SOFA.

    0: MAP >= 70, 1: MAP < 70, 2: dopamine <= 5 or any dobutamine,
    3: dopamine > 5 or norepi/epi <= 0.1, 4: dopamine > 15 or norepi/epi > 0.1.
    None -> 0.
    """
    dopa = dopamine_rate or 0.0
    norepi = norepinephrine_rate or 0.0
    epi = epinephrine_rate or 0.0
    dobut = dobutamine_rate or 0.0

    # Score 4: high-dose vasopressors
    if dopa > 15 or norepi > 0.1 or epi > 0.1:
        return 4
    # Score 3: moderate vasopressors
    if dopa > 5 or norepi > 0 or epi > 0:
        return 3
    # Score 2: low-dose dopamine or any dobutamine
    if dopa > 0 or dobut > 0:
        return 2
    # Score 1: MAP < 70
    if map_mmhg is not None and map_mmhg < 70:
        return 1
    # Score 0: MAP >= 70 or no data
    return 0


def sofa_cns(gcs: Optional[int] = None) -> int:
    """Score the central nervous system component of SOFA (Glasgow Coma Scale).

    0: 15, 1: 13-14, 2: 10-12, 3: 6-9, 4: < 6.
    None -> 0.
    """
    if gcs is None:
        return 0
    if gcs >= 15:
        return 0
    if gcs >= 13:
        return 1
    if gcs >= 10:
        return 2
    if gcs >= 6:
        return 3
    return 4


def sofa_renal(creatinine: Optional[float] = None) -> int:
    """Score the renal component of SOFA (creatinine mg/dL).

    0: < 1.2, 1: 1.2-1.9, 2: 2.0-3.4, 3: 3.5-4.9, 4: >= 5.0.
    None -> 0.
    """
    if creatinine is None:
        return 0
    if creatinine < 1.2:
        return 0
    if creatinine < 2.0:
        return 1
    if creatinine < 3.5:
        return 2
    if creatinine < 5.0:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Composite SOFA score computation
# ---------------------------------------------------------------------------


def _safe_get(row: pd.Series, col: str) -> Optional[float]:
    """Get a value from a row, returning None if column missing or value is NaN."""
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return float(val)


def compute_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row SOFA component and total scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with optional columns: gcs, map, platelets,
        bilirubin_total, creatinine, pao2_fio2, dopamine_rate,
        norepinephrine_rate, epinephrine_rate, dobutamine_rate.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: sofa_resp, sofa_coag,
        sofa_liver, sofa_cardio, sofa_cns, sofa_renal, sofa_total.
    """
    result = df.copy()

    sofa_resp_vals = []
    sofa_coag_vals = []
    sofa_liver_vals = []
    sofa_cardio_vals = []
    sofa_cns_vals = []
    sofa_renal_vals = []

    for _, row in result.iterrows():
        sofa_resp_vals.append(sofa_respiratory(_safe_get(row, "pao2_fio2")))
        sofa_coag_vals.append(sofa_coagulation(_safe_get(row, "platelets")))
        sofa_liver_vals.append(sofa_liver(_safe_get(row, "bilirubin_total")))
        sofa_cardio_vals.append(
            sofa_cardiovascular(
                map_mmhg=_safe_get(row, "map"),
                dopamine_rate=_safe_get(row, "dopamine_rate"),
                norepinephrine_rate=_safe_get(row, "norepinephrine_rate"),
                epinephrine_rate=_safe_get(row, "epinephrine_rate"),
                dobutamine_rate=_safe_get(row, "dobutamine_rate"),
            )
        )
        sofa_cns_vals.append(sofa_cns(_safe_get(row, "gcs")))
        sofa_renal_vals.append(sofa_renal(_safe_get(row, "creatinine")))

    result["sofa_resp"] = sofa_resp_vals
    result["sofa_coag"] = sofa_coag_vals
    result["sofa_liver"] = sofa_liver_vals
    result["sofa_cardio"] = sofa_cardio_vals
    result["sofa_cns"] = sofa_cns_vals
    result["sofa_renal"] = sofa_renal_vals
    result["sofa_total"] = (
        result["sofa_resp"]
        + result["sofa_coag"]
        + result["sofa_liver"]
        + result["sofa_cardio"]
        + result["sofa_cns"]
        + result["sofa_renal"]
    )
    return result


# ---------------------------------------------------------------------------
# Suspected infection detection
# ---------------------------------------------------------------------------


def find_suspected_infections(
    antibiotics: pd.DataFrame,
    cultures: pd.DataFrame,
    window_hours: int = 72,
) -> pd.DataFrame:
    """Identify suspected infections from antibiotic + culture proximity.

    Parameters
    ----------
    antibiotics : pd.DataFrame
        Columns: subject_id, hadm_id, starttime, drug.
    cultures : pd.DataFrame
        Columns: subject_id, hadm_id, charttime, spec_type_desc.
    window_hours : int
        Maximum hours between antibiotic and culture to count as
        suspected infection. Default 72.

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, hadm_id, t_suspected_infection.
        One row per admission with a detected suspected infection.
        t_suspected_infection = min(abx_time, culture_time) for the
        closest pair within the window.
    """
    if antibiotics.empty or cultures.empty:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "t_suspected_infection"]
        )

    window = pd.Timedelta(hours=window_hours)
    results = []

    # Merge on admission
    merged = antibiotics.merge(
        cultures, on=["subject_id", "hadm_id"], how="inner"
    )

    if merged.empty:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "t_suspected_infection"]
        )

    # Calculate time difference
    merged["time_diff"] = (
        merged["starttime"] - merged["charttime"]
    ).abs()

    # Filter to within window
    merged = merged[merged["time_diff"] <= window]

    if merged.empty:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "t_suspected_infection"]
        )

    # t_suspected = min(abx_time, culture_time) for each pair
    merged["t_suspected_infection"] = merged[["starttime", "charttime"]].min(
        axis=1
    )

    # One infection per admission: take the earliest
    result = (
        merged.groupby(["subject_id", "hadm_id"])["t_suspected_infection"]
        .min()
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------------
# Sepsis onset derivation
# ---------------------------------------------------------------------------


def derive_sepsis_onset(
    sofa_series: pd.DataFrame,
    infections: pd.DataFrame,
    sofa_increase_threshold: int = 2,
    window_hours: int = 48,
) -> pd.DataFrame:
    """Derive Sepsis-3 onset times.

    For each admission with a suspected infection, check if SOFA increased
    by >= `sofa_increase_threshold` from baseline within +/- `window_hours`
    of the infection time. Baseline = minimum SOFA before infection.

    Parameters
    ----------
    sofa_series : pd.DataFrame
        Columns: subject_id, hadm_id, charttime, sofa_total.
    infections : pd.DataFrame
        Columns: subject_id, hadm_id, t_suspected_infection.
    sofa_increase_threshold : int
        Minimum SOFA increase to qualify as sepsis. Default 2.
    window_hours : int
        Window around infection time. Default 48.

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, hadm_id, t_sepsis_onset.
    """
    if infections.empty or sofa_series.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "t_sepsis_onset"])

    window = pd.Timedelta(hours=window_hours)
    results = []

    for _, inf_row in infections.iterrows():
        sid = inf_row["subject_id"]
        hadm = inf_row["hadm_id"]
        t_inf = inf_row["t_suspected_infection"]

        # Get SOFA measurements for this admission
        adm_sofa = sofa_series[
            (sofa_series["subject_id"] == sid)
            & (sofa_series["hadm_id"] == hadm)
        ].copy()

        if adm_sofa.empty:
            continue

        adm_sofa = adm_sofa.sort_values("charttime")

        # Baseline SOFA = minimum SOFA before infection
        pre_infection = adm_sofa[adm_sofa["charttime"] <= t_inf]
        if pre_infection.empty:
            baseline = 0
        else:
            baseline = pre_infection["sofa_total"].min()

        # Check for SOFA increase within window of infection
        window_sofa = adm_sofa[
            (adm_sofa["charttime"] >= t_inf - window)
            & (adm_sofa["charttime"] <= t_inf + window)
        ]

        if window_sofa.empty:
            continue

        max_sofa = window_sofa["sofa_total"].max()
        increase = max_sofa - baseline

        if increase >= sofa_increase_threshold:
            # Onset = earliest time SOFA crossed the threshold
            threshold_sofa = window_sofa[
                window_sofa["sofa_total"] >= baseline + sofa_increase_threshold
            ]
            if not threshold_sofa.empty:
                t_onset = threshold_sofa["charttime"].min()
                results.append(
                    {
                        "subject_id": sid,
                        "hadm_id": hadm,
                        "t_sepsis_onset": t_onset,
                    }
                )

    if not results:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "t_sepsis_onset"])

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Observation labeling
# ---------------------------------------------------------------------------


def label_observations(
    observations: pd.DataFrame,
    onsets: pd.DataFrame,
    icd_fallback_hadms: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """Assign per-observation binary sepsis labels.

    Parameters
    ----------
    observations : pd.DataFrame
        Must contain subject_id, hadm_id, charttime columns.
    onsets : pd.DataFrame
        Columns: subject_id, hadm_id, t_sepsis_onset.
    icd_fallback_hadms : Optional[Set[int]]
        Set of hadm_ids that have ICD sepsis codes. Admissions in this
        set that lack a Sepsis-3 onset get label=1, source='icd_fallback'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: sepsis_label (0/1),
        label_source ('sepsis3' or 'icd_fallback').
    """
    result = observations.copy()
    result["sepsis_label"] = 0
    result["label_source"] = "sepsis3"

    if icd_fallback_hadms is None:
        icd_fallback_hadms = set()

    # Build set of hadm_ids that have a Sepsis-3 onset
    sepsis3_hadms = set()

    if not onsets.empty:
        onset_map = onsets.set_index("hadm_id")["t_sepsis_onset"].to_dict()
        sepsis3_hadms = set(onset_map.keys())

        for hadm_id, t_onset in onset_map.items():
            mask = (result["hadm_id"] == hadm_id) & (
                result["charttime"] >= t_onset
            )
            result.loc[mask, "sepsis_label"] = 1
            result.loc[result["hadm_id"] == hadm_id, "label_source"] = "sepsis3"

    # ICD fallback: admissions with ICD codes but no Sepsis-3 onset
    fallback_hadms = icd_fallback_hadms - sepsis3_hadms
    if fallback_hadms:
        mask = result["hadm_id"].isin(fallback_hadms)
        result.loc[mask, "sepsis_label"] = 1
        result.loc[mask, "label_source"] = "icd_fallback"

    return result
