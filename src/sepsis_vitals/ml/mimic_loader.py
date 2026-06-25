"""
sepsis_vitals.ml.mimic_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MIMIC-IV data loader for clinical model validation.

Loads and preprocesses data from MIMIC-IV (v2.2+) for sepsis prediction
model training and validation.  Requires credentialed access to PhysioNet
(https://physionet.org/content/mimiciv/).

This module extracts:
- Vital signs from chartevents (temperature, HR, RR, SBP, DBP, SpO2, GCS, MAP)
- Lab results (lactate, WBC, procalcitonin)
- SOFA-relevant labs (creatinine, bilirubin, platelets, PaO2)
- Antibiotic prescriptions and IV antibiotics
- Microbiology cultures
- Vasopressor infusions
- Sepsis labels derived from Sepsis-3 criteria (suspected infection + SOFA >= 2)
  with ICD fallback for admissions lacking Sepsis-3 data
- Demographics (age, sex)
- Comorbidities via ICD-10 codes

Usage::

    from sepsis_vitals.ml.mimic_loader import MIMICLoader

    loader = MIMICLoader("/path/to/mimic-iv/")
    df = loader.build_training_dataset()
    # df is ready for trainer.py

    # Or use the MIMIC-IV Demo dataset:
    loader = MIMICLoader.from_demo()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -- MIMIC-IV itemid mappings (chartevents / labevents) --------------------
# These map MIMIC-IV d_items/d_labitems IDs to our internal feature names.

CHART_VITALS: dict[int, str] = {
    # Temperature (degC)
    223762: "temperature",  # "Temperature Celsius"
    223761: "temperature",  # "Temperature Fahrenheit" -> converted to degC
    # Heart rate
    220045: "heart_rate",
    # Respiratory rate
    220210: "resp_rate",
    224690: "resp_rate",
    # Blood pressure
    220179: "sbp",   # Non-invasive SBP
    220050: "sbp",   # Arterial SBP
    220180: "dbp",   # Non-invasive DBP
    220051: "dbp",   # Arterial DBP
    # SpO2
    220277: "spo2",
    # GCS components -> we sum them
    223901: "gcs_eye",
    223900: "gcs_verbal",
    220739: "gcs_motor",
    # MAP
    220052: "map",   # Arterial MAP
    220181: "map",   # Non-invasive MAP
}

LAB_ITEMS: dict[int, str] = {
    50813: "lactate",        # Lactate (blood gas)
    51265: "wbc",            # White blood cells
    # Procalcitonin -- MIMIC-IV does not consistently include PCT;
    # when present it uses the following itemid.
    50889: "procalcitonin",
}

# Fahrenheit itemids that need conversion
_FAHRENHEIT_ITEMS = {223761}

# ICD-10 prefixes for comorbidity extraction
COMORBIDITY_ICD10: dict[str, str] = {
    "has_hypertension": "I10",
    "has_diabetes": "E11",
    "has_ckd": "N18",
    "has_copd": "J44",
    "has_heart_failure": "I50",
}


class MIMICLoader:
    """Load and preprocess MIMIC-IV data for sepsis model training.

    Parameters
    ----------
    mimic_path : str or Path
        Root directory of the MIMIC-IV dataset (contains ``hosp/`` and
        ``icu/`` subdirectories with CSV or gzipped CSV files).
    """

    def __init__(self, mimic_path: str | Path) -> None:
        self.root = Path(mimic_path)
        self._validate_paths()

    @classmethod
    def from_demo(cls) -> "MIMICLoader":
        """Create a loader pointing to the MIMIC-IV Demo 2.2 dataset."""
        demo_path = Path("physionet.org/files/mimic-iv-demo/2.2")
        return cls(demo_path)

    def _validate_paths(self) -> None:
        """Check that required MIMIC-IV files exist."""
        required = [
            "hosp/patients.csv.gz",
            "hosp/admissions.csv.gz",
            "icu/icustays.csv.gz",
        ]
        missing = []
        for f in required:
            path = self.root / f
            # Also check without .gz
            if not path.exists() and not (self.root / f.replace(".gz", "")).exists():
                missing.append(f)

        if missing:
            raise FileNotFoundError(
                f"Missing MIMIC-IV files in {self.root}: {missing}. "
                f"Download from https://physionet.org/content/mimiciv/"
            )

    def _read_csv(self, relative_path: str, **kwargs) -> pd.DataFrame:
        """Read a MIMIC-IV CSV file (handles both .csv and .csv.gz)."""
        gz_path = self.root / relative_path
        plain_path = self.root / relative_path.replace(".gz", "")

        if gz_path.exists():
            return pd.read_csv(gz_path, **kwargs)
        elif plain_path.exists():
            return pd.read_csv(plain_path, **kwargs)
        else:
            raise FileNotFoundError(f"Cannot find {relative_path} in {self.root}")

    def load_demographics(self) -> pd.DataFrame:
        """Load patient demographics (age, sex)."""
        patients = self._read_csv(
            "hosp/patients.csv.gz",
            usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
        )
        admissions = self._read_csv(
            "hosp/admissions.csv.gz",
            usecols=["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"],
            parse_dates=["admittime", "dischtime"],
        )

        # Compute age at admission
        merged = admissions.merge(patients, on="subject_id")
        merged["age_years"] = merged["anchor_age"]

        # Binary sex encoding
        merged["sex_m"] = (merged["gender"] == "M").astype(int)

        return merged[["subject_id", "hadm_id", "age_years", "sex_m",
                        "admittime", "hospital_expire_flag"]].copy()

    def load_vitals(self, stay_ids: Optional[set] = None) -> pd.DataFrame:
        """Load vital signs from ICU chartevents.

        Parameters
        ----------
        stay_ids : set, optional
            If provided, only load vitals for these ICU stays.
        """
        logger.info("Loading chartevents (this may take several minutes)...")

        chart_itemids = list(CHART_VITALS.keys())

        # Read in chunks to manage memory
        chunks = []
        for chunk in pd.read_csv(
            self.root / "icu" / "chartevents.csv.gz",
            usecols=["stay_id", "charttime", "itemid", "valuenum"],
            chunksize=500_000,
            dtype={"stay_id": int, "itemid": int, "valuenum": float},
            parse_dates=["charttime"],
        ):
            chunk = chunk[chunk["itemid"].isin(chart_itemids)]
            if stay_ids is not None:
                chunk = chunk[chunk["stay_id"].isin(stay_ids)]
            chunk = chunk.dropna(subset=["valuenum"])
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

        # Map itemid -> vital name
        df["vital_name"] = df["itemid"].map(CHART_VITALS)

        # Convert Fahrenheit to Celsius
        fahr_mask = df["itemid"].isin(_FAHRENHEIT_ITEMS)
        df.loc[fahr_mask, "valuenum"] = (df.loc[fahr_mask, "valuenum"] - 32) * 5.0 / 9.0

        # Handle GCS: sum components per (stay_id, charttime)
        gcs_mask = df["vital_name"].isin({"gcs_eye", "gcs_verbal", "gcs_motor"})
        if gcs_mask.any():
            gcs = df[gcs_mask].copy()
            gcs_total = (
                gcs.pivot_table(
                    index=["stay_id", "charttime"],
                    columns="vital_name",
                    values="valuenum",
                    aggfunc="first",
                )
                .sum(axis=1)
                .reset_index(name="valuenum")
            )
            gcs_total["vital_name"] = "gcs"
            gcs_total["itemid"] = 0
            df = pd.concat([df[~gcs_mask], gcs_total], ignore_index=True)

        logger.info("Loaded %d vital sign observations", len(df))
        return df

    def load_labs(self, hadm_ids: Optional[set] = None) -> pd.DataFrame:
        """Load lab results (lactate, WBC, procalcitonin)."""
        logger.info("Loading lab events...")

        lab_itemids = list(LAB_ITEMS.keys())

        chunks = []
        for chunk in pd.read_csv(
            self.root / "hosp" / "labevents.csv.gz",
            usecols=["hadm_id", "charttime", "itemid", "valuenum"],
            chunksize=500_000,
            dtype={"hadm_id": float, "itemid": int, "valuenum": float},
            parse_dates=["charttime"],
        ):
            chunk = chunk[chunk["itemid"].isin(lab_itemids)]
            if hadm_ids is not None:
                chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
            chunk = chunk.dropna(subset=["valuenum", "hadm_id"])
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        df["vital_name"] = df["itemid"].map(LAB_ITEMS)
        df["hadm_id"] = df["hadm_id"].astype(int)

        logger.info("Loaded %d lab observations", len(df))
        return df

    def load_sofa_labs(self, hadm_ids: Optional[set] = None) -> pd.DataFrame:
        """Load SOFA-relevant labs: creatinine, bilirubin, platelets, PaO2.

        Reads labevents in chunks, filtering by SOFA_LAB_ITEMS and PAO2_ITEM
        from sepsis3_labeler.

        Parameters
        ----------
        hadm_ids : set, optional
            If provided, only load labs for these hospital admissions.

        Returns
        -------
        pd.DataFrame
            Columns: hadm_id, charttime, itemid, valuenum, lab_name.
        """
        from sepsis_vitals.ml.sepsis3_labeler import SOFA_LAB_ITEMS, PAO2_ITEM

        logger.info("Loading SOFA-relevant labs...")

        sofa_itemids = set(SOFA_LAB_ITEMS.keys()) | {PAO2_ITEM}
        item_map = {**SOFA_LAB_ITEMS, PAO2_ITEM: "pao2"}

        chunks = []
        for chunk in pd.read_csv(
            self.root / "hosp" / "labevents.csv.gz",
            usecols=["hadm_id", "charttime", "itemid", "valuenum"],
            chunksize=500_000,
            dtype={"hadm_id": float, "itemid": int, "valuenum": float},
            parse_dates=["charttime"],
        ):
            chunk = chunk[chunk["itemid"].isin(sofa_itemids)]
            if hadm_ids is not None:
                chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
            chunk = chunk.dropna(subset=["valuenum", "hadm_id"])
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(
                columns=["hadm_id", "charttime", "itemid", "valuenum", "lab_name"]
            )

        df = pd.concat(chunks, ignore_index=True)
        df["lab_name"] = df["itemid"].map(item_map)
        df["hadm_id"] = df["hadm_id"].astype(int)

        logger.info("Loaded %d SOFA lab observations", len(df))
        return df

    def load_antibiotics(self) -> pd.DataFrame:
        """Load antibiotic prescriptions and IV antibiotics.

        Combines:
        - hosp/prescriptions.csv.gz filtered by ANTIBIOTIC_KEYWORDS
        - icu/inputevents.csv.gz filtered by ordercategoryname containing
          "Antibiotic"

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, hadm_id, starttime, drug.
        """
        from sepsis_vitals.ml.sepsis3_labeler import ANTIBIOTIC_KEYWORDS

        logger.info("Loading antibiotic prescriptions...")

        # --- Prescriptions ---
        rx = self._read_csv(
            "hosp/prescriptions.csv.gz",
            usecols=["subject_id", "hadm_id", "starttime", "drug"],
        )
        rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
        rx = rx.dropna(subset=["drug", "starttime", "hadm_id"])

        # Filter by antibiotic keywords (case-insensitive)
        drug_lower = rx["drug"].str.lower()
        abx_mask = drug_lower.apply(
            lambda x: any(kw in x for kw in ANTIBIOTIC_KEYWORDS)
        )
        rx_abx = rx.loc[abx_mask, ["subject_id", "hadm_id", "starttime", "drug"]].copy()

        # --- IV antibiotics from inputevents ---
        iv_parts = []
        try:
            iv = self._read_csv(
                "icu/inputevents.csv.gz",
                usecols=["subject_id", "hadm_id", "starttime", "ordercategoryname"],
            )
            iv["starttime"] = pd.to_datetime(iv["starttime"], errors="coerce")
            iv_abx = iv[
                iv["ordercategoryname"].str.contains("Antibiotic", case=False, na=False)
            ].copy()
            iv_abx["drug"] = iv_abx["ordercategoryname"]
            iv_parts.append(
                iv_abx[["subject_id", "hadm_id", "starttime", "drug"]].copy()
            )
        except (FileNotFoundError, KeyError):
            logger.debug("inputevents not available for IV antibiotics")

        result = pd.concat([rx_abx] + iv_parts, ignore_index=True)
        result["hadm_id"] = result["hadm_id"].astype(int)

        logger.info("Loaded %d antibiotic administrations", len(result))
        return result

    def load_cultures(self) -> pd.DataFrame:
        """Load microbiology cultures.

        Reads hosp/microbiologyevents.csv.gz and filters by CULTURE_SPECIMENS.
        Deduplicates by (subject_id, hadm_id, spec_type_desc, chart_date).

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, hadm_id, charttime, spec_type_desc.
        """
        from sepsis_vitals.ml.sepsis3_labeler import CULTURE_SPECIMENS

        logger.info("Loading microbiology cultures...")

        micro = self._read_csv(
            "hosp/microbiologyevents.csv.gz",
            usecols=["subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc"],
        )
        micro = micro.dropna(subset=["hadm_id", "spec_type_desc"])

        # Filter to culture specimens
        micro = micro[micro["spec_type_desc"].isin(CULTURE_SPECIMENS)].copy()

        # Use charttime if available, fallback to chartdate
        micro["charttime"] = pd.to_datetime(micro["charttime"], errors="coerce")
        micro["chartdate"] = pd.to_datetime(micro["chartdate"], errors="coerce")
        micro["charttime"] = micro["charttime"].fillna(micro["chartdate"])
        micro = micro.dropna(subset=["charttime"])

        # Deduplicate by (subject_id, hadm_id, spec_type_desc, chart_date)
        micro["_chart_date"] = micro["charttime"].dt.date
        micro = micro.drop_duplicates(
            subset=["subject_id", "hadm_id", "spec_type_desc", "_chart_date"]
        )
        micro = micro.drop(columns=["_chart_date", "chartdate"])

        micro["hadm_id"] = micro["hadm_id"].astype(int)

        result = micro[["subject_id", "hadm_id", "charttime", "spec_type_desc"]].copy()
        logger.info("Loaded %d culture events", len(result))
        return result

    def load_vasopressors(self, stay_ids: Optional[set] = None) -> pd.DataFrame:
        """Load vasopressor infusions from icu/inputevents.csv.gz.

        Filters by VASOPRESSOR_ITEMS from sepsis3_labeler.

        Parameters
        ----------
        stay_ids : set, optional
            If provided, only load vasopressors for these ICU stays.

        Returns
        -------
        pd.DataFrame
            Columns: stay_id, starttime, endtime, vasopressor, rate.
        """
        from sepsis_vitals.ml.sepsis3_labeler import VASOPRESSOR_ITEMS

        logger.info("Loading vasopressor infusions...")

        iv = self._read_csv(
            "icu/inputevents.csv.gz",
            usecols=["stay_id", "starttime", "endtime", "itemid", "rate"],
        )
        iv["starttime"] = pd.to_datetime(iv["starttime"], errors="coerce")
        iv["endtime"] = pd.to_datetime(iv["endtime"], errors="coerce")

        vp = iv[iv["itemid"].isin(VASOPRESSOR_ITEMS.keys())].copy()
        if stay_ids is not None:
            vp = vp[vp["stay_id"].isin(stay_ids)]

        vp["vasopressor"] = vp["itemid"].map(VASOPRESSOR_ITEMS)
        vp = vp.dropna(subset=["starttime"])

        result = vp[["stay_id", "starttime", "endtime", "vasopressor", "rate"]].copy()
        logger.info("Loaded %d vasopressor infusion records", len(result))
        return result

    def load_comorbidities(self) -> pd.DataFrame:
        """Extract comorbidity flags from ICD-10 diagnosis codes."""
        diag = self._read_csv(
            "hosp/diagnoses_icd.csv.gz",
            usecols=["hadm_id", "icd_code", "icd_version"],
        )
        # Only ICD-10
        diag = diag[diag["icd_version"] == 10]

        result = diag[["hadm_id"]].drop_duplicates()
        for col_name, prefix in COMORBIDITY_ICD10.items():
            mask = diag["icd_code"].str.startswith(prefix)
            positive_hadms = diag.loc[mask, "hadm_id"].unique()
            result[col_name] = result["hadm_id"].isin(positive_hadms).astype(int)

        return result

    def derive_sepsis_labels(self) -> pd.DataFrame:
        """Derive Sepsis-3 labels: suspected infection + SOFA >= 2.

        Uses the full Sepsis-3 operational definition from Singer et al.,
        JAMA 2016:
        1. Find suspected infections (antibiotic + culture proximity)
        2. Compute SOFA scores from vitals, labs, and vasopressors
        3. Identify sepsis onset (SOFA increase >= 2 near infection)
        4. Fall back to ICD codes for admissions without Sepsis-3 data

        Returns a DataFrame with columns: subject_id, hadm_id, stay_id,
        intime, outtime, sepsis_label, label_source.
        """
        from sepsis_vitals.ml.sepsis3_labeler import (
            compute_sofa_scores,
            find_suspected_infections,
            derive_sepsis_onset,
        )

        logger.info("Deriving Sepsis-3 labels...")

        # Load ICU stays
        stays = self._read_csv(
            "icu/icustays.csv.gz",
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
            parse_dates=["intime", "outtime"],
        )

        hadm_id_set = set(stays["hadm_id"])
        stay_id_set = set(stays["stay_id"])

        # -- Step 1: Suspected infections ---
        antibiotics = self.load_antibiotics()
        cultures = self.load_cultures()
        infections = find_suspected_infections(antibiotics, cultures)

        # -- Step 2: Build SOFA observation DataFrame ---
        vitals = self.load_vitals(stay_id_set)
        sofa_labs = self.load_sofa_labs(hadm_id_set)
        vasopressors = self.load_vasopressors(stay_id_set)

        # Map stays to hadm_id for linking labs
        stay_hadm = stays.set_index("stay_id")["hadm_id"].to_dict()
        stay_subject = stays.set_index("stay_id")["subject_id"].to_dict()

        sofa_rows = []
        for sid in stay_id_set:
            hadm = stay_hadm.get(sid)
            subject = stay_subject.get(sid)
            if hadm is None or subject is None:
                continue

            sv = vitals[vitals["stay_id"] == sid]
            if sv.empty:
                continue

            # Get labs for this admission
            sl = sofa_labs[sofa_labs["hadm_id"] == hadm] if not sofa_labs.empty else pd.DataFrame()

            # Get vasopressors for this stay
            svp = vasopressors[vasopressors["stay_id"] == sid] if not vasopressors.empty else pd.DataFrame()

            # For each unique charttime in vitals, create a SOFA observation row
            unique_times = sv["charttime"].dropna().unique()
            for ct in unique_times:
                row: dict = {
                    "subject_id": subject,
                    "hadm_id": hadm,
                    "stay_id": sid,
                    "charttime": ct,
                }

                # Get GCS and MAP from vitals at this time
                vt = sv[sv["charttime"] == ct]
                gcs_vals = vt[vt["vital_name"] == "gcs"]["valuenum"]
                if not gcs_vals.empty:
                    row["gcs"] = gcs_vals.iloc[0]
                map_vals = vt[vt["vital_name"] == "map"]["valuenum"]
                if not map_vals.empty:
                    row["map"] = map_vals.iloc[0]

                # Find closest labs within 6 hours
                if not sl.empty:
                    ct_ts = pd.Timestamp(ct)
                    time_diffs = (sl["charttime"] - ct_ts).abs()
                    within_6h = sl[time_diffs <= pd.Timedelta(hours=6)]

                    if not within_6h.empty:
                        for lab_name in ["creatinine", "bilirubin_total", "platelets", "pao2"]:
                            lab_rows = within_6h[within_6h["lab_name"] == lab_name]
                            if not lab_rows.empty:
                                # Use closest measurement
                                closest_idx = (lab_rows["charttime"] - ct_ts).abs().idxmin()
                                row[lab_name] = lab_rows.loc[closest_idx, "valuenum"]

                # Check vasopressor rates at this time
                if not svp.empty:
                    ct_ts = pd.Timestamp(ct)
                    for _, vp_row in svp.iterrows():
                        vp_start = vp_row["starttime"]
                        vp_end = vp_row["endtime"]
                        if pd.notna(vp_start) and pd.notna(vp_end):
                            if vp_start <= ct_ts <= vp_end:
                                rate_col = f"{vp_row['vasopressor']}_rate"
                                rate_val = vp_row["rate"]
                                if pd.notna(rate_val):
                                    row[rate_col] = float(rate_val)

                sofa_rows.append(row)

        # Compute SOFA scores
        if sofa_rows:
            sofa_df = pd.DataFrame(sofa_rows)
            sofa_df["charttime"] = pd.to_datetime(sofa_df["charttime"])
            sofa_df = compute_sofa_scores(sofa_df)
        else:
            sofa_df = pd.DataFrame(
                columns=["subject_id", "hadm_id", "stay_id", "charttime", "sofa_total"]
            )

        # -- Step 3: Derive sepsis onset ---
        onsets = derive_sepsis_onset(sofa_df, infections)
        sepsis3_hadms = set(onsets["hadm_id"]) if not onsets.empty else set()

        # -- Step 4: ICD fallback ---
        diag = self._read_csv(
            "hosp/diagnoses_icd.csv.gz",
            usecols=["hadm_id", "icd_code", "icd_version"],
        )
        icd_sepsis_prefixes = ["A40", "A41", "R652", "R6520", "R6521", "99591", "99592", "78552"]
        icd_mask = diag["icd_code"].apply(
            lambda x: any(str(x).startswith(p) for p in icd_sepsis_prefixes)
        )
        icd_sepsis_hadms = set(diag.loc[icd_mask, "hadm_id"].unique())

        # Assign labels to stays
        stays["sepsis_label"] = 0
        stays["label_source"] = ""

        # Sepsis-3 positive
        s3_mask = stays["hadm_id"].isin(sepsis3_hadms)
        stays.loc[s3_mask, "sepsis_label"] = 1
        stays.loc[s3_mask, "label_source"] = "sepsis3"

        # ICD fallback: only for hadms not already covered by Sepsis-3
        fallback_hadms = icd_sepsis_hadms - sepsis3_hadms
        fb_mask = stays["hadm_id"].isin(fallback_hadms)
        stays.loc[fb_mask, "sepsis_label"] = 1
        stays.loc[fb_mask, "label_source"] = "icd_fallback"

        # Stays with neither Sepsis-3 nor ICD codes
        neither_mask = ~s3_mask & ~fb_mask
        stays.loc[neither_mask, "label_source"] = "sepsis3"

        n_positive = stays["sepsis_label"].sum()
        n_total = len(stays)
        n_s3 = s3_mask.sum()
        n_fb = fb_mask.sum()

        logger.info(
            "Sepsis-3 labels: %d/%d stays positive. Sources: %d Sepsis-3, %d ICD fallback",
            n_positive, n_total, n_s3, n_fb,
        )

        return stays

    def build_training_dataset(
        self,
        max_patients: Optional[int] = None,
        window_hours: int = 6,
    ) -> pd.DataFrame:
        """Build a complete training dataset from MIMIC-IV.

        Produces a DataFrame compatible with the existing trainer.py pipeline,
        with the same feature columns as the synthetic data generator.

        Parameters
        ----------
        max_patients : int, optional
            Limit the number of patients (for faster iteration).
        window_hours : int
            Size of the observation window for temporal features (default 6h).

        Returns
        -------
        pd.DataFrame
            Training-ready DataFrame with vitals, labs, scores, demographics,
            comorbidities, and sepsis labels.
        """
        # 1. Get ICU stays with sepsis labels
        stays = self.derive_sepsis_labels()
        if max_patients:
            stay_ids = stays["stay_id"].unique()[:max_patients]
            stays = stays[stays["stay_id"].isin(stay_ids)]

        stay_id_set = set(stays["stay_id"])
        hadm_id_set = set(stays["hadm_id"])

        # 2. Load demographics
        demo = self.load_demographics()
        stays = stays.merge(
            demo[["hadm_id", "age_years", "sex_m"]],
            on="hadm_id",
            how="left",
        )

        # 3. Load vitals and compute temporal features
        vitals = self.load_vitals(stay_id_set)

        # 4. Load labs
        labs = self.load_labs(hadm_id_set)

        # 5. Load comorbidities
        comorbidities = self.load_comorbidities()

        # 6. Build feature matrix per stay
        records = []
        vital_names = ["temperature", "heart_rate", "resp_rate", "sbp",
                       "dbp", "spo2", "gcs", "map"]
        lab_names = ["lactate", "wbc", "procalcitonin"]

        for _, stay in stays.iterrows():
            sid = stay["stay_id"]
            hid = stay["hadm_id"]
            intime = stay["intime"]

            # Get vitals for this stay
            sv = vitals[vitals["stay_id"] == sid].copy()
            if sv.empty:
                continue

            # Get labs for this admission
            sl = labs[labs["hadm_id"] == hid].copy() if not labs.empty else pd.DataFrame()

            row: dict = {
                "patient_id": sid,
                "age_years": stay.get("age_years", 65),
                "sepsis_label": stay["sepsis_label"],
                "label_source": stay.get("label_source", ""),
            }

            # Latest vitals and rolling stats
            for vname in vital_names:
                vdata = sv[sv["vital_name"] == vname]["valuenum"].values
                if len(vdata) > 0:
                    row[vname] = vdata[-1]
                    row[f"{vname}_roll_mean"] = float(np.mean(vdata[-6:]))
                    row[f"{vname}_roll_std"] = float(np.std(vdata[-6:], ddof=1)) if len(vdata) > 1 else 0.0
                    row[f"{vname}_delta"] = float(vdata[-1] - vdata[-2]) if len(vdata) >= 2 else 0.0
                    row[f"{vname}_missing"] = 0
                else:
                    row[f"{vname}_missing"] = 1

            # Labs
            for lname in lab_names:
                if not sl.empty:
                    ldata = sl[sl["vital_name"] == lname]["valuenum"].values
                else:
                    ldata = np.array([])
                if len(ldata) > 0:
                    row[lname] = ldata[-1]
                    row[f"{lname}_roll_mean"] = float(np.mean(ldata[-6:]))
                    row[f"{lname}_delta"] = float(ldata[-1] - ldata[-2]) if len(ldata) >= 2 else 0.0
                    row[f"{lname}_missing"] = 0
                else:
                    row[f"{lname}_missing"] = 1

            # Comorbidities
            comorb = comorbidities[comorbidities["hadm_id"] == hid]
            for col in COMORBIDITY_ICD10:
                row[col] = int(comorb[col].values[0]) if not comorb.empty and col in comorb.columns else 0

            # Missing counts
            row["n_vitals_missing"] = sum(1 for v in vital_names if row.get(f"{v}_missing", 1))
            row["n_labs_missing"] = sum(1 for v in lab_names if row.get(f"{v}_missing", 1))

            records.append(row)

        df = pd.DataFrame(records)
        logger.info(
            "Built training dataset: %d stays, %.1f%% sepsis",
            len(df), 100 * df["sepsis_label"].mean() if len(df) > 0 else 0,
        )
        return df
