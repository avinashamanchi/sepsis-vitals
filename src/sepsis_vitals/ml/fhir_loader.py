"""
sepsis_vitals.ml.fhir_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Streaming FHIR NDJSON parser for MIMIC-IV FHIR Demo data.

Provides an alternative data source to the CSV-based MIMICLoader.
Both parse the same 100-patient MIMIC-IV Demo dataset, but this module
reads the FHIR (HL7) NDJSON export instead of flat CSV files.

Key design constraint: streaming line-by-line parsing with O(1) memory
per record.  The chartevent NDJSON file has ~668K records.

Usage::

    from sepsis_vitals.ml.fhir_loader import FHIRLoader

    loader = FHIRLoader.from_demo()
    patients = loader.load_patients()
    vitals   = loader.load_vitals()
    labs     = loader.load_labs()
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from sepsis_vitals.ml.mimic_loader import CHART_VITALS, LAB_ITEMS, _FAHRENHEIT_ITEMS

logger = logging.getLogger(__name__)

# Combine CHART_VITALS and LAB_ITEMS for unified itemid -> vital_name lookup
_ALL_ITEMS: dict[int, str] = {**CHART_VITALS, **LAB_ITEMS}

# MIMIC identifier system URI
_MIMIC_PATIENT_SYSTEM = "http://mimic.mit.edu/fhir/mimic/identifier/patient"
_MIMIC_STAY_SYSTEM = "http://mimic.mit.edu/fhir/mimic/identifier/encounter"


def stream_ndjson(path: str | Path) -> Iterator[dict]:
    """Yield parsed JSON dicts from an NDJSON file, line by line.

    Handles both plain ``.ndjson`` and gzipped ``.ndjson.gz`` files
    transparently.  Each line is parsed independently so memory usage
    is O(1) per record.

    Parameters
    ----------
    path : str or Path
        Path to the NDJSON (or NDJSON.GZ) file.

    Yields
    ------
    dict
        One parsed FHIR resource per line.
    """
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_patients(path: str | Path) -> pd.DataFrame:
    """Parse FHIR Patient resources from an NDJSON file.

    Extracts:
    - ``fhir_id``: FHIR resource id
    - ``subject_id``: MIMIC subject_id from identifier with the MIMIC
      patient system URI
    - ``gender``: administrative gender
    - ``birth_date``: date of birth string

    Parameters
    ----------
    path : str or Path
        Path to the Patient NDJSON file.

    Returns
    -------
    pd.DataFrame
    """
    records: list[dict] = []

    for resource in stream_ndjson(path):
        fhir_id = resource.get("id", "")
        gender = resource.get("gender", "")
        birth_date = resource.get("birthDate", "")

        # Extract MIMIC subject_id from identifiers
        subject_id: Optional[int] = None
        for ident in resource.get("identifier", []):
            if ident.get("system") == _MIMIC_PATIENT_SYSTEM:
                subject_id = int(ident["value"])
                break

        records.append(
            {
                "fhir_id": fhir_id,
                "subject_id": subject_id,
                "gender": gender,
                "birth_date": birth_date,
            }
        )

    df = pd.DataFrame(records)
    logger.info("Parsed %d FHIR Patient resources", len(df))
    return df


def parse_observations(
    path: str | Path,
    itemid_filter: Optional[set[int]] = None,
) -> pd.DataFrame:
    """Parse FHIR Observation resources from an NDJSON file.

    Extracts:
    - ``itemid``: MIMIC item ID from code.coding (int)
    - ``valuenum``: numeric value from valueQuantity.value
    - ``patient_fhir_id``: patient FHIR ID from subject.reference
      (``Patient/`` prefix stripped)
    - ``charttime``: effective date/time (converted to UTC, tz-naive)
    - ``vital_name``: mapped from itemid via CHART_VITALS / LAB_ITEMS

    Fahrenheit temperatures are converted to Celsius in-place.

    Parameters
    ----------
    path : str or Path
        Path to the Observation NDJSON file.
    itemid_filter : set of int, optional
        If provided, only keep observations whose itemid is in this set.

    Returns
    -------
    pd.DataFrame
    """
    records: list[dict] = []

    for resource in stream_ndjson(path):
        # Extract itemid from code.coding
        itemid: Optional[int] = None
        coding_list = resource.get("code", {}).get("coding", [])
        for coding in coding_list:
            code_val = coding.get("code")
            if code_val is not None:
                try:
                    itemid = int(code_val)
                except (ValueError, TypeError):
                    continue
                break

        if itemid is None:
            continue

        # Apply filter early to skip unwanted records
        if itemid_filter is not None and itemid not in itemid_filter:
            continue

        # Extract valuenum
        value_quantity = resource.get("valueQuantity", {})
        valuenum = value_quantity.get("value")
        if valuenum is None:
            continue

        # Extract patient FHIR ID from subject reference
        subject_ref = resource.get("subject", {}).get("reference", "")
        patient_fhir_id = subject_ref.replace("Patient/", "") if subject_ref else ""

        # Extract charttime
        charttime_str = resource.get("effectiveDateTime", "")

        records.append(
            {
                "itemid": itemid,
                "valuenum": float(valuenum),
                "patient_fhir_id": patient_fhir_id,
                "charttime": charttime_str,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["itemid", "valuenum", "patient_fhir_id", "charttime", "vital_name"]
        )

    df = pd.DataFrame(records)

    # Convert Fahrenheit to Celsius
    fahr_mask = df["itemid"].isin(_FAHRENHEIT_ITEMS)
    df.loc[fahr_mask, "valuenum"] = (df.loc[fahr_mask, "valuenum"] - 32) * 5.0 / 9.0

    # Map itemid -> vital_name
    df["vital_name"] = df["itemid"].map(_ALL_ITEMS)

    # Convert charttime to pandas datetime (UTC, then strip tz)
    df["charttime"] = pd.to_datetime(df["charttime"], utc=True)
    df["charttime"] = df["charttime"].dt.tz_localize(None)

    logger.info("Parsed %d FHIR Observation resources", len(df))
    return df


class FHIRLoader:
    """Load MIMIC-IV FHIR Demo data from NDJSON files.

    Parameters
    ----------
    fhir_path : str or Path
        Directory containing the FHIR NDJSON ``.ndjson.gz`` files
        (e.g. ``MimicPatient.ndjson.gz``).
    """

    def __init__(self, fhir_path: str | Path) -> None:
        self.root = Path(fhir_path)
        patient_file = self.root / "MimicPatient.ndjson.gz"
        if not patient_file.exists():
            raise FileNotFoundError(
                f"MimicPatient.ndjson.gz not found in {self.root}. "
                f"Download the MIMIC-IV FHIR Demo from "
                f"https://physionet.org/content/mimic-iv-fhir-demo/"
            )

    @classmethod
    def from_demo(cls) -> "FHIRLoader":
        """Create a loader pointing to the MIMIC-IV FHIR Demo 2.1.0 dataset."""
        demo_path = Path("physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/")
        return cls(demo_path)

    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics from MimicPatient.ndjson.gz."""
        return parse_patients(self.root / "MimicPatient.ndjson.gz")

    def load_vitals(self) -> pd.DataFrame:
        """Load vital signs from MimicObservationChartevents.ndjson.gz."""
        return parse_observations(
            self.root / "MimicObservationChartevents.ndjson.gz",
            itemid_filter=set(CHART_VITALS.keys()),
        )

    def load_labs(self) -> pd.DataFrame:
        """Load lab results from MimicObservationLabevents.ndjson.gz."""
        return parse_observations(
            self.root / "MimicObservationLabevents.ndjson.gz",
            itemid_filter=set(LAB_ITEMS.keys()),
        )

    def load_encounters_icu(self) -> pd.DataFrame:
        """Load ICU encounters from MimicEncounterICU.ndjson.gz.

        Extracts:
        - ``fhir_id``: FHIR resource id
        - ``patient_fhir_id``: patient FHIR ID from subject reference
        - ``stay_id``: MIMIC stay_id from identifier
        - ``intime``: period start
        - ``outtime``: period end
        """
        records: list[dict] = []

        for resource in stream_ndjson(self.root / "MimicEncounterICU.ndjson.gz"):
            fhir_id = resource.get("id", "")

            # Extract patient FHIR ID
            subject_ref = resource.get("subject", {}).get("reference", "")
            patient_fhir_id = subject_ref.replace("Patient/", "") if subject_ref else ""

            # Extract stay_id from identifier
            stay_id: Optional[int] = None
            for ident in resource.get("identifier", []):
                val = ident.get("value")
                if val is not None:
                    try:
                        stay_id = int(val)
                    except (ValueError, TypeError):
                        continue

            # Extract period
            period = resource.get("period", {})
            intime = period.get("start", "")
            outtime = period.get("end", "")

            records.append(
                {
                    "fhir_id": fhir_id,
                    "patient_fhir_id": patient_fhir_id,
                    "stay_id": stay_id,
                    "intime": intime,
                    "outtime": outtime,
                }
            )

        df = pd.DataFrame(records)

        if not df.empty:
            df["intime"] = pd.to_datetime(df["intime"], utc=True)
            df["intime"] = df["intime"].dt.tz_localize(None)
            df["outtime"] = pd.to_datetime(df["outtime"], utc=True)
            df["outtime"] = df["outtime"].dt.tz_localize(None)

        logger.info("Parsed %d FHIR EncounterICU resources", len(df))
        return df

    def load_conditions(self) -> pd.DataFrame:
        """Load conditions (diagnoses) from MimicCondition.ndjson.gz.

        Extracts:
        - ``icd_code``: ICD code from code.coding
        - ``icd_system``: coding system URI
        - ``patient_fhir_id``: patient FHIR ID from subject reference
        - ``encounter_fhir_id``: encounter FHIR ID from encounter reference
        """
        records: list[dict] = []

        for resource in stream_ndjson(self.root / "MimicCondition.ndjson.gz"):
            # Extract ICD code from code.coding
            coding_list = resource.get("code", {}).get("coding", [])
            for coding in coding_list:
                icd_code = coding.get("code", "")
                icd_system = coding.get("system", "")

                # Extract patient FHIR ID
                subject_ref = resource.get("subject", {}).get("reference", "")
                patient_fhir_id = (
                    subject_ref.replace("Patient/", "") if subject_ref else ""
                )

                # Extract encounter FHIR ID
                encounter_ref = resource.get("encounter", {}).get("reference", "")
                encounter_fhir_id = (
                    encounter_ref.replace("Encounter/", "") if encounter_ref else ""
                )

                records.append(
                    {
                        "icd_code": icd_code,
                        "icd_system": icd_system,
                        "patient_fhir_id": patient_fhir_id,
                        "encounter_fhir_id": encounter_fhir_id,
                    }
                )

        df = pd.DataFrame(records)
        logger.info("Parsed %d FHIR Condition resources", len(df))
        return df
