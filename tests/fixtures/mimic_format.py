"""Tiny synthetic tables in MIMIC-IV *format* (no patient data).

Reproduces the file layout, column names and item IDs the loader reads, so CI
can exercise parsing, joins, timestamps, units and labelling without the
credentialed or demo datasets. Passing these tests says nothing about
behaviour on real MIMIC data.

Subject 1001 (stay 3001) is constructed to meet Sepsis-3: a blood culture at
T0, vancomycin two hours later, and creatinine 0.9 -> 3.6 mg/dL and platelets
250 -> 40 K/uL eight hours after T0 (SOFA rise of 6). Subject 1002 (stay 3002)
has normal values and no infection workup.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

T0 = pd.Timestamp("2150-03-01 12:00:00")  # MIMIC-style shifted dates


def _ts(hours: float) -> str:
    return (T0 + pd.Timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def _write(root: Path, rel: str, rows: list[dict]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


def write_mimic_fixture(root: Path) -> Path:
    _write(root, "hosp/patients.csv.gz", [
        {"subject_id": 1001, "gender": "F", "anchor_age": 67, "anchor_year": 2150, "anchor_year_group": "2017 - 2019", "dod": ""},
        {"subject_id": 1002, "gender": "M", "anchor_age": 45, "anchor_year": 2150, "anchor_year_group": "2017 - 2019", "dod": ""},
    ])
    _write(root, "hosp/admissions.csv.gz", [
        {"subject_id": 1001, "hadm_id": 2001, "admittime": _ts(-12), "dischtime": _ts(96), "hospital_expire_flag": 0},
        {"subject_id": 1002, "hadm_id": 2002, "admittime": _ts(-12), "dischtime": _ts(48), "hospital_expire_flag": 0},
    ])
    _write(root, "icu/icustays.csv.gz", [
        {"subject_id": 1001, "hadm_id": 2001, "stay_id": 3001, "intime": _ts(-6), "outtime": _ts(48), "los": 2.25},
        {"subject_id": 1002, "hadm_id": 2002, "stay_id": 3002, "intime": _ts(-6), "outtime": _ts(24), "los": 1.25},
    ])

    chart = []

    def vitals(stay: int, hours: float, hr: float, rr: float, sbp: float, dbp: float, mbp: float,
               spo2: float, temp: tuple[int, float], gcs: tuple | None) -> None:
        when = _ts(hours)
        for itemid, value in ((220045, hr), (220210, rr), (220179, sbp), (220180, dbp),
                              (220181, mbp), (220277, spo2), temp):
            chart.append({"subject_id": 0, "hadm_id": 0, "stay_id": stay, "charttime": when,
                          "itemid": itemid, "valuenum": value})
        if gcs:
            for itemid, value in gcs:
                chart.append({"subject_id": 0, "hadm_id": 0, "stay_id": stay, "charttime": when,
                              "itemid": itemid, "valuenum": value})

    full_gcs = ((220739, 4), (223900, 5), (223901, 6))       # eye, verbal, motor = 15
    no_verbal = ((220739, 3), (223901, 5))                   # intubated: verbal not charted
    # septic stay: deteriorating; temperature charted in Fahrenheit at +8 h
    vitals(3001, -4, 88, 16, 124, 78, 93, 97, (223762, 37.1), full_gcs)
    vitals(3001, 0, 104, 22, 108, 64, 79, 95, (223762, 38.2), full_gcs)
    vitals(3001, 4, 116, 26, 96, 56, 69, 93, (223762, 38.6), no_verbal)
    vitals(3001, 8, 124, 28, 88, 50, 63, 91, (223761, 101.3), full_gcs)
    vitals(3001, 12, 126, 30, 86, 48, 61, 90, (223762, 38.9), full_gcs)
    # control stay: stable
    for h in (-4, 0, 4, 8):
        vitals(3002, h, 76, 15, 128, 80, 96, 98, (223762, 36.8), full_gcs)
    for row in chart:
        row["subject_id"] = 1001 if row["stay_id"] == 3001 else 1002
        row["hadm_id"] = 2001 if row["stay_id"] == 3001 else 2002
    _write(root, "icu/chartevents.csv.gz", chart)

    labs = []

    def lab(hadm: int, hours: float, itemid: int, value: float) -> None:
        labs.append({"subject_id": hadm - 1000, "hadm_id": hadm, "charttime": _ts(hours),
                     "itemid": itemid, "valuenum": value})

    for hadm, late_cr, late_plt in ((2001, 3.6, 40), (2002, 0.8, 240)):
        lab(hadm, -4, 50912, 0.9)      # creatinine (SOFA)
        lab(hadm, 8, 50912, late_cr)
        lab(hadm, -4, 51265, 250)      # platelets (SOFA) - must never be read as WBC
        lab(hadm, 8, 51265, late_plt)
        lab(hadm, -4, 50885, 0.6)      # bilirubin (SOFA)
        lab(hadm, 0, 50813, 3.1 if hadm == 2001 else 1.0)   # lactate
        lab(hadm, 0, 51301, 14.2 if hadm == 2001 else 7.4)  # WBC, K/uL
        lab(hadm, 0, 50889, 180.0)     # C-reactive protein - must never be read as procalcitonin
    _write(root, "hosp/labevents.csv.gz", labs)

    _write(root, "hosp/prescriptions.csv.gz", [
        {"subject_id": 1001, "hadm_id": 2001, "starttime": _ts(2), "stoptime": _ts(72), "drug": "Vancomycin"},
        {"subject_id": 1002, "hadm_id": 2002, "starttime": _ts(1), "stoptime": _ts(2), "drug": "Acetaminophen"},
    ])
    _write(root, "hosp/microbiologyevents.csv.gz", [
        {"subject_id": 1001, "hadm_id": 2001, "chartdate": _ts(0)[:10], "charttime": _ts(0),
         "spec_type_desc": "BLOOD CULTURE"},
    ])
    _write(root, "icu/inputevents.csv.gz", [
        {"subject_id": 1001, "hadm_id": 2001, "stay_id": 3001, "starttime": _ts(10), "endtime": _ts(14),
         "itemid": 221906, "rate": 0.08, "ordercategoryname": "01-Drips"},
    ])
    _write(root, "hosp/diagnoses_icd.csv.gz", [
        {"subject_id": 1001, "hadm_id": 2001, "seq_num": 1, "icd_code": "I10", "icd_version": 10},
        {"subject_id": 1002, "hadm_id": 2002, "seq_num": 1, "icd_code": "E11", "icd_version": 10},
    ])
    return root
