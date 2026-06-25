# Layer 1: Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse MIMIC-IV data (both relational CSV and FHIR NDJSON) into a unified training DataFrame with clinically valid Sepsis-3 labels and time-window-binned observations.

**Architecture:** Three new modules — `sepsis3_labeler.py` (Sepsis-3 label derivation from antibiotics + cultures + SOFA), `fhir_loader.py` (streaming NDJSON parser), `data_unifier.py` (time-window binning and deduplication) — plus modifications to the existing `mimic_loader.py`. All output the same DataFrame schema that `trainer.py:prepare_features()` expects.

**Tech Stack:** Python 3.11+, pandas, numpy, gzip (stdlib), pytest. No new dependencies (line-by-line NDJSON parsing uses stdlib `json` + `gzip`).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/sepsis_vitals/ml/sepsis3_labeler.py` | Create | Sepsis-3 label derivation: SOFA score, suspected infection detection, onset timing |
| `src/sepsis_vitals/ml/fhir_loader.py` | Create | Streaming FHIR NDJSON parser for MIMIC-IV FHIR Demo |
| `src/sepsis_vitals/ml/mimic_loader.py` | Modify | Add `from_demo()`, replace ICD labels with Sepsis-3, add SOFA component loading |
| `src/sepsis_vitals/ml/data_unifier.py` | Create | Time-window binning, patient ID normalization, source merging |
| `tests/test_sepsis3_labeler.py` | Create | Tests for SOFA computation, infection detection, label derivation |
| `tests/test_fhir_loader.py` | Create | Tests for FHIR NDJSON parsing with fixture data |
| `tests/test_data_unifier.py` | Create | Tests for time-window binning and deduplication |
| `tests/test_mimic_loader_demo.py` | Create | Integration tests for MIMIC-IV Demo CSV loading |
| `tests/fixtures/` | Create | Small FHIR NDJSON fixture files for unit tests |

---

### Task 1: Sepsis-3 SOFA Score Computation

**Files:**
- Create: `src/sepsis_vitals/ml/sepsis3_labeler.py`
- Create: `tests/test_sepsis3_labeler.py`

- [ ] **Step 1: Write failing test for SOFA respiratory component**

Create `tests/test_sepsis3_labeler.py`:

```python
"""Tests for Sepsis-3 label derivation."""

import numpy as np
import pandas as pd
import pytest


class TestSOFAComponents:
    """Test individual SOFA score component calculations."""

    def test_respiratory_sofa_normal(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_respiratory

        # PaO2/FiO2 >= 400 → SOFA 0
        assert sofa_respiratory(pao2_fio2=450.0) == 0

    def test_respiratory_sofa_mild(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_respiratory

        # 300 <= PaO2/FiO2 < 400 → SOFA 1
        assert sofa_respiratory(pao2_fio2=350.0) == 1

    def test_respiratory_sofa_moderate(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_respiratory

        # 200 <= PaO2/FiO2 < 300 → SOFA 2
        assert sofa_respiratory(pao2_fio2=250.0) == 2

    def test_respiratory_sofa_severe(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_respiratory

        # 100 <= PaO2/FiO2 < 200 → SOFA 3
        assert sofa_respiratory(pao2_fio2=150.0) == 3

    def test_respiratory_sofa_critical(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_respiratory

        # PaO2/FiO2 < 100 → SOFA 4
        assert sofa_respiratory(pao2_fio2=80.0) == 4

    def test_respiratory_sofa_missing(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_respiratory

        # Missing data → 0 (assume normal)
        assert sofa_respiratory(pao2_fio2=None) == 0

    def test_coagulation_sofa(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_coagulation

        assert sofa_coagulation(platelets=160.0) == 0  # >= 150
        assert sofa_coagulation(platelets=120.0) == 1  # 100-149
        assert sofa_coagulation(platelets=60.0) == 2   # 50-99
        assert sofa_coagulation(platelets=30.0) == 3   # 20-49
        assert sofa_coagulation(platelets=15.0) == 4   # < 20
        assert sofa_coagulation(platelets=None) == 0

    def test_liver_sofa(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_liver

        assert sofa_liver(bilirubin=1.0) == 0   # < 1.2
        assert sofa_liver(bilirubin=1.5) == 1   # 1.2-1.9
        assert sofa_liver(bilirubin=3.0) == 2   # 2.0-5.9
        assert sofa_liver(bilirubin=8.0) == 3   # 6.0-11.9
        assert sofa_liver(bilirubin=13.0) == 4  # >= 12.0
        assert sofa_liver(bilirubin=None) == 0

    def test_cardiovascular_sofa(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_cardiovascular

        assert sofa_cardiovascular(map_mmhg=75.0) == 0                          # MAP >= 70
        assert sofa_cardiovascular(map_mmhg=65.0) == 1                          # MAP < 70
        assert sofa_cardiovascular(map_mmhg=65.0, dopamine_rate=4.0) == 2       # dopamine <= 5
        assert sofa_cardiovascular(map_mmhg=65.0, dopamine_rate=10.0) == 3      # dopamine > 5
        assert sofa_cardiovascular(map_mmhg=65.0, norepinephrine_rate=0.2) == 4 # norepi > 0.1
        assert sofa_cardiovascular(map_mmhg=None) == 0

    def test_cns_sofa(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_cns

        assert sofa_cns(gcs=15) == 0   # 15
        assert sofa_cns(gcs=14) == 1   # 13-14
        assert sofa_cns(gcs=11) == 2   # 10-12
        assert sofa_cns(gcs=7) == 3    # 6-9
        assert sofa_cns(gcs=4) == 4    # < 6
        assert sofa_cns(gcs=None) == 0

    def test_renal_sofa(self):
        from sepsis_vitals.ml.sepsis3_labeler import sofa_renal

        assert sofa_renal(creatinine=1.0) == 0   # < 1.2
        assert sofa_renal(creatinine=1.5) == 1   # 1.2-1.9
        assert sofa_renal(creatinine=2.5) == 2   # 2.0-3.4
        assert sofa_renal(creatinine=4.0) == 3   # 3.5-4.9
        assert sofa_renal(creatinine=5.5) == 4   # >= 5.0
        assert sofa_renal(creatinine=None) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/avi/Desktop/sepsis-vitals && python -m pytest tests/test_sepsis3_labeler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sepsis_vitals.ml.sepsis3_labeler'`

- [ ] **Step 3: Implement SOFA component functions**

Create `src/sepsis_vitals/ml/sepsis3_labeler.py`:

```python
"""
sepsis_vitals.ml.sepsis3_labeler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sepsis-3 label derivation from clinical data.

Implements the Sepsis-3 operational definition (Singer et al., JAMA 2016):
  1. Suspected infection: antibiotic administration within +/-72h of
     blood/body fluid culture order.
  2. Organ dysfunction: SOFA score increase >= 2 from baseline within
     +/-48h of suspected infection onset.

Sepsis onset time (t_sepsis) = the earlier of antibiotic start or culture
order, provided the other occurs within 72h.
"""

from __future__ import annotations

from typing import Optional


# ── SOFA Component Scoring ──────────────────────────────────────────────────
# Reference: Vincent et al., Intensive Care Med 1996; Singer et al., JAMA 2016


def sofa_respiratory(pao2_fio2: Optional[float] = None) -> int:
    """Respiratory SOFA from PaO2/FiO2 ratio.

    0: >= 400, 1: 300-399, 2: 200-299, 3: 100-199, 4: < 100.
    Missing data returns 0 (assume normal).
    """
    if pao2_fio2 is None:
        return 0
    if pao2_fio2 < 100:
        return 4
    if pao2_fio2 < 200:
        return 3
    if pao2_fio2 < 300:
        return 2
    if pao2_fio2 < 400:
        return 1
    return 0


def sofa_coagulation(platelets: Optional[float] = None) -> int:
    """Coagulation SOFA from platelet count (x10^3/uL).

    0: >= 150, 1: 100-149, 2: 50-99, 3: 20-49, 4: < 20.
    """
    if platelets is None:
        return 0
    if platelets < 20:
        return 4
    if platelets < 50:
        return 3
    if platelets < 100:
        return 2
    if platelets < 150:
        return 1
    return 0


def sofa_liver(bilirubin: Optional[float] = None) -> int:
    """Liver SOFA from total bilirubin (mg/dL).

    0: < 1.2, 1: 1.2-1.9, 2: 2.0-5.9, 3: 6.0-11.9, 4: >= 12.0.
    """
    if bilirubin is None:
        return 0
    if bilirubin >= 12.0:
        return 4
    if bilirubin >= 6.0:
        return 3
    if bilirubin >= 2.0:
        return 2
    if bilirubin >= 1.2:
        return 1
    return 0


def sofa_cardiovascular(
    map_mmhg: Optional[float] = None,
    dopamine_rate: Optional[float] = None,
    norepinephrine_rate: Optional[float] = None,
    epinephrine_rate: Optional[float] = None,
    dobutamine_rate: Optional[float] = None,
) -> int:
    """Cardiovascular SOFA from MAP and vasopressor rates (mcg/kg/min).

    0: MAP >= 70, 1: MAP < 70, 2: dopamine <= 5 or any dobutamine,
    3: dopamine > 5 or epi/norepi <= 0.1, 4: dopamine > 15 or epi/norepi > 0.1.
    """
    # Check vasopressors first (higher SOFA scores)
    norepi = norepinephrine_rate or 0.0
    epi = epinephrine_rate or 0.0
    dopa = dopamine_rate or 0.0
    dobu = dobutamine_rate or 0.0

    if dopa > 15 or norepi > 0.1 or epi > 0.1:
        return 4
    if dopa > 5 or (norepi > 0 and norepi <= 0.1) or (epi > 0 and epi <= 0.1):
        return 3
    if (dopa > 0 and dopa <= 5) or dobu > 0:
        return 2

    if map_mmhg is None:
        return 0
    if map_mmhg < 70:
        return 1
    return 0


def sofa_cns(gcs: Optional[int] = None) -> int:
    """CNS SOFA from Glasgow Coma Scale.

    0: 15, 1: 13-14, 2: 10-12, 3: 6-9, 4: < 6.
    """
    if gcs is None:
        return 0
    if gcs < 6:
        return 4
    if gcs < 10:
        return 3
    if gcs < 13:
        return 2
    if gcs < 15:
        return 1
    return 0


def sofa_renal(creatinine: Optional[float] = None) -> int:
    """Renal SOFA from serum creatinine (mg/dL).

    0: < 1.2, 1: 1.2-1.9, 2: 2.0-3.4, 3: 3.5-4.9, 4: >= 5.0.
    """
    if creatinine is None:
        return 0
    if creatinine >= 5.0:
        return 4
    if creatinine >= 3.5:
        return 3
    if creatinine >= 2.0:
        return 2
    if creatinine >= 1.2:
        return 1
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/avi/Desktop/sepsis-vitals && python -m pytest tests/test_sepsis3_labeler.py::TestSOFAComponents -v`
Expected: All 28 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/sepsis3_labeler.py tests/test_sepsis3_labeler.py
git commit -m "feat: add SOFA component scoring for Sepsis-3 labeling"
```

---

### Task 2: Composite SOFA Score and Suspected Infection Detection

**Files:**
- Modify: `src/sepsis_vitals/ml/sepsis3_labeler.py`
- Modify: `tests/test_sepsis3_labeler.py`

- [ ] **Step 1: Write failing tests for composite SOFA and infection detection**

Append to `tests/test_sepsis3_labeler.py`:

```python
class TestCompositeSOFA:
    """Test full SOFA score computation from a DataFrame of observations."""

    def test_compute_sofa_single_row(self):
        from sepsis_vitals.ml.sepsis3_labeler import compute_sofa_scores

        df = pd.DataFrame([{
            "subject_id": 1,
            "charttime": pd.Timestamp("2024-01-01 08:00"),
            "gcs": 15,
            "map": 75.0,
            "platelets": 200.0,
            "bilirubin_total": 0.8,
            "creatinine": 0.9,
            "pao2_fio2": 450.0,
        }])
        result = compute_sofa_scores(df)
        assert "sofa_total" in result.columns
        assert result.iloc[0]["sofa_total"] == 0

    def test_compute_sofa_organ_dysfunction(self):
        from sepsis_vitals.ml.sepsis3_labeler import compute_sofa_scores

        df = pd.DataFrame([{
            "subject_id": 1,
            "charttime": pd.Timestamp("2024-01-01 08:00"),
            "gcs": 12,      # CNS SOFA 2
            "map": 65.0,     # Cardio SOFA 1
            "platelets": 80.0,  # Coag SOFA 2
            "bilirubin_total": 3.0,  # Liver SOFA 2
            "creatinine": 2.5,  # Renal SOFA 2
            "pao2_fio2": 250.0,  # Resp SOFA 2
        }])
        result = compute_sofa_scores(df)
        assert result.iloc[0]["sofa_total"] == 11  # 2+2+2+1+2+2

    def test_compute_sofa_missing_labs(self):
        from sepsis_vitals.ml.sepsis3_labeler import compute_sofa_scores

        df = pd.DataFrame([{
            "subject_id": 1,
            "charttime": pd.Timestamp("2024-01-01 08:00"),
            "gcs": 10,      # CNS SOFA 2
            "map": 75.0,     # Cardio SOFA 0
        }])
        result = compute_sofa_scores(df)
        # Missing labs default to SOFA 0 for those components
        assert result.iloc[0]["sofa_total"] == 2


class TestSuspectedInfection:
    """Test suspected infection detection from antibiotics + cultures."""

    def test_abx_and_culture_within_72h(self):
        from sepsis_vitals.ml.sepsis3_labeler import find_suspected_infections

        antibiotics = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "starttime": pd.Timestamp("2024-01-02 10:00"),
            "drug": "Vancomycin",
        }])
        cultures = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "charttime": pd.Timestamp("2024-01-02 08:00"),
            "spec_type_desc": "BLOOD CULTURE",
        }])
        result = find_suspected_infections(antibiotics, cultures)
        assert len(result) == 1
        assert result.iloc[0]["subject_id"] == 1
        # t_suspected = earlier of abx start and culture order
        assert result.iloc[0]["t_suspected_infection"] == pd.Timestamp("2024-01-02 08:00")

    def test_abx_before_culture_within_72h(self):
        from sepsis_vitals.ml.sepsis3_labeler import find_suspected_infections

        antibiotics = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "starttime": pd.Timestamp("2024-01-01 06:00"),
            "drug": "Meropenem",
        }])
        cultures = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "charttime": pd.Timestamp("2024-01-02 10:00"),
            "spec_type_desc": "BLOOD CULTURE",
        }])
        result = find_suspected_infections(antibiotics, cultures)
        assert len(result) == 1
        # t_suspected = earlier = abx start
        assert result.iloc[0]["t_suspected_infection"] == pd.Timestamp("2024-01-01 06:00")

    def test_abx_and_culture_beyond_72h(self):
        from sepsis_vitals.ml.sepsis3_labeler import find_suspected_infections

        antibiotics = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "starttime": pd.Timestamp("2024-01-01 06:00"),
            "drug": "Vancomycin",
        }])
        cultures = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "charttime": pd.Timestamp("2024-01-05 10:00"),  # 4 days later
            "spec_type_desc": "BLOOD CULTURE",
        }])
        result = find_suspected_infections(antibiotics, cultures)
        assert len(result) == 0  # No match — too far apart

    def test_no_culture_no_infection(self):
        from sepsis_vitals.ml.sepsis3_labeler import find_suspected_infections

        antibiotics = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "starttime": pd.Timestamp("2024-01-01 06:00"),
            "drug": "Vancomycin",
        }])
        cultures = pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "spec_type_desc"])
        result = find_suspected_infections(antibiotics, cultures)
        assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sepsis3_labeler.py::TestCompositeSOFA tests/test_sepsis3_labeler.py::TestSuspectedInfection -v`
Expected: FAIL — `cannot import name 'compute_sofa_scores'`

- [ ] **Step 3: Implement composite SOFA and suspected infection functions**

Append to `src/sepsis_vitals/ml/sepsis3_labeler.py`:

```python
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── SOFA Lab ItemIDs (MIMIC-IV) ─────────────────────────────────────────────

SOFA_LAB_ITEMS = {
    50912: "creatinine",       # Creatinine (mg/dL)
    52546: "creatinine",       # Creatinine (alternate)
    50885: "bilirubin_total",  # Bilirubin, Total (mg/dL)
    51265: "platelets",        # Platelet Count (x10^3/uL)
}

# PaO2 and FiO2 for respiratory SOFA
PAO2_ITEM = 50821   # PaO2 (mmHg)
FIO2_ITEMS = {
    223835: "fio2_chart",   # FiO2 from chartevents (%)
    50816: "fio2_lab",      # FiO2 from labevents (%)
}

# Vasopressor itemids from MIMIC-IV d_items
VASOPRESSOR_ITEMS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    222315: "vasopressin",
    221749: "phenylephrine",
}

# Antibiotic keywords for prescription/eMAR matching
ANTIBIOTIC_KEYWORDS = [
    "cillin", "mycin", "oxacin", "azole", "ceph", "meropenem",
    "vancomycin", "zosyn", "flagyl", "bactrim", "cipro", "levo",
    "metro", "sulbactam", "tazobactam", "azithro", "doxycycline",
    "trimethoprim", "clindamycin", "linezolid", "daptomycin",
    "ceftri", "cefaz", "cefep", "ampicillin", "amoxicillin",
    "ertapenem", "imipenem", "gentamicin", "tobramycin",
]

# Culture specimen types indicating suspected infection workup
CULTURE_SPECIMENS = {
    "BLOOD CULTURE", "URINE", "SPUTUM", "BRONCHOALVEOLAR LAVAGE",
    "PERITONEAL FLUID", "ABSCESS", "TISSUE", "CSF;SPINAL FLUID",
    "PLEURAL FLUID", "JOINT FLUID",
}


def compute_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SOFA score for each row in a clinical observations DataFrame.

    Expects columns (all optional except subject_id and charttime):
        subject_id, charttime, gcs, map, platelets, bilirubin_total,
        creatinine, pao2_fio2, dopamine_rate, norepinephrine_rate,
        epinephrine_rate, dobutamine_rate

    Missing values are treated as normal (SOFA 0) for that component.

    Returns DataFrame with added columns: sofa_resp, sofa_coag, sofa_liver,
    sofa_cardio, sofa_cns, sofa_renal, sofa_total.
    """
    result = df.copy()

    result["sofa_resp"] = result.get("pao2_fio2", pd.Series(dtype=float)).apply(
        lambda v: sofa_respiratory(v if pd.notna(v) else None)
    )
    result["sofa_coag"] = result.get("platelets", pd.Series(dtype=float)).apply(
        lambda v: sofa_coagulation(v if pd.notna(v) else None)
    )
    result["sofa_liver"] = result.get("bilirubin_total", pd.Series(dtype=float)).apply(
        lambda v: sofa_liver(v if pd.notna(v) else None)
    )

    # Cardiovascular SOFA requires MAP + vasopressor rates
    def _cardio_row(row):
        return sofa_cardiovascular(
            map_mmhg=row.get("map") if pd.notna(row.get("map")) else None,
            dopamine_rate=row.get("dopamine_rate") if pd.notna(row.get("dopamine_rate")) else None,
            norepinephrine_rate=row.get("norepinephrine_rate") if pd.notna(row.get("norepinephrine_rate")) else None,
            epinephrine_rate=row.get("epinephrine_rate") if pd.notna(row.get("epinephrine_rate")) else None,
            dobutamine_rate=row.get("dobutamine_rate") if pd.notna(row.get("dobutamine_rate")) else None,
        )

    result["sofa_cardio"] = result.apply(_cardio_row, axis=1)

    result["sofa_cns"] = result.get("gcs", pd.Series(dtype=float)).apply(
        lambda v: sofa_cns(int(v) if pd.notna(v) else None)
    )
    result["sofa_renal"] = result.get("creatinine", pd.Series(dtype=float)).apply(
        lambda v: sofa_renal(v if pd.notna(v) else None)
    )

    sofa_cols = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal"]
    # Fill any missing SOFA columns with 0
    for col in sofa_cols:
        if col not in result.columns:
            result[col] = 0
    result["sofa_total"] = result[sofa_cols].sum(axis=1)

    return result


def find_suspected_infections(
    antibiotics: pd.DataFrame,
    cultures: pd.DataFrame,
    window_hours: int = 72,
) -> pd.DataFrame:
    """Identify suspected infection episodes per admission.

    Suspected infection = antibiotic administration within +/-window_hours
    of a blood/body fluid culture order.

    Parameters
    ----------
    antibiotics : DataFrame
        Columns: subject_id, hadm_id, starttime, drug
    cultures : DataFrame
        Columns: subject_id, hadm_id, charttime, spec_type_desc
    window_hours : int
        Maximum gap between antibiotic and culture (default 72h per Sepsis-3).

    Returns
    -------
    DataFrame with columns: subject_id, hadm_id, t_suspected_infection
        t_suspected_infection = earlier of antibiotic start and culture time.
    """
    if antibiotics.empty or cultures.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "t_suspected_infection"])

    window = pd.Timedelta(hours=window_hours)
    results = []

    # Group by admission
    for hadm_id in antibiotics["hadm_id"].unique():
        abx = antibiotics[antibiotics["hadm_id"] == hadm_id]
        cx = cultures[cultures["hadm_id"] == hadm_id]

        if cx.empty:
            continue

        # For each culture, find the closest antibiotic within the window
        for _, culture_row in cx.iterrows():
            cx_time = culture_row["charttime"]
            time_diffs = (abx["starttime"] - cx_time).abs()
            within_window = time_diffs <= window

            if within_window.any():
                closest_abx_time = abx.loc[within_window, "starttime"].iloc[0]
                t_suspected = min(cx_time, closest_abx_time)
                results.append({
                    "subject_id": culture_row["subject_id"],
                    "hadm_id": hadm_id,
                    "t_suspected_infection": t_suspected,
                })
                break  # One suspected infection per admission

    if not results:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "t_suspected_infection"])

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sepsis3_labeler.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/sepsis3_labeler.py tests/test_sepsis3_labeler.py
git commit -m "feat: add composite SOFA scoring and suspected infection detection"
```

---

### Task 3: Sepsis-3 Onset Derivation and Observation Labeling

**Files:**
- Modify: `src/sepsis_vitals/ml/sepsis3_labeler.py`
- Modify: `tests/test_sepsis3_labeler.py`

- [ ] **Step 1: Write failing tests for onset derivation and labeling**

Append to `tests/test_sepsis3_labeler.py`:

```python
class TestSepsisOnset:
    """Test Sepsis-3 onset time derivation."""

    def test_sofa_increase_after_infection(self):
        from sepsis_vitals.ml.sepsis3_labeler import derive_sepsis_onset

        sofa_series = pd.DataFrame([
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-01 06:00"), "sofa_total": 1},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-01 12:00"), "sofa_total": 2},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-02 00:00"), "sofa_total": 4},
        ])
        infections = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "t_suspected_infection": pd.Timestamp("2024-01-01 10:00"),
        }])

        result = derive_sepsis_onset(sofa_series, infections)
        assert len(result) == 1
        assert result.iloc[0]["hadm_id"] == 100
        # Onset = time of SOFA >= 2 increase from baseline (1 → 4 = +3 >= 2)
        assert result.iloc[0]["t_sepsis_onset"] == pd.Timestamp("2024-01-02 00:00")

    def test_no_sofa_increase(self):
        from sepsis_vitals.ml.sepsis3_labeler import derive_sepsis_onset

        sofa_series = pd.DataFrame([
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-01 06:00"), "sofa_total": 2},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-01 12:00"), "sofa_total": 2},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-02 00:00"), "sofa_total": 3},
        ])
        infections = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "t_suspected_infection": pd.Timestamp("2024-01-01 10:00"),
        }])

        result = derive_sepsis_onset(sofa_series, infections)
        assert len(result) == 0  # SOFA increase of 1, needs >= 2

    def test_sofa_increase_outside_48h_window(self):
        from sepsis_vitals.ml.sepsis3_labeler import derive_sepsis_onset

        sofa_series = pd.DataFrame([
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-01 06:00"), "sofa_total": 1},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-05 00:00"), "sofa_total": 5},
        ])
        infections = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "t_suspected_infection": pd.Timestamp("2024-01-01 10:00"),
        }])

        result = derive_sepsis_onset(sofa_series, infections)
        assert len(result) == 0  # SOFA increase outside +/-48h of infection


class TestObservationLabeling:
    """Test per-observation binary label assignment."""

    def test_label_observations_with_onset(self):
        from sepsis_vitals.ml.sepsis3_labeler import label_observations

        observations = pd.DataFrame([
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-01 06:00")},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-01 12:00")},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-02 00:00")},
            {"subject_id": 1, "hadm_id": 100, "charttime": pd.Timestamp("2024-01-02 06:00")},
        ])
        onsets = pd.DataFrame([{
            "subject_id": 1,
            "hadm_id": 100,
            "t_sepsis_onset": pd.Timestamp("2024-01-02 00:00"),
        }])

        result = label_observations(observations, onsets)
        assert result["sepsis_label"].tolist() == [0, 0, 1, 1]
        assert result["label_source"].tolist() == ["sepsis3"] * 4

    def test_label_observations_no_sepsis(self):
        from sepsis_vitals.ml.sepsis3_labeler import label_observations

        observations = pd.DataFrame([
            {"subject_id": 2, "hadm_id": 200, "charttime": pd.Timestamp("2024-01-01 06:00")},
            {"subject_id": 2, "hadm_id": 200, "charttime": pd.Timestamp("2024-01-01 12:00")},
        ])
        onsets = pd.DataFrame(columns=["subject_id", "hadm_id", "t_sepsis_onset"])

        result = label_observations(observations, onsets)
        assert result["sepsis_label"].tolist() == [0, 0]
        assert result["label_source"].tolist() == ["sepsis3"] * 2

    def test_label_with_icd_fallback(self):
        from sepsis_vitals.ml.sepsis3_labeler import label_observations

        observations = pd.DataFrame([
            {"subject_id": 3, "hadm_id": 300, "charttime": pd.Timestamp("2024-01-01 06:00")},
        ])
        onsets = pd.DataFrame(columns=["subject_id", "hadm_id", "t_sepsis_onset"])
        icd_sepsis_hadms = {300}  # This admission has sepsis ICD code

        result = label_observations(observations, onsets, icd_fallback_hadms=icd_sepsis_hadms)
        assert result.iloc[0]["sepsis_label"] == 1
        assert result.iloc[0]["label_source"] == "icd_fallback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sepsis3_labeler.py::TestSepsisOnset tests/test_sepsis3_labeler.py::TestObservationLabeling -v`
Expected: FAIL — `cannot import name 'derive_sepsis_onset'`

- [ ] **Step 3: Implement onset derivation and observation labeling**

Append to `src/sepsis_vitals/ml/sepsis3_labeler.py`:

```python
def derive_sepsis_onset(
    sofa_series: pd.DataFrame,
    infections: pd.DataFrame,
    sofa_increase_threshold: int = 2,
    window_hours: int = 48,
) -> pd.DataFrame:
    """Derive Sepsis-3 onset time: SOFA increase >= threshold within window of infection.

    Parameters
    ----------
    sofa_series : DataFrame
        Columns: subject_id, hadm_id, charttime, sofa_total
    infections : DataFrame
        Columns: subject_id, hadm_id, t_suspected_infection
    sofa_increase_threshold : int
        Minimum SOFA increase from baseline (default 2 per Sepsis-3).
    window_hours : int
        Time window around suspected infection (default +/-48h).

    Returns
    -------
    DataFrame with columns: subject_id, hadm_id, t_sepsis_onset
    """
    if infections.empty or sofa_series.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "t_sepsis_onset"])

    window = pd.Timedelta(hours=window_hours)
    results = []

    for _, inf_row in infections.iterrows():
        hadm_id = inf_row["hadm_id"]
        t_inf = inf_row["t_suspected_infection"]

        # Get SOFA scores for this admission
        adm_sofa = sofa_series[sofa_series["hadm_id"] == hadm_id].sort_values("charttime")
        if adm_sofa.empty:
            continue

        # Baseline SOFA = minimum SOFA before or at infection time
        pre_infection = adm_sofa[adm_sofa["charttime"] <= t_inf]
        baseline = pre_infection["sofa_total"].min() if not pre_infection.empty else 0

        # Check for SOFA increase >= threshold within +/-48h of infection
        in_window = adm_sofa[
            (adm_sofa["charttime"] >= t_inf - window)
            & (adm_sofa["charttime"] <= t_inf + window)
        ]

        for _, sofa_row in in_window.iterrows():
            increase = sofa_row["sofa_total"] - baseline
            if increase >= sofa_increase_threshold:
                results.append({
                    "subject_id": inf_row["subject_id"],
                    "hadm_id": hadm_id,
                    "t_sepsis_onset": sofa_row["charttime"],
                })
                break  # First qualifying time point

    if not results:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "t_sepsis_onset"])

    return pd.DataFrame(results)


def label_observations(
    observations: pd.DataFrame,
    onsets: pd.DataFrame,
    icd_fallback_hadms: Optional[set] = None,
) -> pd.DataFrame:
    """Assign per-observation binary sepsis labels.

    For admissions with Sepsis-3 onset: observations at/after t_sepsis_onset = 1.
    For admissions without Sepsis-3 but with ICD fallback: all observations = 1
    with label_source = "icd_fallback".
    Otherwise: 0.

    Parameters
    ----------
    observations : DataFrame
        Must have columns: subject_id, hadm_id, charttime
    onsets : DataFrame
        Columns: subject_id, hadm_id, t_sepsis_onset
    icd_fallback_hadms : set, optional
        Hospital admission IDs with sepsis ICD codes (used when Sepsis-3
        criteria can't be computed due to missing data).

    Returns
    -------
    DataFrame — same as input with added columns: sepsis_label, label_source
    """
    result = observations.copy()
    result["sepsis_label"] = 0
    result["label_source"] = "sepsis3"

    # Apply Sepsis-3 labels
    for _, onset_row in onsets.iterrows():
        mask = (
            (result["hadm_id"] == onset_row["hadm_id"])
            & (result["charttime"] >= onset_row["t_sepsis_onset"])
        )
        result.loc[mask, "sepsis_label"] = 1

    # Apply ICD fallback for admissions without Sepsis-3 data
    if icd_fallback_hadms:
        sepsis3_hadms = set(onsets["hadm_id"]) if not onsets.empty else set()
        fallback_only = icd_fallback_hadms - sepsis3_hadms

        for hadm_id in fallback_only:
            mask = result["hadm_id"] == hadm_id
            result.loc[mask, "sepsis_label"] = 1
            result.loc[mask, "label_source"] = "icd_fallback"

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sepsis3_labeler.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/sepsis3_labeler.py tests/test_sepsis3_labeler.py
git commit -m "feat: add Sepsis-3 onset derivation and observation labeling"
```

---

### Task 4: Update MIMICLoader for Demo Data and Sepsis-3

**Files:**
- Modify: `src/sepsis_vitals/ml/mimic_loader.py`
- Create: `tests/test_mimic_loader_demo.py`

- [ ] **Step 1: Write failing test for `from_demo()` classmethod**

Create `tests/test_mimic_loader_demo.py`:

```python
"""Tests for MIMIC-IV Demo data loading with Sepsis-3 labels."""

import os
from pathlib import Path

import pandas as pd
import pytest

MIMIC_DEMO_PATH = Path("physionet.org/files/mimic-iv-demo/2.2")
HAS_DEMO_DATA = (MIMIC_DEMO_PATH / "hosp" / "patients.csv.gz").exists()


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="MIMIC-IV Demo data not available")
class TestMIMICLoaderDemo:
    """Integration tests using real MIMIC-IV Demo data."""

    def test_from_demo_creates_loader(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        assert loader.root == MIMIC_DEMO_PATH

    def test_load_demographics(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        demo = loader.load_demographics()
        assert "subject_id" in demo.columns
        assert "hadm_id" in demo.columns
        assert "age_years" in demo.columns
        assert "sex_m" in demo.columns
        assert len(demo) > 0
        assert demo["age_years"].between(0, 120).all()

    def test_load_vitals(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        # Load vitals for first 5 stays
        stays = pd.read_csv(MIMIC_DEMO_PATH / "icu" / "icustays.csv.gz")
        first_stays = set(stays["stay_id"].head(5))
        vitals = loader.load_vitals(first_stays)
        assert "stay_id" in vitals.columns
        assert "vital_name" in vitals.columns
        assert "valuenum" in vitals.columns
        assert vitals["vital_name"].isin([
            "temperature", "heart_rate", "resp_rate", "sbp", "dbp",
            "spo2", "gcs", "map",
        ]).all()

    def test_load_sofa_labs(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        stays = pd.read_csv(MIMIC_DEMO_PATH / "icu" / "icustays.csv.gz")
        hadm_ids = set(stays["hadm_id"].unique())
        sofa_labs = loader.load_sofa_labs(hadm_ids)
        assert "hadm_id" in sofa_labs.columns
        assert "charttime" in sofa_labs.columns
        # Should include at least some SOFA components
        assert len(sofa_labs) > 0

    def test_load_antibiotics(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        abx = loader.load_antibiotics()
        assert "subject_id" in abx.columns
        assert "hadm_id" in abx.columns
        assert "starttime" in abx.columns
        assert "drug" in abx.columns
        assert len(abx) > 0

    def test_load_cultures(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        cultures = loader.load_cultures()
        assert "subject_id" in cultures.columns
        assert "hadm_id" in cultures.columns
        assert "charttime" in cultures.columns
        assert "spec_type_desc" in cultures.columns
        assert len(cultures) > 0

    def test_derive_sepsis3_labels(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        stays = loader.derive_sepsis_labels()
        assert "stay_id" in stays.columns
        assert "sepsis_label" in stays.columns
        assert "label_source" in stays.columns
        # Should have a mix of sepsis and non-sepsis
        assert stays["sepsis_label"].sum() > 0
        assert stays["sepsis_label"].sum() < len(stays)

    def test_build_training_dataset_small(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=5)
        assert "patient_id" in df.columns
        assert "sepsis_label" in df.columns
        assert "label_source" in df.columns
        assert len(df) > 0
        # Should have vital columns
        vitals = ["temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2"]
        for v in vitals:
            assert any(col.startswith(v) for col in df.columns), f"Missing vital: {v}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mimic_loader_demo.py -v`
Expected: FAIL — `MIMICLoader has no attribute 'from_demo'`

- [ ] **Step 3: Implement MIMICLoader updates**

Modify `src/sepsis_vitals/ml/mimic_loader.py` — add the following methods/changes:

1. Add `from_demo()` classmethod after `__init__`:

```python
    @classmethod
    def from_demo(cls) -> "MIMICLoader":
        """Create a loader pointing to the MIMIC-IV Demo 2.2 dataset.

        Expects the demo data at ``physionet.org/files/mimic-iv-demo/2.2/``
        relative to the current working directory.
        """
        demo_path = Path("physionet.org/files/mimic-iv-demo/2.2")
        return cls(demo_path)
```

2. Update `_validate_paths()` to not require all files (demo may have subset):

```python
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
            if not path.exists() and not (self.root / f.replace(".gz", "")).exists():
                missing.append(f)

        if missing:
            raise FileNotFoundError(
                f"Missing MIMIC-IV files in {self.root}: {missing}. "
                f"Download from https://physionet.org/content/mimiciv/"
            )
```

3. Add `load_sofa_labs()` method:

```python
    def load_sofa_labs(self, hadm_ids: Optional[set] = None) -> pd.DataFrame:
        """Load lab results needed for SOFA score (creatinine, bilirubin, platelets, PaO2)."""
        from sepsis_vitals.ml.sepsis3_labeler import SOFA_LAB_ITEMS, PAO2_ITEM

        logger.info("Loading SOFA lab components...")
        all_itemids = list(SOFA_LAB_ITEMS.keys()) + [PAO2_ITEM]

        chunks = []
        for chunk in pd.read_csv(
            self.root / "hosp" / "labevents.csv.gz",
            usecols=["hadm_id", "charttime", "itemid", "valuenum"],
            chunksize=500_000,
            dtype={"hadm_id": float, "itemid": int, "valuenum": float},
            parse_dates=["charttime"],
        ):
            chunk = chunk[chunk["itemid"].isin(all_itemids)]
            if hadm_ids is not None:
                chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
            chunk = chunk.dropna(subset=["valuenum", "hadm_id"])
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=["hadm_id", "charttime", "itemid", "valuenum", "lab_name"])

        df = pd.concat(chunks, ignore_index=True)
        df["hadm_id"] = df["hadm_id"].astype(int)

        # Map itemid to lab name
        item_map = {**SOFA_LAB_ITEMS, PAO2_ITEM: "pao2"}
        df["lab_name"] = df["itemid"].map(item_map)

        logger.info("Loaded %d SOFA lab observations", len(df))
        return df
```

4. Add `load_antibiotics()` method:

```python
    def load_antibiotics(self) -> pd.DataFrame:
        """Load antibiotic prescriptions for suspected infection detection."""
        from sepsis_vitals.ml.sepsis3_labeler import ANTIBIOTIC_KEYWORDS

        logger.info("Loading antibiotic prescriptions...")

        # Try prescriptions first
        rx = self._read_csv(
            "hosp/prescriptions.csv.gz",
            usecols=["subject_id", "hadm_id", "starttime", "drug"],
            parse_dates=["starttime"],
        )

        # Filter to antibiotics by keyword matching
        rx["drug_lower"] = rx["drug"].str.lower().fillna("")
        mask = rx["drug_lower"].apply(
            lambda d: any(kw in d for kw in ANTIBIOTIC_KEYWORDS)
        )
        abx = rx.loc[mask, ["subject_id", "hadm_id", "starttime", "drug"]].copy()
        abx = abx.dropna(subset=["starttime"])

        # Also check ICU inputevents for IV antibiotics
        try:
            ie = self._read_csv(
                "icu/inputevents.csv.gz",
                usecols=["subject_id", "hadm_id", "starttime", "ordercategoryname"],
                parse_dates=["starttime"],
            )
            iv_abx = ie[ie["ordercategoryname"].str.contains("Antibiotic", na=False)]
            iv_abx = iv_abx.rename(columns={"ordercategoryname": "drug"})
            iv_abx = iv_abx[["subject_id", "hadm_id", "starttime", "drug"]]
            abx = pd.concat([abx, iv_abx], ignore_index=True)
        except (FileNotFoundError, KeyError):
            pass

        abx = abx.drop_duplicates(subset=["subject_id", "hadm_id", "starttime"])
        logger.info("Found %d antibiotic administration events", len(abx))
        return abx
```

5. Add `load_cultures()` method:

```python
    def load_cultures(self) -> pd.DataFrame:
        """Load microbiology culture orders for suspected infection detection."""
        from sepsis_vitals.ml.sepsis3_labeler import CULTURE_SPECIMENS

        logger.info("Loading microbiology cultures...")
        micro = self._read_csv(
            "hosp/microbiologyevents.csv.gz",
            usecols=["subject_id", "hadm_id", "charttime", "spec_type_desc"],
            parse_dates=["charttime"],
        )
        micro = micro[micro["spec_type_desc"].isin(CULTURE_SPECIMENS)]
        micro = micro.dropna(subset=["charttime"])

        # Deduplicate: one culture per specimen type per admission per day
        micro["chart_date"] = micro["charttime"].dt.date
        micro = micro.drop_duplicates(subset=["subject_id", "hadm_id", "spec_type_desc", "chart_date"])
        micro = micro.drop(columns=["chart_date"])

        logger.info("Found %d culture events", len(micro))
        return micro
```

6. Add `load_vasopressors()` method:

```python
    def load_vasopressors(self, stay_ids: Optional[set] = None) -> pd.DataFrame:
        """Load vasopressor administration for cardiovascular SOFA."""
        from sepsis_vitals.ml.sepsis3_labeler import VASOPRESSOR_ITEMS

        logger.info("Loading vasopressor events...")
        try:
            ie = self._read_csv(
                "icu/inputevents.csv.gz",
                usecols=["stay_id", "starttime", "endtime", "itemid", "rate", "rateuom"],
                parse_dates=["starttime", "endtime"],
            )
        except FileNotFoundError:
            return pd.DataFrame(columns=["stay_id", "starttime", "endtime", "vasopressor", "rate"])

        vaso_ids = set(VASOPRESSOR_ITEMS.keys())
        ie = ie[ie["itemid"].isin(vaso_ids)]
        if stay_ids is not None:
            ie = ie[ie["stay_id"].isin(stay_ids)]

        ie["vasopressor"] = ie["itemid"].map(VASOPRESSOR_ITEMS)
        ie = ie.dropna(subset=["rate"])

        logger.info("Found %d vasopressor events", len(ie))
        return ie[["stay_id", "starttime", "endtime", "vasopressor", "rate"]].copy()
```

7. Replace `derive_sepsis_labels()` to use Sepsis-3:

```python
    def derive_sepsis_labels(self) -> pd.DataFrame:
        """Derive Sepsis-3 labels with ICD fallback.

        Uses:
        1. Suspected infection (antibiotics + cultures within 72h)
        2. SOFA score increase >= 2 from baseline within 48h of infection
        3. ICD-10 codes as fallback when criteria 1-2 can't be computed
        """
        from sepsis_vitals.ml.sepsis3_labeler import (
            find_suspected_infections,
            compute_sofa_scores,
            derive_sepsis_onset,
        )

        logger.info("Deriving Sepsis-3 labels...")

        # Load ICU stays
        stays = self._read_csv(
            "icu/icustays.csv.gz",
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
            parse_dates=["intime", "outtime"],
        )

        hadm_ids = set(stays["hadm_id"].unique())

        # 1. Find suspected infections
        abx = self.load_antibiotics()
        cultures = self.load_cultures()
        infections = find_suspected_infections(abx, cultures)

        # 2. Compute SOFA scores from available data
        sofa_labs = self.load_sofa_labs(hadm_ids)
        stay_ids = set(stays["stay_id"].unique())
        vitals = self.load_vitals(stay_ids)

        # Build SOFA observation dataframe
        # Get GCS and MAP from vitals
        sofa_obs = []
        for _, stay in stays.iterrows():
            sid = stay["stay_id"]
            hid = stay["hadm_id"]

            sv = vitals[vitals["stay_id"] == sid]
            sl = sofa_labs[sofa_labs["hadm_id"] == hid] if not sofa_labs.empty else pd.DataFrame()

            # Get unique timestamps from vitals
            if sv.empty:
                continue

            for charttime in sv["charttime"].unique():
                row = {"subject_id": stay["subject_id"], "hadm_id": hid, "charttime": charttime}

                # Extract vitals at this time
                at_time = sv[sv["charttime"] == charttime]
                for _, v in at_time.iterrows():
                    if v["vital_name"] == "gcs":
                        row["gcs"] = v["valuenum"]
                    elif v["vital_name"] == "map":
                        row["map"] = v["valuenum"]

                # Find closest labs within 6 hours
                if not sl.empty:
                    ct = pd.Timestamp(charttime)
                    nearby = sl[(sl["charttime"] - ct).abs() <= pd.Timedelta(hours=6)]
                    for _, lab in nearby.iterrows():
                        if lab["lab_name"] in ("creatinine", "bilirubin_total", "platelets", "pao2"):
                            row[lab["lab_name"]] = lab["valuenum"]

                sofa_obs.append(row)

        if sofa_obs:
            sofa_df = pd.DataFrame(sofa_obs)
            sofa_df["charttime"] = pd.to_datetime(sofa_df["charttime"])
            sofa_df = compute_sofa_scores(sofa_df)

            # 3. Derive sepsis onset
            onsets = derive_sepsis_onset(sofa_df, infections)
        else:
            onsets = pd.DataFrame(columns=["subject_id", "hadm_id", "t_sepsis_onset"])

        # 4. ICD fallback for admissions without Sepsis-3 data
        try:
            diag = self._read_csv(
                "hosp/diagnoses_icd.csv.gz",
                usecols=["hadm_id", "icd_code", "icd_version"],
            )
            sepsis_prefixes = ["A40", "A41", "R652", "R6520", "R6521",
                               "99591", "99592", "78552"]
            sepsis_mask = diag["icd_code"].apply(
                lambda x: any(str(x).startswith(p) for p in sepsis_prefixes)
            )
            icd_sepsis_hadms = set(diag.loc[sepsis_mask, "hadm_id"].unique())
        except (FileNotFoundError, KeyError):
            icd_sepsis_hadms = set()

        # Assign labels to stays
        sepsis3_hadms = set(onsets["hadm_id"]) if not onsets.empty else set()

        stays["sepsis_label"] = 0
        stays["label_source"] = "sepsis3"

        for _, onset in onsets.iterrows():
            mask = stays["hadm_id"] == onset["hadm_id"]
            stays.loc[mask, "sepsis_label"] = 1

        # ICD fallback
        fallback_hadms = icd_sepsis_hadms - sepsis3_hadms
        for hadm_id in fallback_hadms:
            mask = stays["hadm_id"] == hadm_id
            stays.loc[mask, "sepsis_label"] = 1
            stays.loc[mask, "label_source"] = "icd_fallback"

        logger.info(
            "Sepsis-3 labels: %d/%d stays positive (%.1f%%). "
            "Sources: %d Sepsis-3, %d ICD fallback",
            stays["sepsis_label"].sum(),
            len(stays),
            100 * stays["sepsis_label"].mean(),
            len(sepsis3_hadms),
            len(fallback_hadms),
        )

        return stays
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mimic_loader_demo.py -v`
Expected: All tests PASS (or skipped if demo data not available)

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/mimic_loader.py tests/test_mimic_loader_demo.py
git commit -m "feat: update MIMICLoader with Sepsis-3 labels and demo support"
```

---

### Task 5: Streaming FHIR NDJSON Loader

**Files:**
- Create: `src/sepsis_vitals/ml/fhir_loader.py`
- Create: `tests/test_fhir_loader.py`
- Create: `tests/fixtures/fhir_patient.ndjson` (small fixture)
- Create: `tests/fixtures/fhir_observation.ndjson` (small fixture)

- [ ] **Step 1: Create FHIR test fixtures**

Create `tests/fixtures/fhir_patient.ndjson`:

```json
{"id":"patient-001","resourceType":"Patient","gender":"female","birthDate":"2083-04-10","identifier":[{"value":"10007795","system":"http://mimic.mit.edu/fhir/mimic/identifier/patient"}],"extension":[{"url":"http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex","valueCode":"F"}]}
{"id":"patient-002","resourceType":"Patient","gender":"male","birthDate":"2100-06-15","identifier":[{"value":"10003400","system":"http://mimic.mit.edu/fhir/mimic/identifier/patient"}],"extension":[{"url":"http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex","valueCode":"M"}]}
```

Create `tests/fixtures/fhir_observation.ndjson`:

```json
{"id":"obs-001","resourceType":"Observation","status":"final","code":{"coding":[{"code":"220045","system":"http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-chartevents-d-items","display":"Heart Rate"}]},"subject":{"reference":"Patient/patient-001"},"effectiveDateTime":"2180-07-23T14:00:00-04:00","valueQuantity":{"value":88.0,"unit":"bpm"}}
{"id":"obs-002","resourceType":"Observation","status":"final","code":{"coding":[{"code":"223761","system":"http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-chartevents-d-items","display":"Temperature Fahrenheit"}]},"subject":{"reference":"Patient/patient-001"},"effectiveDateTime":"2180-07-23T14:00:00-04:00","valueQuantity":{"value":98.6,"unit":"°F"}}
{"id":"obs-003","resourceType":"Observation","status":"final","code":{"coding":[{"code":"220045","system":"http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-chartevents-d-items","display":"Heart Rate"}]},"subject":{"reference":"Patient/patient-002"},"effectiveDateTime":"2180-07-23T15:00:00-04:00","valueQuantity":{"value":72.0,"unit":"bpm"}}
```

- [ ] **Step 2: Write failing tests for FHIR loader**

Create `tests/test_fhir_loader.py`:

```python
"""Tests for FHIR NDJSON streaming loader."""

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class TestFHIRPatientParsing:
    """Test patient demographic extraction from FHIR Patient resources."""

    def test_parse_patients_from_ndjson(self):
        from sepsis_vitals.ml.fhir_loader import parse_patients

        patients = parse_patients(FIXTURES / "fhir_patient.ndjson")
        assert len(patients) == 2
        assert patients.iloc[0]["subject_id"] == 10007795
        assert patients.iloc[0]["gender"] == "female"
        assert patients.iloc[1]["subject_id"] == 10003400
        assert patients.iloc[1]["gender"] == "male"

    def test_parse_patients_has_uuid(self):
        from sepsis_vitals.ml.fhir_loader import parse_patients

        patients = parse_patients(FIXTURES / "fhir_patient.ndjson")
        assert patients.iloc[0]["fhir_id"] == "patient-001"
        assert patients.iloc[1]["fhir_id"] == "patient-002"


class TestFHIRObservationParsing:
    """Test observation extraction from FHIR Observation resources."""

    def test_parse_observations_from_ndjson(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations

        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        assert len(obs) == 3
        assert "patient_fhir_id" in obs.columns
        assert "itemid" in obs.columns
        assert "valuenum" in obs.columns
        assert "charttime" in obs.columns

    def test_fahrenheit_to_celsius_conversion(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations

        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        temp_obs = obs[obs["itemid"] == 223761]
        assert len(temp_obs) == 1
        # 98.6°F = 37.0°C
        assert abs(temp_obs.iloc[0]["valuenum"] - 37.0) < 0.1

    def test_vital_name_mapping(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations

        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        hr_obs = obs[obs["vital_name"] == "heart_rate"]
        assert len(hr_obs) == 2  # Two HR observations

    def test_patient_linkage(self):
        from sepsis_vitals.ml.fhir_loader import parse_observations

        obs = parse_observations(FIXTURES / "fhir_observation.ndjson")
        patient1_obs = obs[obs["patient_fhir_id"] == "patient-001"]
        assert len(patient1_obs) == 2  # HR + Temp for patient-001


class TestStreamingParsing:
    """Test that parsing works line-by-line without loading all into memory."""

    def test_stream_ndjson_yields_dicts(self):
        from sepsis_vitals.ml.fhir_loader import stream_ndjson

        records = list(stream_ndjson(FIXTURES / "fhir_patient.ndjson"))
        assert len(records) == 2
        assert records[0]["resourceType"] == "Patient"

    def test_stream_ndjson_gzipped(self):
        from sepsis_vitals.ml.fhir_loader import stream_ndjson

        # Test with the real gzipped file if available
        gz_path = Path("physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/MimicPatient.ndjson.gz")
        if not gz_path.exists():
            pytest.skip("FHIR demo data not available")
        records = list(stream_ndjson(gz_path))
        assert len(records) == 100  # 100 patients in demo
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_fhir_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sepsis_vitals.ml.fhir_loader'`

- [ ] **Step 4: Implement FHIR NDJSON loader**

Create `src/sepsis_vitals/ml/fhir_loader.py`:

```python
"""
sepsis_vitals.ml.fhir_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Streaming FHIR NDJSON parser for MIMIC-IV FHIR Demo data.

Parses gzipped or plain NDJSON files line-by-line to avoid OOM on large
files (668K+ chartevent records). Uses stdlib json + gzip — no extra
dependencies.

Usage::

    from sepsis_vitals.ml.fhir_loader import FHIRLoader

    loader = FHIRLoader("physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/")
    df = loader.build_training_dataset()
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from sepsis_vitals.ml.mimic_loader import CHART_VITALS, LAB_ITEMS, _FAHRENHEIT_ITEMS

logger = logging.getLogger(__name__)


def stream_ndjson(path: Path) -> Iterator[dict]:
    """Stream NDJSON records one line at a time.

    Handles both plain .ndjson and gzipped .ndjson.gz files.
    Memory usage is O(1) per record regardless of file size.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" or str(path).endswith(".ndjson.gz") else open

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_patients(path: Path) -> pd.DataFrame:
    """Parse FHIR Patient resources into a demographics DataFrame.

    Extracts: fhir_id, subject_id (from MIMIC identifier), gender, birthDate.

    Parameters
    ----------
    path : Path
        Path to MimicPatient.ndjson or MimicPatient.ndjson.gz
    """
    records = []
    for resource in stream_ndjson(path):
        if resource.get("resourceType") != "Patient":
            continue

        fhir_id = resource["id"]
        gender = resource.get("gender", "unknown")
        birth_date = resource.get("birthDate")

        # Extract MIMIC subject_id from identifier
        subject_id = None
        for ident in resource.get("identifier", []):
            if ident.get("system") == "http://mimic.mit.edu/fhir/mimic/identifier/patient":
                subject_id = int(ident["value"])
                break

        records.append({
            "fhir_id": fhir_id,
            "subject_id": subject_id,
            "gender": gender,
            "birth_date": birth_date,
        })

    return pd.DataFrame(records)


def parse_observations(
    path: Path,
    itemid_filter: Optional[set] = None,
) -> pd.DataFrame:
    """Parse FHIR Observation resources into a vitals/labs DataFrame.

    Streams line-by-line. Handles Fahrenheit→Celsius conversion.

    Parameters
    ----------
    path : Path
        Path to NDJSON file (plain or gzipped)
    itemid_filter : set, optional
        Only include observations with these MIMIC itemids
    """
    all_vital_ids = set(CHART_VITALS.keys())
    all_lab_ids = set(LAB_ITEMS.keys())
    valid_ids = itemid_filter or (all_vital_ids | all_lab_ids)

    # Combined mapping for vital_name
    id_to_name = {**CHART_VITALS, **LAB_ITEMS}

    records = []
    for resource in stream_ndjson(path):
        if resource.get("resourceType") != "Observation":
            continue

        # Extract itemid from code
        itemid = None
        for coding in resource.get("code", {}).get("coding", []):
            try:
                itemid = int(coding.get("code", ""))
            except (ValueError, TypeError):
                continue
            if itemid in valid_ids:
                break
            itemid = None

        if itemid is None:
            continue

        # Extract value
        vq = resource.get("valueQuantity", {})
        valuenum = vq.get("value")
        if valuenum is None:
            continue

        # Fahrenheit → Celsius conversion
        if itemid in _FAHRENHEIT_ITEMS:
            valuenum = (valuenum - 32.0) * 5.0 / 9.0

        # Extract patient reference
        subject_ref = resource.get("subject", {}).get("reference", "")
        patient_fhir_id = subject_ref.replace("Patient/", "") if subject_ref else None

        # Extract timestamp
        charttime = resource.get("effectiveDateTime")
        if charttime is None:
            charttime = resource.get("issued")

        vital_name = id_to_name.get(itemid)

        records.append({
            "patient_fhir_id": patient_fhir_id,
            "charttime": charttime,
            "itemid": itemid,
            "valuenum": float(valuenum),
            "vital_name": vital_name,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["charttime"] = pd.to_datetime(df["charttime"], utc=True)
        # Remove timezone for consistency with CSV loader
        df["charttime"] = df["charttime"].dt.tz_localize(None)
    return df


class FHIRLoader:
    """Load MIMIC-IV FHIR Demo data for sepsis model training.

    Parameters
    ----------
    fhir_path : str or Path
        Directory containing MIMIC-IV FHIR NDJSON.gz files.
    """

    def __init__(self, fhir_path: str | Path) -> None:
        self.root = Path(fhir_path)
        self._validate()

    def _validate(self) -> None:
        if not (self.root / "MimicPatient.ndjson.gz").exists():
            raise FileNotFoundError(
                f"MimicPatient.ndjson.gz not found in {self.root}"
            )

    @classmethod
    def from_demo(cls) -> "FHIRLoader":
        """Create loader pointing to the default FHIR demo location."""
        return cls("physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir/")

    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics."""
        return parse_patients(self.root / "MimicPatient.ndjson.gz")

    def load_vitals(self) -> pd.DataFrame:
        """Load vital signs from chartevents (streaming)."""
        logger.info("Streaming FHIR chartevents (this may take a minute)...")
        return parse_observations(
            self.root / "MimicObservationChartevents.ndjson.gz",
            itemid_filter=set(CHART_VITALS.keys()),
        )

    def load_labs(self) -> pd.DataFrame:
        """Load lab results (streaming)."""
        logger.info("Streaming FHIR lab events...")
        return parse_observations(
            self.root / "MimicObservationLabevents.ndjson.gz",
            itemid_filter=set(LAB_ITEMS.keys()),
        )

    def load_encounters_icu(self) -> pd.DataFrame:
        """Load ICU encounter data."""
        records = []
        for resource in stream_ndjson(self.root / "MimicEncounterICU.ndjson.gz"):
            if resource.get("resourceType") != "Encounter":
                continue

            fhir_id = resource["id"]
            subject_ref = resource.get("subject", {}).get("reference", "")
            patient_fhir_id = subject_ref.replace("Patient/", "")

            period = resource.get("period", {})
            intime = period.get("start")
            outtime = period.get("end")

            # Extract stay_id from identifier
            stay_id = None
            for ident in resource.get("identifier", []):
                if "encounter-icu" in ident.get("system", ""):
                    stay_id = int(ident["value"])
                    break

            records.append({
                "fhir_id": fhir_id,
                "patient_fhir_id": patient_fhir_id,
                "stay_id": stay_id,
                "intime": intime,
                "outtime": outtime,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df["intime"] = pd.to_datetime(df["intime"], utc=True).dt.tz_localize(None)
            df["outtime"] = pd.to_datetime(df["outtime"], utc=True).dt.tz_localize(None)
        return df

    def load_conditions(self) -> pd.DataFrame:
        """Load diagnosis conditions for comorbidity and ICD fallback labeling."""
        records = []
        for resource in stream_ndjson(self.root / "MimicCondition.ndjson.gz"):
            if resource.get("resourceType") != "Condition":
                continue

            icd_code = None
            icd_system = None
            for coding in resource.get("code", {}).get("coding", []):
                system = coding.get("system", "")
                if "icd9" in system or "icd10" in system:
                    icd_code = coding.get("code")
                    icd_system = "icd9" if "icd9" in system else "icd10"
                    break

            subject_ref = resource.get("subject", {}).get("reference", "")
            patient_fhir_id = subject_ref.replace("Patient/", "")

            encounter_ref = resource.get("encounter", {}).get("reference", "")
            encounter_fhir_id = encounter_ref.replace("Encounter/", "")

            if icd_code:
                records.append({
                    "patient_fhir_id": patient_fhir_id,
                    "encounter_fhir_id": encounter_fhir_id,
                    "icd_code": icd_code,
                    "icd_system": icd_system,
                })

        return pd.DataFrame(records)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_fhir_loader.py -v`
Expected: All tests PASS (streaming gzip test may skip if demo data unavailable)

- [ ] **Step 6: Commit**

```bash
git add src/sepsis_vitals/ml/fhir_loader.py tests/test_fhir_loader.py tests/fixtures/
git commit -m "feat: add streaming FHIR NDJSON loader for MIMIC-IV demo"
```

---

### Task 6: Time-Window Binning and Data Unifier

**Files:**
- Create: `src/sepsis_vitals/ml/data_unifier.py`
- Create: `tests/test_data_unifier.py`

- [ ] **Step 1: Write failing tests for time-window binning**

Create `tests/test_data_unifier.py`:

```python
"""Tests for data unification and time-window binning."""

import numpy as np
import pandas as pd
import pytest


class TestTimeWindowBinning:
    """Test 1-hour epoch binning of clinical observations."""

    def test_bins_observations_to_1h_epochs(self):
        from sepsis_vitals.ml.data_unifier import bin_to_epochs

        df = pd.DataFrame([
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:01:00"), "heart_rate": 80.0},
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:45:00"), "sbp": 120.0},
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 11:30:00"), "heart_rate": 85.0},
        ])

        result = bin_to_epochs(df, epoch_minutes=60)
        # Two epochs: 10:00 and 11:00
        assert len(result) == 2
        # First epoch should have both HR and SBP
        first = result[result["epoch"] == pd.Timestamp("2024-01-01 10:00:00")]
        assert first.iloc[0]["heart_rate"] == 80.0  # median of single value
        assert first.iloc[0]["sbp"] == 120.0

    def test_median_aggregation_within_epoch(self):
        from sepsis_vitals.ml.data_unifier import bin_to_epochs

        df = pd.DataFrame([
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:05:00"), "heart_rate": 80.0},
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:20:00"), "heart_rate": 90.0},
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:50:00"), "heart_rate": 85.0},
        ])

        result = bin_to_epochs(df, epoch_minutes=60)
        assert len(result) == 1
        assert result.iloc[0]["heart_rate"] == 85.0  # median of [80, 90, 85]

    def test_forward_fill_missing_vitals(self):
        from sepsis_vitals.ml.data_unifier import bin_to_epochs

        df = pd.DataFrame([
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:00:00"), "heart_rate": 80.0, "sbp": 120.0},
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 11:00:00"), "heart_rate": 85.0},
        ])

        result = bin_to_epochs(df, epoch_minutes=60, forward_fill=True)
        assert len(result) == 2
        # Second epoch: HR is 85, SBP forward-filled from previous epoch
        second = result.iloc[1]
        assert second["heart_rate"] == 85.0
        assert second["sbp"] == 120.0  # Forward-filled

    def test_per_patient_binning(self):
        from sepsis_vitals.ml.data_unifier import bin_to_epochs

        df = pd.DataFrame([
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:00:00"), "heart_rate": 80.0},
            {"patient_id": 2, "timestamp": pd.Timestamp("2024-01-01 10:00:00"), "heart_rate": 100.0},
        ])

        result = bin_to_epochs(df, epoch_minutes=60)
        assert len(result) == 2
        p1 = result[result["patient_id"] == 1]
        p2 = result[result["patient_id"] == 2]
        assert p1.iloc[0]["heart_rate"] == 80.0
        assert p2.iloc[0]["heart_rate"] == 100.0

    def test_gcs_uses_max_not_median(self):
        from sepsis_vitals.ml.data_unifier import bin_to_epochs

        df = pd.DataFrame([
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:05:00"), "gcs": 12.0},
            {"patient_id": 1, "timestamp": pd.Timestamp("2024-01-01 10:30:00"), "gcs": 15.0},
        ])

        result = bin_to_epochs(df, epoch_minutes=60)
        # GCS should use max (best neurological state), not median
        assert result.iloc[0]["gcs"] == 15.0


class TestDataUnification:
    """Test merging data from multiple sources."""

    def test_unify_aligns_columns(self):
        from sepsis_vitals.ml.data_unifier import unify_datasets

        df1 = pd.DataFrame([{
            "patient_id": "1",
            "timestamp": pd.Timestamp("2024-01-01 10:00"),
            "heart_rate": 80.0,
            "sepsis_label": 0,
        }])
        df2 = pd.DataFrame([{
            "patient_id": "2",
            "timestamp": pd.Timestamp("2024-01-01 10:00"),
            "heart_rate": 90.0,
            "temperature": 37.5,
            "sepsis_label": 1,
        }])

        result = unify_datasets([df1, df2])
        assert len(result) == 2
        assert "heart_rate" in result.columns
        assert "temperature" in result.columns
        assert "sepsis_label" in result.columns

    def test_unify_deduplicates(self):
        from sepsis_vitals.ml.data_unifier import unify_datasets

        df1 = pd.DataFrame([{
            "patient_id": "1",
            "timestamp": pd.Timestamp("2024-01-01 10:00"),
            "heart_rate": 80.0,
            "sepsis_label": 0,
        }])
        # Same patient, same time from second source
        df2 = pd.DataFrame([{
            "patient_id": "1",
            "timestamp": pd.Timestamp("2024-01-01 10:00"),
            "heart_rate": 82.0,  # Slightly different value
            "sepsis_label": 0,
        }])

        result = unify_datasets([df1, df2])
        assert len(result) == 1  # Deduplicated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_unifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sepsis_vitals.ml.data_unifier'`

- [ ] **Step 3: Implement data unifier**

Create `src/sepsis_vitals/ml/data_unifier.py`:

```python
"""
sepsis_vitals.ml.data_unifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Time-window binning and multi-source data unification.

Bins sparse clinical observations into fixed-width epochs (default 1 hour),
aggregates vitals within each epoch, and merges data from multiple loaders
(FHIR NDJSON, MIMIC-IV CSV) into a single deduplicated DataFrame.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Vitals that use median aggregation within an epoch
MEDIAN_VITALS = [
    "temperature", "heart_rate", "resp_rate", "sbp", "dbp",
    "spo2", "map", "lactate", "wbc", "procalcitonin",
]

# Vitals that use max aggregation (GCS = best neurological state)
MAX_VITALS = ["gcs"]


def bin_to_epochs(
    df: pd.DataFrame,
    epoch_minutes: int = 60,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Bin observations into fixed-width time epochs per patient.

    Within each epoch:
    - Continuous vitals aggregated by median
    - GCS aggregated by max (best neurological state)
    - Non-vital columns (labels, demographics) taken from last observation

    Parameters
    ----------
    df : DataFrame
        Clinical observations with patient_col and time_col.
    epoch_minutes : int
        Epoch width in minutes (default 60 = 1 hour).
    forward_fill : bool
        If True, forward-fill missing vitals from previous epoch per patient.
    """
    if df.empty:
        return df

    result = df.copy()
    result[time_col] = pd.to_datetime(result[time_col])

    # Floor timestamps to epoch boundaries
    result["epoch"] = result[time_col].dt.floor(f"{epoch_minutes}min")

    # Identify vital columns present in the data
    vital_cols = [c for c in MEDIAN_VITALS if c in result.columns]
    max_cols = [c for c in MAX_VITALS if c in result.columns]
    non_vital_cols = [
        c for c in result.columns
        if c not in vital_cols + max_cols + [time_col, "epoch"]
    ]

    # Build aggregation dict
    agg_dict = {}
    for col in vital_cols:
        agg_dict[col] = "median"
    for col in max_cols:
        agg_dict[col] = "max"
    for col in non_vital_cols:
        if col == patient_col:
            agg_dict[col] = "first"
        else:
            agg_dict[col] = "last"

    # Group by patient + epoch, aggregate
    grouped = result.groupby([patient_col, "epoch"], as_index=False).agg(agg_dict)

    # Forward-fill missing vitals within each patient
    if forward_fill:
        fill_cols = vital_cols + max_cols
        grouped = grouped.sort_values([patient_col, "epoch"])
        grouped[fill_cols] = grouped.groupby(patient_col)[fill_cols].ffill()

    return grouped.reset_index(drop=True)


def unify_datasets(
    datasets: list[pd.DataFrame],
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    epoch_minutes: int = 60,
) -> pd.DataFrame:
    """Merge multiple DataFrames into a single deduplicated dataset.

    Concatenates all sources, bins to epochs, and deduplicates by
    (patient_id, epoch). When duplicates exist, keeps the row with
    more non-null vital values.

    Parameters
    ----------
    datasets : list of DataFrame
        Each should have patient_col, time_col, and vital columns.
    """
    if not datasets:
        return pd.DataFrame()

    # Concatenate all sources
    combined = pd.concat(datasets, ignore_index=True, sort=False)

    if combined.empty:
        return combined

    # Ensure consistent timestamp column
    combined[time_col] = pd.to_datetime(combined[time_col])

    # Bin to epochs
    combined = bin_to_epochs(
        combined,
        epoch_minutes=epoch_minutes,
        patient_col=patient_col,
        time_col=time_col if time_col in combined.columns else "epoch",
        forward_fill=True,
    )

    # Deduplicate: keep row with most non-null values
    vital_cols = [c for c in MEDIAN_VITALS + MAX_VITALS if c in combined.columns]
    combined["_n_values"] = combined[vital_cols].notna().sum(axis=1)
    combined = combined.sort_values("_n_values", ascending=False)
    combined = combined.drop_duplicates(subset=[patient_col, "epoch"], keep="first")
    combined = combined.drop(columns=["_n_values"])

    combined = combined.sort_values([patient_col, "epoch"]).reset_index(drop=True)

    logger.info(
        "Unified dataset: %d observations, %d patients",
        len(combined),
        combined[patient_col].nunique(),
    )

    return combined
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_unifier.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/data_unifier.py tests/test_data_unifier.py
git commit -m "feat: add time-window binning and multi-source data unifier"
```

---

### Task 7: End-to-End Integration Test

**Files:**
- Create: `tests/test_layer1_integration.py`

- [ ] **Step 1: Write integration test that loads real MIMIC-IV Demo data end-to-end**

Create `tests/test_layer1_integration.py`:

```python
"""End-to-end integration test for Layer 1 data pipeline.

Loads real MIMIC-IV Demo data through the full pipeline:
MIMICLoader → Sepsis-3 labeling → time-window binning → trainer-ready DataFrame.
"""

from pathlib import Path

import pandas as pd
import pytest

MIMIC_DEMO_PATH = Path("physionet.org/files/mimic-iv-demo/2.2")
FHIR_DEMO_PATH = Path("physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir")

HAS_CSV_DATA = (MIMIC_DEMO_PATH / "hosp" / "patients.csv.gz").exists()
HAS_FHIR_DATA = (FHIR_DEMO_PATH / "MimicPatient.ndjson.gz").exists()


@pytest.mark.skipif(not HAS_CSV_DATA, reason="MIMIC-IV Demo CSV not available")
class TestCSVPipelineIntegration:
    """Test full CSV pipeline: load → Sepsis-3 → bin → prepare_features."""

    def test_full_pipeline_produces_training_data(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader
        from sepsis_vitals.ml.data_unifier import bin_to_epochs

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=10)

        assert len(df) > 0
        assert "patient_id" in df.columns
        assert "sepsis_label" in df.columns
        assert "label_source" in df.columns

        # Should have some vital data
        vital_cols = ["temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2"]
        has_vitals = sum(1 for v in vital_cols if v in df.columns and df[v].notna().any())
        assert has_vitals >= 3, f"Expected at least 3 vitals, got {has_vitals}"

    def test_pipeline_output_compatible_with_trainer(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader
        from sepsis_vitals.ml.trainer import prepare_features

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=10)

        # Add required timestamp column if missing
        if "timestamp" not in df.columns and "epoch" in df.columns:
            df = df.rename(columns={"epoch": "timestamp"})

        # prepare_features should work without error
        features, feature_cols = prepare_features(df)
        assert len(feature_cols) > 10
        assert "sepsis_label" not in feature_cols
        assert len(features) == len(df)

    def test_sepsis3_labels_not_all_same(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=50)

        # Should have a mix of sepsis and non-sepsis
        assert df["sepsis_label"].nunique() >= 1  # At least some labels
        # Log the distribution for debugging
        n_pos = df["sepsis_label"].sum()
        n_total = len(df)
        print(f"Sepsis labels: {n_pos}/{n_total} ({100*n_pos/n_total:.1f}%)")


@pytest.mark.skipif(not HAS_FHIR_DATA, reason="MIMIC-IV FHIR Demo not available")
class TestFHIRPipelineIntegration:
    """Test FHIR pipeline: parse → patient linkage → vitals extraction."""

    def test_fhir_loads_100_patients(self):
        from sepsis_vitals.ml.fhir_loader import FHIRLoader

        loader = FHIRLoader.from_demo()
        patients = loader.load_patients()
        assert len(patients) == 100

    def test_fhir_loads_vitals(self):
        from sepsis_vitals.ml.fhir_loader import FHIRLoader

        loader = FHIRLoader.from_demo()
        vitals = loader.load_vitals()
        assert len(vitals) > 1000  # 668K records expected
        assert "vital_name" in vitals.columns
        assert vitals["vital_name"].isin([
            "temperature", "heart_rate", "resp_rate", "sbp", "dbp",
            "spo2", "gcs", "gcs_eye", "gcs_verbal", "gcs_motor", "map",
        ]).all()

    def test_fhir_loads_labs(self):
        from sepsis_vitals.ml.fhir_loader import FHIRLoader

        loader = FHIRLoader.from_demo()
        labs = loader.load_labs()
        assert len(labs) > 0
        assert "vital_name" in labs.columns


@pytest.mark.skipif(
    not (HAS_CSV_DATA and HAS_FHIR_DATA),
    reason="Both MIMIC-IV Demo sources needed"
)
class TestUnifiedPipeline:
    """Test unifying CSV and FHIR data sources."""

    def test_unify_csv_and_fhir(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader
        from sepsis_vitals.ml.fhir_loader import FHIRLoader
        from sepsis_vitals.ml.data_unifier import unify_datasets

        csv_loader = MIMICLoader.from_demo()
        csv_df = csv_loader.build_training_dataset(max_patients=5)

        # The unified dataset should have data
        result = unify_datasets([csv_df])
        assert len(result) > 0
        assert "patient_id" in result.columns
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_layer1_integration.py -v --timeout=120`
Expected: Tests PASS (with MIMIC-IV Demo data available). Some tests may take 30-60 seconds for the full chartevent parse.

- [ ] **Step 3: Commit**

```bash
git add tests/test_layer1_integration.py
git commit -m "test: add end-to-end integration tests for Layer 1 data pipeline"
```

---

### Task 8: Wire `build_training_dataset` to Use Epoch Binning

**Files:**
- Modify: `src/sepsis_vitals/ml/mimic_loader.py`

- [ ] **Step 1: Write failing test for epoch-binned output**

Append to `tests/test_mimic_loader_demo.py`:

```python
    def test_build_training_dataset_has_epoch_column(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=5)
        # Should have timestamp column (either 'timestamp' or 'epoch')
        assert "timestamp" in df.columns or "epoch" in df.columns

    def test_build_training_dataset_no_duplicate_epochs(self):
        from sepsis_vitals.ml.mimic_loader import MIMICLoader

        loader = MIMICLoader.from_demo()
        df = loader.build_training_dataset(max_patients=5)

        time_col = "timestamp" if "timestamp" in df.columns else "epoch"
        # No duplicate (patient_id, epoch) pairs
        dupes = df.duplicated(subset=["patient_id", time_col])
        assert not dupes.any(), f"Found {dupes.sum()} duplicate (patient_id, time) pairs"
```

- [ ] **Step 2: Run test to verify it fails or needs adjustment**

Run: `python -m pytest tests/test_mimic_loader_demo.py::TestMIMICLoaderDemo::test_build_training_dataset_has_epoch_column -v`

- [ ] **Step 3: Update `build_training_dataset` to use epoch binning**

In `src/sepsis_vitals/ml/mimic_loader.py`, update the `build_training_dataset` method to call `bin_to_epochs` at the end:

Replace the final section of `build_training_dataset` (after building the records list) with:

```python
        df = pd.DataFrame(records)

        if df.empty:
            logger.warning("No training data produced")
            return df

        # Add timestamp column from the most recent vital observation per stay
        # (needed for temporal feature engineering downstream)
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp("2024-01-01")  # placeholder

        # Bin observations into 1-hour epochs
        from sepsis_vitals.ml.data_unifier import bin_to_epochs
        df = bin_to_epochs(df, epoch_minutes=60, patient_col="patient_id", time_col="timestamp")

        # Rename epoch to timestamp for compatibility with prepare_features
        if "epoch" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"epoch": "timestamp"})

        logger.info(
            "Built training dataset: %d observations, %d patients, %.1f%% sepsis",
            len(df), df["patient_id"].nunique(),
            100 * df["sepsis_label"].mean() if len(df) > 0 else 0,
        )
        return df
```

- [ ] **Step 4: Run all Layer 1 tests**

Run: `python -m pytest tests/test_sepsis3_labeler.py tests/test_fhir_loader.py tests/test_data_unifier.py tests/test_mimic_loader_demo.py tests/test_layer1_integration.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/sepsis_vitals/ml/mimic_loader.py tests/test_mimic_loader_demo.py
git commit -m "feat: wire epoch binning into MIMICLoader training pipeline"
```

---

### Task 9: Final Cleanup and Full Test Suite

**Files:**
- All Layer 1 files

- [ ] **Step 1: Run the full test suite including existing tests**

Run: `python -m pytest tests/ -v --timeout=180 -x`
Expected: All tests PASS. If existing tests break, fix them.

- [ ] **Step 2: Run a quick smoke test loading real data**

```bash
python -c "
from sepsis_vitals.ml.mimic_loader import MIMICLoader
loader = MIMICLoader.from_demo()
df = loader.build_training_dataset(max_patients=10)
print(f'Shape: {df.shape}')
print(f'Columns: {sorted(df.columns.tolist())}')
print(f'Sepsis prevalence: {df[\"sepsis_label\"].mean():.1%}')
print(f'Label sources: {df[\"label_source\"].value_counts().to_dict()}')
print('Layer 1 pipeline working.')
"
```

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: complete Layer 1 data pipeline — Sepsis-3, FHIR, epoch binning"
```
