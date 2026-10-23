"""
tests/test_mimic_itemids.py — MIMIC-IV item mappings must match the dictionaries.

Regression for mis-mapped labs: 51265 (Platelet Count) was loaded as WBC and
50889 (C-Reactive Protein) as procalcitonin; GCS eye/motor were swapped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sepsis_vitals.ml.mimic_loader import CHART_VITALS, LAB_ITEMS

DEMO = Path(__file__).resolve().parents[1] / "physionet.org" / "files" / "mimic-iv-demo" / "2.2"


def test_lab_mapping_has_no_known_mismaps():
    assert LAB_ITEMS.get(51301) == "wbc"
    assert 51265 not in LAB_ITEMS, "51265 is Platelet Count"
    assert 50889 not in LAB_ITEMS, "50889 is C-Reactive Protein"
    assert "procalcitonin" not in LAB_ITEMS.values(), "MIMIC-IV has no procalcitonin item"


def test_gcs_components_are_not_swapped():
    assert CHART_VITALS[220739] == "gcs_eye"
    assert CHART_VITALS[223900] == "gcs_verbal"
    assert CHART_VITALS[223901] == "gcs_motor"


@pytest.mark.skipif(not DEMO.exists(), reason="MIMIC-IV demo not downloaded")
def test_mappings_match_local_dictionaries():
    labs = pd.read_csv(DEMO / "hosp" / "d_labitems.csv.gz").set_index("itemid")["label"]
    items = pd.read_csv(DEMO / "icu" / "d_items.csv.gz").set_index("itemid")["label"]
    expected_lab = {"lactate": "lactate", "wbc": ("white blood cells", "wbc count")}
    for itemid, name in LAB_ITEMS.items():
        label = labs[itemid].lower()
        allowed = expected_lab[name]
        assert label in allowed if isinstance(allowed, tuple) else label == allowed, (itemid, label)
    for itemid, name in CHART_VITALS.items():
        if name.startswith("gcs_"):
            part = {"gcs_eye": "eye", "gcs_verbal": "verbal", "gcs_motor": "motor"}[name]
            assert part in items[itemid].lower(), (itemid, items[itemid])
