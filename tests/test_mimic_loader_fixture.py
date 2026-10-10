"""
tests/test_mimic_loader_fixture.py — MIMIC loader coverage that runs in CI.

Uses tiny synthetic tables in MIMIC-IV format (tests/fixtures/mimic_format.py),
so ordinary CI exercises parsing, joins, timestamps, units, item mappings and
Sepsis-3 labelling. Real-dataset integration tests stay in
test_mimic_loader_demo.py, gated on the locally downloaded demo. These tests
do not show that the loader is correct on real MIMIC data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.fixtures.mimic_format import T0, write_mimic_fixture


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    from sepsis_vitals.ml.mimic_loader import MIMICLoader

    return MIMICLoader(write_mimic_fixture(tmp_path_factory.mktemp("mimic")))


def test_missing_required_tables_are_reported(tmp_path):
    from sepsis_vitals.ml.mimic_loader import MIMICLoader

    (tmp_path / "hosp").mkdir()
    with pytest.raises(FileNotFoundError, match="patients.csv.gz"):
        MIMICLoader(tmp_path)


def test_fahrenheit_is_converted_and_units_are_plausible(loader):
    vitals = loader.load_vitals({3001})
    temps = vitals[vitals["vital_name"] == "temperature"].set_index("charttime")["valuenum"]
    assert temps.loc[T0 + pd.Timedelta(hours=8)] == pytest.approx((101.3 - 32) * 5 / 9, abs=0.05)
    assert temps.between(30, 45).all()


def test_gcs_total_requires_all_three_components(loader):
    vitals = loader.load_vitals({3001})
    gcs = vitals[vitals["vital_name"] == "gcs"].set_index("charttime")["valuenum"]
    assert gcs.loc[T0 - pd.Timedelta(hours=4)] == 15
    assert (T0 + pd.Timedelta(hours=4)) not in gcs.index  # verbal missing: no falsely low total
    assert not vitals["vital_name"].isin({"gcs_eye", "gcs_verbal", "gcs_motor"}).any()


def test_labs_use_correct_items(loader):
    labs = loader.load_labs({2001, 2002})
    names = set(labs["vital_name"])
    assert "procalcitonin" not in names  # CRP (50889) must not appear as procalcitonin
    wbc = labs[labs["vital_name"] == "wbc"]["valuenum"]
    assert sorted(wbc) == [7.4, 14.2]     # 51301 only; platelets (250) never read as WBC


def test_sofa_labs_include_platelets(loader):
    sofa = loader.load_sofa_labs({2001})
    assert set(sofa["lab_name"]) >= {"creatinine", "platelets"}


def test_sepsis3_labels_distinguish_the_two_stays(loader):
    labels = loader.derive_sepsis_labels().set_index("stay_id")
    assert labels.loc[3001, "sepsis_label"] == 1
    assert labels.loc[3001, "label_source"] == "sepsis3"
    assert labels.loc[3002, "sepsis_label"] == 0


def test_training_dataset_is_ordered_unique_and_labelled(loader):
    df = loader.build_training_dataset(max_patients=2)
    assert {"patient_id", "timestamp", "sepsis_label"} <= set(df.columns)
    assert not df.duplicated(["patient_id", "timestamp"]).any()
    for _, rows in df.groupby("patient_id"):
        assert rows["timestamp"].is_monotonic_increasing
    by_patient = df.groupby("patient_id")["sepsis_label"].max()
    assert sorted(by_patient.tolist()) == [0, 1]
    if "wbc" in df:
        assert df["wbc"].dropna().le(100).all()  # K/uL, never a platelet count
