"""Test isolation: never read or write a developer's local database.

``sepsis_vitals.db`` binds its engine at import time from DATABASE_URL, which
defaults to ``./sepsis_vitals.db``. Point it at a throwaway SQLite file before
any test imports the package. Set SEPSIS_TEST_DATABASE_URL to run the suite
against a dedicated test database instead.
"""

import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="sepsis-vitals-tests-")
os.environ["DATABASE_URL"] = os.environ.get(
    "SEPSIS_TEST_DATABASE_URL", f"sqlite:///{_tmp_dir}/test.db"
)


# ---------------------------------------------------------------------------
# Shared MIMIC-IV demo dataset (built once per session; ~60 s per build)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

MIMIC_DEMO_ROOT = "physionet.org/files/mimic-iv-demo/2.2"


@pytest.fixture(scope="session")
def mimic_demo_dataset():
    """Training dataset for the first 50 MIMIC-IV demo patients (skips if absent)."""
    from pathlib import Path

    if not (Path(MIMIC_DEMO_ROOT) / "hosp" / "patients.csv.gz").exists():
        pytest.skip("MIMIC-IV Demo data not available")
    from sepsis_vitals.ml.mimic_loader import MIMICLoader

    return MIMICLoader.from_demo().build_training_dataset(max_patients=50)


def first_patients(df, n):
    """Rows belonging to the first *n* patients of a dataset."""
    keep = list(dict.fromkeys(df["patient_id"]))[:n]
    return df[df["patient_id"].isin(keep)]
