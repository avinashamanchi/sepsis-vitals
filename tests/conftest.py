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


# ---------------------------------------------------------------------------
# Skip guard: a test may skip only for a known reason
# ---------------------------------------------------------------------------
# In CI the only expected skips are tests that need the MIMIC-IV / FHIR demo
# files, which are not in the repository. With SEPSIS_STRICT_SKIPS=1 (the CI
# test jobs) any other skip reason fails the run, so a missing dependency or
# a broken fixture cannot hide tests. The list of skips is always printed.

EXPECTED_SKIP_REASONS = (
    "MIMIC-IV Demo data not available",
    "MIMIC-IV Demo CSV not available",
    "MIMIC-IV FHIR Demo not available",
    "MIMIC-IV demo not downloaded",
    "FHIR demo data not available",
    "Both sources needed",
)

_skipped: list = []


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, "wasxfail"):
        _skipped.append((report.nodeid, _skip_reason(report)))


def _unexpected_skips() -> list:
    return [(node, reason) for node, reason in _skipped
            if not any(expected in reason for expected in EXPECTED_SKIP_REASONS)]


def pytest_terminal_summary(terminalreporter):
    unexpected = _unexpected_skips()
    if unexpected:
        terminalreporter.section("unexpected skips")
        for node, reason in unexpected:
            terminalreporter.write_line(f"{node}: {reason}")
        if os.environ.get("SEPSIS_STRICT_SKIPS") == "1":
            terminalreporter.write_line("SEPSIS_STRICT_SKIPS=1: failing the run.")


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("SEPSIS_STRICT_SKIPS") == "1" and _unexpected_skips() and exitstatus == 0:
        session.exitstatus = 1
