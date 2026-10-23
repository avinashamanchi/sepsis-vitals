"""
tests/test_skip_guard.py — with SEPSIS_STRICT_SKIPS=1 a skip for an
unlisted reason fails the run; the known MIMIC-demo skips do not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")


def _run(tmp_path: Path, reason: str) -> subprocess.CompletedProcess:
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n\n"
        "def test_runs():\n    assert True\n\n"
        f"def test_skips():\n    pytest.skip({reason!r})\n"
    )
    env = {**os.environ, "SEPSIS_STRICT_SKIPS": "1", "SEPSIS_TEST_DATABASE_URL": f"sqlite:///{tmp_path}/t.db"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path), str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )


def test_unexpected_skip_reason_fails_the_run(tmp_path):
    result = _run(tmp_path, "optional dependency missing")
    assert result.returncode == 1, result.stdout
    assert "unexpected skips" in result.stdout and "optional dependency missing" in result.stdout


@pytest.mark.parametrize("reason", ["MIMIC-IV Demo data not available", "FHIR demo data not available"])
def test_known_skip_reasons_pass(tmp_path, reason):
    result = _run(tmp_path, reason)
    assert result.returncode == 0, result.stdout
    assert "unexpected skips" not in result.stdout
