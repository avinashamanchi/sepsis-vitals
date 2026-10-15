"""Provenance recorded with every generated report: code, environment, backends.

A report must say which code produced it (commit, and whether the working
tree had uncommitted changes), in which environment (Python, platform,
package versions, dependency lock), and which optional model backends were
actually available. A run on a host without LightGBM/XGBoost is not
equivalent to a candidate-complete run.
"""

from __future__ import annotations

import platform
import subprocess  # nosec B404 - fixed git commands only
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[3]
PACKAGES = ("scikit-learn", "numpy", "pandas", "scipy", "lightgbm", "xgboost")


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed arguments, no shell
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def code_version() -> Dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "uncommitted_changes": bool(status) if status is not None else None}


def environment() -> Dict[str, Any]:
    versions: Dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "packages": versions,
        "dependency_lock": "requirements/dev.txt",
    }


def run_provenance(backend_status: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code": code_version(),
        "environment": environment(),
    }
    if backend_status is not None:
        record["optional_backends"] = dict(backend_status)
    return record
