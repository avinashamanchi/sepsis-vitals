"""Provenance recorded with every generated report.

A report must say:

- which code produced it (commit, and whether tracked files had uncommitted
  changes);
- in which environment (Python, platform, package versions, the dependency
  lock and its digest);
- which optional model backends were actually available, and which
  candidate models were compared. A run on a host without LightGBM/XGBoost
  is not equivalent to a candidate-complete run;
- the seeds, a fingerprint of the exact dataset, and the runtime.

Reports written before this module existed carry no such record; where one
was added later it is marked ``recorded_after_generation``.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess  # nosec B404 - fixed git commands only
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
    lock = ROOT / "requirements" / "dev.txt"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "packages": versions,
        "dependency_lock": "requirements/dev.txt",
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
    }


def dataset_fingerprint(frames: Iterable[Any]) -> Dict[str, Any]:
    """SHA-256 over the rows of each data frame, in order, plus their shapes.

    Uses pandas' row hashing (values and column names, not the index), so
    the same data gives the same digest on any host with the same pandas
    major version.
    """
    import pandas as pd

    digest = hashlib.sha256()
    shapes = []
    for frame in frames:
        digest.update(",".join(map(str, frame.columns)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        shapes.append(list(frame.shape))
    return {"sha256": digest.hexdigest(), "shapes": shapes, "method": "pandas row hash, ordered"}


def run_provenance(
    backend_status: Optional[Dict[str, str]] = None,
    *,
    seeds: Optional[Dict[str, Any]] = None,
    dataset: Optional[Dict[str, Any]] = None,
    candidates: Optional[Any] = None,
    runtime_seconds: Optional[float] = None,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code": code_version(),
        "environment": environment(),
    }
    if backend_status is not None:
        record["optional_backends"] = dict(backend_status)
    if seeds is not None:
        record["seeds"] = dict(seeds)
    if dataset is not None:
        record["dataset"] = dict(dataset)
    if candidates is not None:
        record["candidate_models"] = candidates
    if runtime_seconds is not None:
        record["runtime_seconds"] = round(runtime_seconds, 1)
    if command is not None:
        record["command"] = command
    return record


def markdown(record: Dict[str, Any]) -> list:
    """Report lines describing a provenance record."""
    code = record.get("code", {})
    env = record.get("environment", {})
    packages = ", ".join(f"{k} {v or 'not installed'}" for k, v in env.get("packages", {}).items())
    lines = [
        "## Provenance",
        "",
        f"- Generated: {record.get('generated_at')}"
        + (f" (runtime {record['runtime_seconds']:,} s)" if "runtime_seconds" in record else ""),
        f"- Code: commit `{code.get('commit')}`"
        + (" with uncommitted changes to tracked files" if code.get("uncommitted_changes") else "")
        + ("" if code.get("uncommitted_changes") is not None else " (working-tree state unknown)"),
        f"- Environment: Python {env.get('python')}, {env.get('platform')} ({env.get('machine')})",
        f"- Packages: {packages}",
        f"- Dependency lock: {env.get('dependency_lock')} (sha256 `{str(env.get('dependency_lock_sha256'))[:16]}…`)",
    ]
    if "optional_backends" in record:
        lines.append("- Optional backends: " + ", ".join(
            f"{name}: {status}" for name, status in record["optional_backends"].items()))
    if "candidate_models" in record:
        lines.append(f"- Candidate models compared: {record['candidate_models']}")
    if "seeds" in record:
        lines.append("- Seeds: " + ", ".join(f"{k} {v}" for k, v in record["seeds"].items()))
    if "dataset" in record:
        data = record["dataset"]
        lines.append(f"- Dataset: {data.get('description', '')} sha256 `{data.get('sha256')}`, "
                     f"shapes {data.get('shapes')}")
    if "command" in record:
        lines.append(f"- Command: `{record['command']}`")
    return lines + [""]
