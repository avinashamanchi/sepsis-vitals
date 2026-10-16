#!/usr/bin/env python3
"""Smoke-test one installation set of sepsis-vitals in a clean environment.

    pip install ".[api]" && python scripts/smoke_extras.py api
    pip install "."      && python scripts/smoke_extras.py ""

Checks that what the set promises works without packages that only another
extra installs, and that those packages really are absent (otherwise the
check would prove nothing). Run from the repository root; the [ml] checks
use the committed synthetic model in models/.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A module that only each extra provides.
MARKERS = {
    "api": "fastapi",
    "ml": "sklearn",
    "train": "shap",
    "copilot": "anthropic",
    "integrations": "stripe",
}
VITALS = {"heart_rate": 118, "resp_rate": 24, "sbp": 96, "temperature": 38.6, "lactate": 2.4}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"ok: {message}")


def _get(port: int, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 - local server
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def smoke_api(with_ml: bool) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tmp = tempfile.mkdtemp(prefix="smoke-extras-")
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp}/smoke.db", "SEPSIS_ENV": "development",
           "SEPSIS_AUTH_ENABLED": "false", "SEPSIS_STATE_DIR": tmp,
           "SEPSIS_MODEL_DIR": str(ROOT / "models") if with_ml else tmp}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "sepsis_vitals.api:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            if server.poll() is not None:
                raise SystemExit("FAIL: server exited during startup:\n" + (server.stdout.read() if server.stdout else ""))
            try:
                if _get(port, "/health")[0] == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise SystemExit("FAIL: server did not become live within 60 s")
            time.sleep(0.5)
        check(_get(port, "/ready")[0] == 200, "/ready: API ready (development database)")
        status = _get(port, "/model/status")[1]
        check(status["clinically_ready"] is False and status["clinical_use"] == "not-permitted",
              "/model/status never reports clinical readiness")
        code, body = _get(port, "/predict", {"patient_id": "smoke", "vitals": VITALS, "age_years": 70})
        if with_ml:
            check(status["state"] == "ready", f"/model/status ready with [ml] ({status['state']})")
            check(code == 200 and body["validation_status"] == "synthetic-development", "/predict serves with [ml]")
        else:
            check(status["state"] == "unavailable", f"/model/status explains the missing ML runtime ({status['state']})")
            check(code == 503, "/predict answers 503 without [ml]")
        check(_get(port, "/score", VITALS)[0] == 200, "/score works")
    finally:
        server.terminate()
        server.wait(timeout=20)


def main() -> int:
    extras = {e for e in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if e}
    unknown = extras - set(MARKERS)
    if unknown:
        raise SystemExit(f"unknown extras: {sorted(unknown)}")
    for extra, module in MARKERS.items():
        installed = importlib.util.find_spec(module) is not None
        # shap depends on scikit-learn, so [train] brings it too (use [ml,train]).
        expected = extra in extras or (extra == "ml" and "train" in extras)
        check(installed == expected, f"{module} {'installed' if expected else 'absent'} for [{','.join(sorted(extras))}]")

    from sepsis_vitals.scores import compute_scores

    check(compute_scores(VITALS).risk_level in {"low", "moderate", "high", "critical"}, "rule-based scores work")
    if "api" in extras:
        smoke_api(with_ml="ml" in extras)
    if "ml" in extras:
        from sepsis_vitals.ml.predictor import SepsisPredictor

        predictor = SepsisPredictor(model_dir=str(ROOT / "models"), state_dir=tempfile.mkdtemp())
        predictor.load()
        result = predictor.predict(VITALS, patient_id="smoke", age_years=70)
        check(0.0 <= result.risk_probability <= 1.0, "model loads (checksum-verified) and predicts")
    if "train" in extras:
        import shap  # noqa: F401

        from sepsis_vitals.ml import trainer

        try:
            import lightgbm  # noqa: F401
            import xgboost  # noqa: F401
        except OSError as exc:
            # macOS wheels need the OpenMP runtime (brew install libomp); Linux
            # wheels bundle it. Documented in docs/dependencies.md.
            if sys.platform == "darwin" and "libomp" in str(exc):
                print("platform limitation: LightGBM/XGBoost need libomp on macOS; "
                      "the trainer skips them (Linux is the supported training platform)")
            else:
                raise
        check(bool(trainer._get_model_configs()), "training stack imports and has candidate models")
    print(f"smoke passed for [{','.join(sorted(extras))}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
