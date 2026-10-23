#!/usr/bin/env python3
"""Drive a running API through the demo cases with synthetic patients only.

    python scripts/demo_synthetic.py --base http://127.0.0.1:8000 [--case all]

Cases:

- `status`: health, readiness and model status.
- `high`: a deteriorating synthetic timeline.
- `low`: a stable synthetic timeline.
- `malformed`: payloads the contract must refuse.
- `failure`: run against a server started without a model; expects "no risk calculated".

Every response is checked, and the script exits non-zero if one is not as
expected. The base URL must be a local address; this script never talks to
anything else. Uses no credentials: start the API in development mode with
SEPSIS_AUTH_ENABLED=false (docs/demo.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def call(base: str, method: str, path: str, body: Optional[dict] = None) -> Tuple[int, Any]:
    if method == "POST":
        time.sleep(0.6)  # stay under the API's ML rate limit (2 requests/second)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 - local base URL enforced
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def timeline(kind: str, hours: int = 4) -> List[Dict[str, Any]]:
    """Hourly synthetic observations ending now. Values are illustrative."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    rows = []
    for i in range(hours):
        t = (now - timedelta(hours=hours - 1 - i)).isoformat()
        if kind == "high":
            rows.append({"timestamp": t, "heart_rate": 96 + 9 * i, "resp_rate": 19 + 2 * i,
                         "sbp": 118 - 9 * i, "temperature": 37.6 + 0.4 * i, "spo2": 96 - i,
                         "procalcitonin": 0.8 + 1.2 * i, "lactate": 1.8 + 0.5 * i})
        else:
            rows.append({"timestamp": t, "heart_rate": 74 + (i % 2), "resp_rate": 14,
                         "sbp": 124 - (i % 2), "temperature": 36.8, "spo2": 98})
    return rows


class Demo:
    def __init__(self, base: str):
        self.base = base
        self.failures = 0

    def check(self, ok: bool, label: str, detail: Any = "") -> bool:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail != "" else ""))
        self.failures += 0 if ok else 1
        return ok

    def status(self) -> None:
        code, _ = call(self.base, "GET", "/health")
        self.check(code == 200, "liveness /health")
        code, ready = call(self.base, "GET", "/ready")
        self.check(code == 200 and ready.get("audit_guard") == "present", "readiness /ready", ready)
        _, model = call(self.base, "GET", "/model/status")
        self.check(model.get("clinical_use") == "not-permitted", "model status", {
            k: model.get(k) for k in ("state", "model_version", "validation_status", "checks", "clinically_ready")})

    def score(self, kind: str) -> Optional[dict]:
        code, body = call(self.base, "POST", "/predict/timeseries", {
            "patient_id": f"synthetic-demo-{kind}", "age_years": 78 if kind == "high" else 34,
            "units": {"temperature": "Cel", "heart_rate": "/min", "sbp": "mm[Hg]"},
            "observations": timeline(kind),
        })
        if not self.check(code == 200, f"{kind}-risk synthetic timeline scored", code):
            print(json.dumps(body, indent=2))
            return None
        print(json.dumps({
            "risk_level": body["risk_level"], "rule_level": body["rule_risk_level"],
            "model_level": body["model_risk_level"], "model_output": body["risk_probability"],
            "alert": body["alert"], "threshold_policy": body["threshold_policy"]["id"],
            "top_factors": [(f["label"], f["contribution"]) for f in body["top_risk_factors"][:4]],
            "data_quality": [w["code"] for w in body["data_quality"]["warnings"]],
            "uncertainty": body["uncertainty"]["method"],
            "request_id": body["request_id"], "audit_record_id": body["audit_record_id"],
            "notice": body["decision_support_notice"],
        }, indent=2))
        self.check(body["clinical_use"] == "not-permitted" and "not a diagnosis" in body["decision_support_notice"],
                   "decision-support language present")
        return body

    def malformed(self) -> None:
        good = {"patient_id": "synthetic-demo-bad", "observations": timeline("low", 3)}
        cases = {
            "unordered timestamps": {**good, "observations": list(reversed(good["observations"]))},
            "Fahrenheit declared": {**good, "units": {"temperature": "[degF]"}},
            "unknown field": {**good, "observations": [*good["observations"][:-1],
                                                       {**good["observations"][-1], "glucose": 5.4}]},
            "timestamp without offset": {**good, "observations": [
                {**good["observations"][0], "timestamp": "2026-01-01T00:00:00"}, *good["observations"][1:]]},
            "out-of-range value": {**good, "observations": [*good["observations"][:-1],
                                                            {**good["observations"][-1], "temperature": 98.6}]},
        }
        for label, payload in cases.items():
            code, body = call(self.base, "POST", "/predict/timeseries", payload)
            reason = body.get("detail", [{}])[0].get("msg", body) if isinstance(body.get("detail"), list) else body
            self.check(code == 422, f"refused: {label}", reason)

    def failure(self) -> None:
        code, body = call(self.base, "POST", "/predict/timeseries",
                          {"patient_id": "synthetic-demo-fail", "observations": timeline("high")})
        detail = body.get("detail")
        text = detail.get("message") if isinstance(detail, dict) else str(detail)
        self.check(code == 503 and "risk_probability" not in body, "backend failure: no risk calculated", text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--case", choices=["all", "status", "high", "low", "malformed", "failure"], default="all")
    opts = parser.parse_args(argv)
    if urlparse(opts.base).hostname not in LOCAL_HOSTS:
        parser.error("--base must be a local address (synthetic demo only)")
    demo = Demo(opts.base.rstrip("/"))
    if opts.case in ("all", "status"):
        demo.status()
    if opts.case in ("all", "high"):
        demo.score("high")
    if opts.case in ("all", "low"):
        demo.score("low")
    if opts.case in ("all", "malformed"):
        demo.malformed()
    if opts.case == "failure":
        demo.failure()
    print(f"\n{demo.failures} check(s) failed" if demo.failures else "\nAll checks passed")
    return 1 if demo.failures else 0


if __name__ == "__main__":
    sys.exit(main())
