#!/usr/bin/env python3
"""Evaluate a site's recorded predictions against its adjudicated outcomes.

    DATABASE_URL=... python scripts/evaluate_local_study.py [--site WARD-A] [--output-dir ./local-study-report]

Runs inside the hospital, on the hospital's database, under the study's
approved protocol. It writes aggregates only (counts, AUROC/AUPRC with
patient bootstrap CIs, alerting before onset, lead time): no patient
identifiers, values or times. Keep the output outside the code repository;
the default directory is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", default="local-study-report")
    opts = parser.parse_args(argv)

    from sepsis_vitals.db import SessionLocal
    from sepsis_vitals.ml.provenance import run_provenance
    from sepsis_vitals.study.evaluation import evaluate_local_study

    db = SessionLocal()
    try:
        report = evaluate_local_study(db, site_id=opts.site, n_boot=opts.bootstrap)
    finally:
        db.close()
    report["site_id"] = opts.site
    report["provenance"] = run_provenance(command="python scripts/evaluate_local_study.py " + " ".join(sys.argv[1:]))
    out = Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "local_study_evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "patients", "sepsis_positive")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
