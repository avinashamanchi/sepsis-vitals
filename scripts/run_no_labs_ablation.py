#!/usr/bin/env python3
"""Run a matched full-feature versus no-labs synthetic ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrain import load_synthetic_data, run_pipeline  # noqa: E402
from sepsis_vitals.ml.ablation import build_comparison_report  # noqa: E402


def _markdown_report(report: dict) -> str:
    full = report["comparison"]["full"]
    no_labs = report["comparison"]["no_labs"]
    delta = report["auroc_delta_no_labs_minus_full"]
    provenance = report["data_provenance"]
    rows = [
        "# No-labs feature ablation",
        "",
        "This is synthetic development evidence, not clinical validation.",
        "",
        "| Feature set | Features | Best model | Held-out AUROC | Held-out AUPRC |",
        "| --- | ---: | --- | ---: | ---: |",
        (
            f"| Full | {full['n_features']} | {full['best_model']} | "
            f"{full['held_out_test_auroc']:.4f} | {full['held_out_test_auprc']:.4f} |"
        ),
        (
            f"| No labs | {no_labs['n_features']} | {no_labs['best_model']} | "
            f"{no_labs['held_out_test_auroc']:.4f} | "
            f"{no_labs['held_out_test_auprc']:.4f} |"
        ),
        "",
        f"AUROC change (no labs minus full): **{delta:+.4f}**.",
        "",
        "## Method",
        "",
        (
            f"{report['method']} Both arms used {provenance['n_patients']:,} generated "
            f"patients, seed {provenance['seed']}, and the same patient-level split."
        ),
        "",
        "## Honest interpretation",
        "",
        (
            "On this hand-authored synthetic held-out set, the no-labs model achieved "
            f"AUROC {no_labs['held_out_test_auroc']:.4f}. This result is useful for "
            "software-development ablation testing only; it cannot establish performance "
            "in a district hospital or any real patient population."
        ),
        "",
        "## Limitations",
        "",
    ]
    rows.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(rows) + "\n"


def main(args: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Run matched full versus no-labs synthetic retraining"
    )
    parser.add_argument("--patients", type=int, default=20_000)
    parser.add_argument("--prevalence", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--work-dir", default=".artifacts/no-labs-ablation")
    parser.add_argument("--report-dir", default="reports")
    opts = parser.parse_args(args)

    train_df, val_df, test_df, provenance = load_synthetic_data(
        n_patients=opts.patients,
        prevalence=opts.prevalence,
        seed=opts.seed,
    )

    work_dir = Path(opts.work_dir)
    full_result = run_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        provenance=provenance,
        output_dir=str(work_dir / "full"),
        cv_folds=opts.cv_folds,
        skip_shap=True,
        feature_set="full",
    )
    no_labs_result = run_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        provenance=provenance,
        output_dir=str(work_dir / "no_labs"),
        cv_folds=opts.cv_folds,
        skip_shap=True,
        feature_set="no_labs",
    )

    report = build_comparison_report(full_result, no_labs_result, provenance)
    report["split"] = {
        "train_patients": int(train_df["patient_id"].nunique()),
        "validation_patients": int(val_df["patient_id"].nunique()),
        "held_out_test_patients": int(test_df["patient_id"].nunique()),
        "train_observations": len(train_df),
        "validation_observations": len(val_df),
        "held_out_test_observations": len(test_df),
    }

    report_dir = Path(opts.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "no_labs_ablation.json"
    markdown_path = report_dir / "no_labs_ablation.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    markdown_path.write_text(_markdown_report(report))

    print(f"\nComparison JSON: {json_path}")
    print(f"Readable report: {markdown_path}")
    return report


if __name__ == "__main__":
    main()
