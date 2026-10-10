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
from sepsis_vitals.ml.ablation import add_uncertainty, build_comparison_report  # noqa: E402


def _markdown_report(report: dict) -> str:
    full = report["comparison"]["full"]
    no_labs = report["comparison"]["no_labs"]
    delta = report["auroc_delta_no_labs_minus_full"]
    provenance = report["data_provenance"]
    unc = report.get("uncertainty")
    rows = [
        "# No-labs feature ablation",
        "",
        "This is synthetic development evidence, not clinical validation.",
        "",
        "| Feature set | Features | Best model | Held-out AUROC (95% CI) | Held-out AUPRC (95% CI) |",
        "| --- | ---: | --- | ---: | ---: |",
        (
            f"| Full | {full['n_features']} | {full['best_model']} | "
            f"{full['held_out_test_auroc']:.4f}{_ci(unc, 'full', 'auroc')} | "
            f"{full['held_out_test_auprc']:.4f}{_ci(unc, 'full', 'auprc')} |"
        ),
        (
            f"| No labs | {no_labs['n_features']} | {no_labs['best_model']} | "
            f"{no_labs['held_out_test_auroc']:.4f}{_ci(unc, 'no_labs', 'auroc')} | "
            f"{no_labs['held_out_test_auprc']:.4f}{_ci(unc, 'no_labs', 'auprc')} |"
        ),
        "",
        f"AUROC change (no labs minus full): **{delta:+.4f}**{_diff_ci(unc, 'auroc')}; "
        f"AUPRC change{_diff(unc, 'auprc')}.",
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
    candidates = report.get("candidate_models", {}).get("full")
    if candidates:
        missing = [m for m in KNOWN_CANDIDATES if m not in candidates]
        env = report.get("environment", {})
        rows += [
            f"- Candidate models compared: {', '.join(candidates)}."
            + (f" Not available on this host: {', '.join(missing)} (they need the OpenMP runtime);"
               " a run where they load may select a different model." if missing else ""),
            f"- Environment: Python {env.get('python', '?')}, {env.get('platform', '?')}, "
            f"scikit-learn {env.get('scikit_learn', '?')}, pins from {env.get('dependency_lock', '?')}.",
        ]
    if unc:
        rows += [
            "",
            "## Uncertainty",
            "",
            f"- Estimand: row-level held-out AUROC/AUPRC of each arm, and the paired difference "
            f"(no labs minus full), on {unc['n_rows']:,} held-out rows from {unc['n_patients']:,} "
            f"patients ({unc['n_positive_rows']:,} positive rows, "
            f"{unc['n_patients_with_positive_rows']:,} patients with a positive row). No rows were excluded.",
            f"- Method: {unc['method']}, {int(unc['level'] * 100)}% level, {unc['n_boot']:,} replicates "
            f"(seed {unc['seed']}); {unc['degenerate_replicates']} single-class replicates excluded"
            + ("" if unc["reliable"] else " (more than 10%: intervals unreliable)") + ".",
            "- The intervals cover sampling variability of this synthetic held-out set given the trained "
            "models. They do not include training variability and are not evidence about patients.",
        ]
    return "\n".join(rows) + "\n"


def _ci(unc: dict | None, arm: str, metric: str) -> str:
    if not unc or not unc["arms"][arm][metric]["ci"]:
        return ""
    lo, hi = unc["arms"][arm][metric]["ci"]
    return f" ({lo:.3f}-{hi:.3f})"


def _diff_ci(unc: dict | None, metric: str) -> str:
    if not unc:
        return ""
    ci = unc["differences_vs_reference"]["no_labs"][metric]["ci"]
    return f" (95% CI {ci[0]:+.3f} to {ci[1]:+.3f})" if ci else ""


def _diff(unc: dict | None, metric: str) -> str:
    if not unc:
        return " not estimated"
    d = unc["differences_vs_reference"]["no_labs"][metric]
    return f" {d['estimate']:+.4f}{_diff_ci(unc, metric)}"


KNOWN_CANDIDATES = ("LightGBM", "XGBoost", "RandomForest", "GradientBoosting", "LogisticRegression")


def _environment() -> dict:
    import platform

    import sklearn

    return {"python": platform.python_version(), "platform": platform.platform(terse=True),
            "scikit_learn": sklearn.__version__, "dependency_lock": "requirements/dev.txt"}


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
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates for CIs")
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
    add_uncertainty(report, full_result, no_labs_result, n_boot=opts.bootstrap, seed=opts.seed)
    report["candidate_models"] = {
        arm: [m["name"] for m in result["report"].get("model_comparison", [])]
        for arm, result in (("full", full_result), ("no_labs", no_labs_result))
    }
    report["environment"] = _environment()
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
