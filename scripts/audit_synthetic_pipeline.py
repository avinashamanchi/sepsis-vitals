#!/usr/bin/env python3
"""Audit what the synthetic development pipeline can and cannot demonstrate.

Writes reports/synthetic_pipeline_audit.{json,md}. Synthetic evidence only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sepsis_vitals.ml.pipeline_audit import run_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _markdown(r: dict) -> str:
    g, fg = r["generator"], r["feature_groups"]
    lab0, lab1 = g["row_mean_by_label"]["0"], g["row_mean_by_label"]["1"]
    lines = [
        "# Synthetic pipeline audit",
        "",
        "Synthetic development evidence only. It describes the hand-authored generator and the",
        "committed development model, not performance in any patient population.",
        "",
        f"Cohort: {r['cohort']['n_patients']:,} generated patients, seed {r['cohort']['seed']}, "
        f"{r['cohort']['test_rows']:,} held-out rows.",
        "",
        "## 1. Generator artefacts",
        "",
        f"- {g['fraction_negative_time_gaps']:.0%} of consecutive observations go backwards in time "
        "(was 32% before the 2026-10 fix to the timestamp accumulation).",
        f"- Rows labelled septic are not more abnormal on average: heart rate {lab1['heart_rate']} vs "
        f"{lab0['heart_rate']}, SBP {lab1['sbp']} vs {lab0['sbp']}, lactate {lab1['lactate']} vs "
        f"{lab0['lactate']} (septic vs non-septic rows).",
        f"- Septic patients are older (mean age {g['mean_age_ever_septic']} vs "
        f"{g['mean_age_never_septic']}); age alone gives patient-level AUROC "
        f"{g['patient_level_auroc_age_alone']:.3f}.",
        "",
        "## 2. Where discrimination comes from",
        "",
        "Same learner (HistGradientBoosting), same patient-level split, different feature groups:",
        "",
        "| Feature group | Features | Held-out row AUROC |",
        "| --- | ---: | ---: |",
    ]
    for name, v in fg.items():
        lines.append(f"| {name.replace('_', ' ')} | {v['n_features']} | {v['test_row_auroc']:.3f} |")
    if "committed_model" in r:
        m = r["committed_model"]
        lo, hi = m["row_auroc_95ci_patient_bootstrap"]
        lines += [
            "",
            "## 3. Committed model (`models/sepsis_model.joblib`)",
            "",
            "| Check | Value |",
            "| --- | ---: |",
            f"| Row AUROC, training-style features (95% CI, patient bootstrap) | "
            f"{m['row_auroc_training_style_features']:.3f} ({lo:.3f}-{hi:.3f}) |",
            f"| Row AUROC, each row scored without history | {m['row_auroc_inference_style_features']:.3f} |",
            f"| Mean predicted risk, with history vs without history | "
            f"{m['mean_predicted_risk_training_style']:.3f} vs {m['mean_predicted_risk_inference_style']:.3f} |",
            f"| Observed positive row rate | {m['observed_positive_row_rate']:.3f} |",
            f"| Pre-onset rows of future-septic patients vs never-septic rows (early warning) | "
            f"{m['pre_onset_vs_never_septic_auroc']:.3f} |",
            f"| Patient-level AUROC (max risk over stay) | {m['patient_level_auroc_max_risk']:.3f} |",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "- The headline synthetic AUROC measures recognising rows *after* labelled onset. Early-warning",
        "  ability (pre-onset rows) is much weaker and is the quantity a sepsis early-warning claim needs.",
        "- A large share of discrimination is available from demographics, which the generator ties to",
        "  the label by construction. The no-labs ablation inherits this.",
        "- Scoring without history under-predicts risk relative to training. The API now passes",
        "  recorded history for registered patients (tests/test_inference_parity.py); unregistered",
        "  IDs are still scored as first observations.",
        "- None of these numbers should be quoted as product performance.",
        "",
        "Reproduce: `python scripts/audit_synthetic_pipeline.py`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    opts = parser.parse_args()

    report = run_audit(n_patients=opts.patients, seed=opts.seed, model_dir=opts.model_dir)
    out = Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_pipeline_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "synthetic_pipeline_audit.md").write_text(_markdown(report))
    print(_markdown(report))


if __name__ == "__main__":
    main()
