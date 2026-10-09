#!/usr/bin/env python3
"""Evaluate synthetic-generator profiles with one protocol (engineering evidence).

    python scripts/evaluate_synthetic_profiles.py --horizon-hours 12

--horizon-hours is required and is used only to demonstrate the
onset-within-horizon label mechanics; it is not a clinical choice. Writes
reports/synthetic_profile_evaluation.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from sepsis_vitals.ml.profile_evaluation import PROFILES, TARGET_SPECIFICITY, evaluate_profile

ROOT = Path(__file__).resolve().parents[1]


def _md(results: list, horizon: float, n: int, seed: int) -> str:
    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        "# Synthetic generator profiles: evaluation",
        "",
        "**Engineering evidence only.** These numbers describe a hand-authored generator, not",
        "patients, and do not support any performance claim. The prediction horizon below",
        f"({horizon:g} h) only demonstrates the label mechanics; choosing a horizon is a",
        "clinical decision for the study team.",
        "",
        f"{n:,} generated patients per profile, seed {seed}. Split: patient-level and temporal",
        "(earliest 60% of admissions train, next 15% validate, latest 25% test). Operating point:",
        f"specificity {TARGET_SPECIFICITY:.2f} on validation (an engineering convention, not a clinical threshold).",
        "",
        "| Profile | Label | AUROC (95% CI) | AUPRC | Brier | Cal. slope | Cal. intercept | Sens @ op. | PPV @ op. | AUROC no labs | AUROC demographics only |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lo, hi = r["auroc_95ci"]
        label = r["label_mode"] if r["label_mode"] == "current_state" else f"onset within {r['horizon_hours']:g} h"
        op = r["operating_point"]
        lines.append(
            f"| {r['profile']} | {label} | {fmt(r['auroc'])} ({lo:.3f}-{hi:.3f}) | {fmt(r['auprc'])} | "
            f"{r['brier']:.3f} | {r['calibration']['slope']:.2f} | {r['calibration']['intercept']:.2f} | "
            f"{fmt(op['sensitivity'])} | {fmt(op['ppv'])} | {fmt(r['auroc_without_labs'])} | "
            f"{fmt(r['auroc_demographics_only'])} |"
        )
    lines += ["", "## Subgroup AUROC (test split)", "", "| Profile | " + " | ".join(results[0]["subgroups"]) + " |",
              "| --- |" + " ---: |" * len(results[0]["subgroups"])]
    for r in results:
        lines.append(f"| {r['profile']} ({'horizon' if r['horizon_hours'] else 'current'}) | "
                     + " | ".join(fmt(v["auroc"]) for v in r["subgroups"].values()) + " |")
    lines += [
        "",
        "## Reading the table",
        "",
        "- *Demographics only* measures how much of the label the generator ties to age and",
        "  comorbidity. In the decoupled profile it should fall towards 0.5.",
        "- *Onset within horizon* scores only pre-onset rows: the early-warning question.",
        "  It is expected to be much harder than recognising rows after onset.",
        "- None of these models is a candidate for clinical use. The committed artifact",
        "  (models/manifest.json) is still the legacy synthetic model.",
        "",
        "Reproduce: `python scripts/evaluate_synthetic_profiles.py --horizon-hours "
        f"{horizon:g} --patients {n} --seed {seed}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizon-hours", type=float, required=True)
    parser.add_argument("--patients", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    opts = parser.parse_args()
    warnings.filterwarnings("ignore")

    results = [
        evaluate_profile("legacy", opts.patients, opts.seed, **PROFILES["legacy"]),
        evaluate_profile("decoupled", opts.patients, opts.seed, **PROFILES["decoupled"]),
        evaluate_profile(
            "decoupled", opts.patients, opts.seed, label_mode="onset_within_horizon",
            horizon_hours=opts.horizon_hours, **PROFILES["decoupled"],
        ),
    ]
    out = Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_profile_evaluation.json").write_text(
        json.dumps({"claim_scope": "synthetic engineering evidence only", "results": results}, indent=2) + "\n"
    )
    md = _md(results, opts.horizon_hours, opts.patients, opts.seed)
    (out / "synthetic_profile_evaluation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
