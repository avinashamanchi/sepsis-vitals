#!/usr/bin/env python3
"""
generate_validation_report.py — Build a pitch-deck-ready HTML validation report.

Reads model artifacts from models/ and generates a standalone HTML report
with ROC curves, clinical performance metrics, model comparison tables,
SHAP feature importance, and data provenance documentation.

Usage:
    python generate_validation_report.py
    python generate_validation_report.py --output docs/validation-report.html
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def load_artifacts(model_dir: str = "models") -> dict:
    """Load all model artifacts."""
    model_path = Path(model_dir)

    with open(model_path / "model_metadata.json") as f:
        metadata = json.load(f)

    with open(model_path / "evaluation_report.json") as f:
        report = json.load(f)

    return {"metadata": metadata, "report": report}


def generate_html_report(artifacts: dict, output_path: str) -> str:
    """Generate a standalone HTML validation report."""
    meta = artifacts["metadata"]
    report = artifacts["report"]

    card = meta.get("model_card", {})
    test = report.get("test_set_performance", {})
    cal = report.get("calibration", {})
    provenance = meta.get("data_provenance", report.get("data_provenance", {}))
    shap_imp = report.get("shap_feature_importance", {})
    comparison = report.get("model_comparison", [])
    roc = report.get("roc_curve", {})
    pr = report.get("pr_curve", {})
    clinical = report.get("clinical_performance", {})
    version = meta.get("version", card.get("version", "1.0.0"))
    source = provenance.get("source", "synthetic")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build ROC/PR curve SVG inline data
    roc_points = _build_svg_path(roc.get("fpr", []), roc.get("tpr", []))
    pr_points = _build_svg_path(pr.get("recall", []), pr.get("precision", []))

    # Top SHAP features
    shap_items = list(shap_imp.items())[:15]
    max_shap = shap_items[0][1] if shap_items else 1

    # Model comparison rows
    comparison_rows = ""
    best_name = report.get("best_model", {}).get("name", meta.get("model_name", ""))
    for m in sorted(comparison, key=lambda x: x.get("val_auroc", 0), reverse=True):
        is_best = m["name"] == best_name
        marker = ' class="best-row"' if is_best else ""
        badge = " *" if is_best else ""
        comparison_rows += f"""
        <tr{marker}>
          <td>{m['name']}{badge}</td>
          <td>{m.get('val_auroc', 0):.4f}</td>
          <td>{m.get('val_auprc', 0):.4f}</td>
          <td>{m.get('val_sensitivity', 0):.1%}</td>
          <td>{m.get('val_specificity', 0):.1%}</td>
          <td>{m.get('val_f1', 0):.4f}</td>
          <td>{m.get('training_time_s', 0):.1f}s</td>
        </tr>"""

    # SHAP rows
    shap_rows = ""
    for i, (feat, imp) in enumerate(shap_items):
        bar_width = (imp / max_shap) * 100 if max_shap > 0 else 0
        shap_rows += f"""
        <tr>
          <td class="rank">{i+1}</td>
          <td class="feat-name">{feat}</td>
          <td class="shap-val">{imp:.4f}</td>
          <td class="bar-cell"><div class="shap-bar" style="width: {bar_width:.1f}%"></div></td>
        </tr>"""

    # Data provenance section
    if source == "MIMIC-IV":
        prov_badge = '<span class="badge badge-green">MIMIC-IV Clinical Data</span>'
        prov_detail = f"""
        <p><strong>Source:</strong> MIMIC-IV v2.2+ (PhysioNet)</p>
        <p><strong>Institution:</strong> Beth Israel Deaconess Medical Center</p>
        <p><strong>Cohort:</strong> {provenance.get('total_stays', 'N/A'):,} ICU stays, {provenance.get('total_patients', 'N/A'):,} patients</p>
        <p><strong>Sepsis Prevalence:</strong> {provenance.get('sepsis_prevalence', 0):.1%}</p>
        <p><strong>Label Derivation:</strong> Sepsis-3 (ICD-10 codes A40, A41, R65.2)</p>
        <p><strong>Split:</strong> Patient-level (no data leakage) &mdash;
           Train: {provenance.get('train_patients', 'N/A'):,} /
           Val: {provenance.get('val_patients', 'N/A'):,} /
           Test: {provenance.get('test_patients', 'N/A'):,}</p>
        """
    else:
        prov_badge = '<span class="badge badge-amber">Synthetic (Pre-Validation)</span>'
        prov_detail = f"""
        <p><strong>Source:</strong> NHANES-calibrated synthetic generator</p>
        <p><strong>Patients:</strong> {provenance.get('n_patients', 'N/A'):,}</p>
        <p><strong>Sepsis Prevalence:</strong> {provenance.get('sepsis_prevalence', 0):.0%}</p>
        <p><strong>Seed:</strong> {provenance.get('seed', 42)}</p>
        <p class="warning-text">These metrics are pre-validation estimates. Clinical validation on
        MIMIC-IV or institutional EHR data is <strong>required</strong> before any deployment claims.</p>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sepsis Vitals &mdash; Model Validation Report v{version}</title>
<style>
  :root {{
    --void: #04080f;
    --surface: #0a1628;
    --card: #111d32;
    --border: #1a2d4a;
    --text: #e0e8f0;
    --muted: #8899aa;
    --accent: #00ff9d;
    --accent2: #00cc7d;
    --red: #ff4d6a;
    --amber: #ffb84d;
    --blue: #4da6ff;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--void);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    color: var(--accent);
    margin-bottom: 0.25rem;
  }}
  h2 {{
    font-size: 1.4rem;
    color: var(--accent);
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  h3 {{ font-size: 1.1rem; color: var(--text); margin: 1.5rem 0 0.75rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }}
  .metric-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    text-align: center;
  }}
  .metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
  }}
  .metric-label {{
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.6rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
  }}
  td {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }}
  .best-row {{ background: rgba(0, 255, 157, 0.08); }}
  .best-row td {{ color: var(--accent); font-weight: 600; }}
  .badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
  }}
  .badge-green {{ background: rgba(0,255,157,0.15); color: var(--accent); border: 1px solid var(--accent); }}
  .badge-amber {{ background: rgba(255,184,77,0.15); color: var(--amber); border: 1px solid var(--amber); }}
  .badge-red {{ background: rgba(255,77,106,0.12); color: var(--red); border: 1px solid var(--red); }}
  .shap-bar {{
    height: 18px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 3px;
    min-width: 2px;
  }}
  .bar-cell {{ width: 50%; }}
  .rank {{ color: var(--muted); width: 30px; }}
  .feat-name {{ color: var(--text); font-family: 'JetBrains Mono', monospace; }}
  .shap-val {{ color: var(--accent); text-align: right; width: 80px; }}
  .svg-chart {{
    width: 100%;
    max-width: 500px;
    aspect-ratio: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  .charts-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  .chart-container {{ flex: 1; min-width: 300px; }}
  .chart-title {{ color: var(--muted); font-size: 0.85rem; text-align: center; margin-bottom: 0.5rem; }}
  .warning-text {{ color: var(--amber); font-style: italic; margin-top: 0.75rem; }}
  .footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.8rem;
    text-align: center;
  }}
  .confusion-grid {{
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    gap: 2px;
    max-width: 350px;
    margin: 1rem auto;
    font-family: 'JetBrains Mono', monospace;
  }}
  .cm-cell {{
    padding: 1rem;
    text-align: center;
    border-radius: 4px;
    font-weight: 600;
  }}
  .cm-header {{ color: var(--muted); font-size: 0.75rem; padding: 0.5rem; text-transform: uppercase; }}
  .cm-tp {{ background: rgba(0,255,157,0.15); color: var(--accent); }}
  .cm-tn {{ background: rgba(0,255,157,0.08); color: var(--accent2); }}
  .cm-fp {{ background: rgba(255,184,77,0.1); color: var(--amber); }}
  .cm-fn {{ background: rgba(255,77,106,0.1); color: var(--red); }}
  p {{ margin-bottom: 0.5rem; }}
  @media (max-width: 700px) {{
    body {{ padding: 1rem; }}
    .metrics-grid {{ grid-template-columns: 1fr 1fr; }}
    .charts-row {{ flex-direction: column; }}
  }}
</style>
</head>
<body>
<div class="container">

<h1>Sepsis Vitals</h1>
<p class="subtitle">
  Model Validation Report &mdash; v{version} &mdash; {generated}
  <br>{prov_badge}
  <span class="badge badge-red">Research Use Only</span>
</p>

<!-- ── Hero metrics ──────────────────────────────────────────────── -->
<div class="card">
  <div class="metrics-grid">
    <div class="metric-box">
      <div class="metric-value">{test.get('test_auroc', 0):.4f}</div>
      <div class="metric-label">Test AUROC</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{test.get('test_recall', 0):.1%}</div>
      <div class="metric-label">Sensitivity</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{test.get('test_specificity', 0):.1%}</div>
      <div class="metric-label">Specificity</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{test.get('test_ppv', 0):.1%}</div>
      <div class="metric-label">PPV</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{test.get('test_npv', 0):.1%}</div>
      <div class="metric-label">NPV</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{cal.get('ece', 0):.4f}</div>
      <div class="metric-label">Calibration ECE</div>
    </div>
  </div>
</div>

<!-- ── ROC + PR Curves ───────────────────────────────────────────── -->
<h2>Receiver Operating Characteristic &amp; Precision-Recall</h2>
<div class="card">
  <div class="charts-row">
    <div class="chart-container">
      <div class="chart-title">ROC Curve (AUROC = {test.get('test_auroc', 0):.4f})</div>
      <svg class="svg-chart" viewBox="0 0 400 400">
        <rect x="50" y="10" width="340" height="340" fill="var(--surface)" stroke="var(--border)" rx="4"/>
        <!-- Grid -->
        <line x1="50" y1="180" x2="390" y2="180" stroke="var(--border)" stroke-dasharray="4"/>
        <line x1="220" y1="10" x2="220" y2="350" stroke="var(--border)" stroke-dasharray="4"/>
        <!-- Diagonal -->
        <line x1="50" y1="350" x2="390" y2="10" stroke="var(--muted)" stroke-dasharray="6" opacity="0.4"/>
        <!-- ROC curve -->
        <polyline points="{roc_points}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>
        <!-- Axes labels -->
        <text x="220" y="385" fill="var(--muted)" font-size="11" text-anchor="middle">False Positive Rate</text>
        <text x="15" y="180" fill="var(--muted)" font-size="11" text-anchor="middle" transform="rotate(-90 15 180)">True Positive Rate</text>
        <!-- Tick labels -->
        <text x="50" y="370" fill="var(--muted)" font-size="9" text-anchor="middle">0</text>
        <text x="220" y="370" fill="var(--muted)" font-size="9" text-anchor="middle">0.5</text>
        <text x="390" y="370" fill="var(--muted)" font-size="9" text-anchor="middle">1.0</text>
        <text x="45" y="353" fill="var(--muted)" font-size="9" text-anchor="end">0</text>
        <text x="45" y="183" fill="var(--muted)" font-size="9" text-anchor="end">0.5</text>
        <text x="45" y="17" fill="var(--muted)" font-size="9" text-anchor="end">1.0</text>
      </svg>
    </div>
    <div class="chart-container">
      <div class="chart-title">Precision-Recall (AUPRC = {test.get('test_auprc', 0):.4f})</div>
      <svg class="svg-chart" viewBox="0 0 400 400">
        <rect x="50" y="10" width="340" height="340" fill="var(--surface)" stroke="var(--border)" rx="4"/>
        <line x1="50" y1="180" x2="390" y2="180" stroke="var(--border)" stroke-dasharray="4"/>
        <line x1="220" y1="10" x2="220" y2="350" stroke="var(--border)" stroke-dasharray="4"/>
        <!-- PR curve -->
        <polyline points="{pr_points}" fill="none" stroke="var(--blue)" stroke-width="2.5"/>
        <text x="220" y="385" fill="var(--muted)" font-size="11" text-anchor="middle">Recall</text>
        <text x="15" y="180" fill="var(--muted)" font-size="11" text-anchor="middle" transform="rotate(-90 15 180)">Precision</text>
        <text x="50" y="370" fill="var(--muted)" font-size="9" text-anchor="middle">0</text>
        <text x="220" y="370" fill="var(--muted)" font-size="9" text-anchor="middle">0.5</text>
        <text x="390" y="370" fill="var(--muted)" font-size="9" text-anchor="middle">1.0</text>
        <text x="45" y="353" fill="var(--muted)" font-size="9" text-anchor="end">0</text>
        <text x="45" y="183" fill="var(--muted)" font-size="9" text-anchor="end">0.5</text>
        <text x="45" y="17" fill="var(--muted)" font-size="9" text-anchor="end">1.0</text>
      </svg>
    </div>
  </div>
</div>

<!-- ── Clinical Performance ──────────────────────────────────────── -->
<h2>Clinical Operating Characteristics</h2>
<div class="card">
  <div class="metrics-grid">
    <div class="metric-box">
      <div class="metric-value">{clinical.get('sensitivity_at_90_specificity', 0):.1%}</div>
      <div class="metric-label">Sensitivity @ 90% Specificity</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{clinical.get('sensitivity_at_95_specificity', 0):.1%}</div>
      <div class="metric-label">Sensitivity @ 95% Specificity</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{test.get('test_f1', 0):.4f}</div>
      <div class="metric-label">F1 Score</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{test.get('test_brier', 0):.4f}</div>
      <div class="metric-label">Brier Score</div>
    </div>
  </div>

  <h3>Confusion Matrix (Test Set)</h3>
  <div class="confusion-grid">
    <div class="cm-header"></div>
    <div class="cm-header">Predicted +</div>
    <div class="cm-header">Predicted &minus;</div>
    <div class="cm-header">Actual +</div>
    <div class="cm-cell cm-tp">TP: {test.get('test_true_positives', 0):,}</div>
    <div class="cm-cell cm-fn">FN: {test.get('test_false_negatives', 0):,}</div>
    <div class="cm-header">Actual &minus;</div>
    <div class="cm-cell cm-fp">FP: {test.get('test_false_positives', 0):,}</div>
    <div class="cm-cell cm-tn">TN: {test.get('test_true_negatives', 0):,}</div>
  </div>
</div>

<!-- ── Model Comparison ──────────────────────────────────────────── -->
<h2>Model Comparison</h2>
<div class="card">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>AUROC</th>
        <th>AUPRC</th>
        <th>Sensitivity</th>
        <th>Specificity</th>
        <th>F1</th>
        <th>Train Time</th>
      </tr>
    </thead>
    <tbody>{comparison_rows}
    </tbody>
  </table>
  <p style="margin-top: 0.75rem; color: var(--muted); font-size: 0.8rem;">
    * = selected model (best validation AUROC with sensitivity tiebreaker). All models Platt-calibrated.
  </p>
</div>

<!-- ── SHAP Feature Importance ───────────────────────────────────── -->
<h2>Feature Importance (SHAP)</h2>
<div class="card">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Feature</th>
        <th>|SHAP|</th>
        <th>Relative Importance</th>
      </tr>
    </thead>
    <tbody>{shap_rows}
    </tbody>
  </table>
</div>

<!-- ── Data Provenance ───────────────────────────────────────────── -->
<h2>Data Provenance</h2>
<div class="card">
  {prov_detail}
</div>

<!-- ── Regulatory & Limitations ──────────────────────────────────── -->
<h2>Regulatory Status &amp; Limitations</h2>
<div class="card">
  <p><strong>Regulatory Status:</strong> {card.get('regulatory_status', 'Research use only.')}</p>
  <p><strong>Intended Use:</strong> {card.get('intended_use', 'Sepsis risk screening. Requires clinician review.')}</p>
  <p><strong>Limitations:</strong> {card.get('limitations', 'See model card.')}</p>
</div>

<div class="footer">
  Sepsis Vitals v{version} &mdash; Model Validation Report &mdash; Generated {generated}
  <br>Research use only. Not FDA-cleared.
</div>

</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"  Validation report written to {output_path}")
    return output_path


def _build_svg_path(x_vals: list, y_vals: list) -> str:
    """Convert data points to SVG polyline points string.

    Maps x:[0,1] to pixel [50,390] and y:[0,1] to pixel [350,10].
    """
    if not x_vals or not y_vals:
        return "50,350 390,10"

    points = []
    for x, y in zip(x_vals, y_vals):
        px = 50 + x * 340
        py = 350 - y * 340
        points.append(f"{px:.0f},{py:.0f}")
    return " ".join(points)


def main():
    parser = argparse.ArgumentParser(description="Generate HTML validation report")
    parser.add_argument(
        "--model-dir", default="models",
        help="Model artifacts directory (default: models)"
    )
    parser.add_argument(
        "--output", default="docs/validation-report.html",
        help="Output HTML path (default: docs/validation-report.html)"
    )
    opts = parser.parse_args()

    artifacts = load_artifacts(opts.model_dir)
    generate_html_report(artifacts, opts.output)


if __name__ == "__main__":
    main()
