#!/usr/bin/env python3
"""
Example: Run the Sepsis Vitals feature engineering pipeline on synthetic data.

Usage:
    PYTHONPATH=src python3 examples/run_feature_pipeline.py
"""

import numpy as np
import pandas as pd

from sepsis_vitals.features import build_feature_set, CORE_VITALS
from sepsis_vitals.data_quality import generate_quality_report
from sepsis_vitals.scores import compute_scores


def generate_synthetic_patients(n_patients: int = 5, obs_per: int = 8) -> pd.DataFrame:
    """Generate synthetic vitals data for demonstration."""
    rng = np.random.default_rng(42)
    rows = []
    base_ts = pd.Timestamp("2024-06-01 08:00")

    for pid in range(n_patients):
        is_septic = rng.random() < 0.3
        for i in range(obs_per):
            temp_base = 38.8 if is_septic else 37.0
            hr_base = 115 if is_septic else 78
            rr_base = 24 if is_septic else 16

            rows.append({
                "patient_id": f"PT-{pid:03d}",
                "timestamp": base_ts + pd.Timedelta(hours=pid * 24 + i * 4),
                "age_years": int(rng.integers(25, 75)),
                "temperature": round(float(rng.normal(temp_base, 0.5)), 1),
                "heart_rate": int(rng.normal(hr_base, 10)),
                "resp_rate": int(rng.normal(rr_base, 3)),
                "sbp": int(rng.normal(110 if not is_septic else 88, 12)),
                "spo2": int(rng.normal(97 if not is_septic else 92, 2)),
                "gcs": 15 if not is_septic else int(rng.choice([13, 14, 15])),
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  Sepsis Vitals — Feature Pipeline Demo")
    print("=" * 60)

    # Generate data
    df = generate_synthetic_patients()
    print(f"\nGenerated {len(df)} observations for {df['patient_id'].nunique()} patients\n")

    # Data quality report
    print("--- Data Quality Report ---")
    report = generate_quality_report(df, site_id="DEMO-SITE")
    print(f"  Overall status: {report['overall_status']}")
    print(f"  Patients: {report['temporal']['n_patients']}")
    print(f"  Observations: {report['temporal']['n_observations']}")
    print()

    # Feature engineering
    print("--- Feature Engineering ---")
    features = build_feature_set(df, score_cols=True)
    print(f"  Input columns: {len(df.columns)}")
    print(f"  Output columns: {len(features.columns)}")
    new_cols = set(features.columns) - set(df.columns)
    print(f"  New features: {len(new_cols)}")
    print()

    # Score a few patients
    print("--- Clinical Scores (latest observation per patient) ---")
    latest = df.sort_values("timestamp").groupby("patient_id").tail(1)
    for _, row in latest.iterrows():
        vitals = {v: row[v] for v in CORE_VITALS if pd.notna(row.get(v))}
        result = compute_scores(vitals)
        risk_icon = {"low": ".", "moderate": "*", "high": "!", "critical": "!!!"}
        print(
            f"  {row['patient_id']}: "
            f"qSOFA={result.qsofa} SIRS={result.sirs_count} "
            f"SI={result.shock_index or 'N/A'} NEWS2={result.news2_style} "
            f"→ {result.risk_level.upper()} {risk_icon.get(result.risk_level, '')}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
