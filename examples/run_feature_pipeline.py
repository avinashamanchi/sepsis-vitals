<<<<<<< HEAD
import numpy as np
import pandas as pd

from sepsis_vitals import build_feature_set
from sepsis_vitals.data_quality import summarize_vitals_quality
from sepsis_vitals.features import get_feature_inventory


def main() -> None:
    raw = pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P001", "P002", "P002"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 10:00",
                    "2026-01-01 12:00",
                    "2026-01-01 09:00",
                    "2026-01-01 11:00",
                ]
            ),
            "temperature": [37.1, 38.4, 39.1, 36.8, np.nan],
            "heart_rate": [88, 95, 124, 76, 102],
            "resp_rate": [18, 23, 28, 16, 22],
            "sbp": [118, 99, 88, 130, 105],
            "spo2": [97, 94, 91, 98, 95],
            "gcs": [15, 14, 13, 15, 15],
            "age_years": [45, 45, 45, 72, 72],
        }
    )

    quality = summarize_vitals_quality(raw)
    features = build_feature_set(raw)
    inventory = get_feature_inventory(features)

    print(f"Input shape: {raw.shape}")
    print(f"Rows with all six vitals: {quality['rows_with_all_six_vitals_rate']:.0%}")
    print(f"Feature shape: {features.shape}")
    print(features[["patient_id", "qsofa_score", "shock_index", "n_vitals_missing"]])
    print(f"Catalogued engineered features: {len(inventory)}")
=======
#!/usr/bin/env python3
"""
examples/run_feature_pipeline.py
=================================
End-to-end demonstration of the sepsis_vitals pipeline without needing
real partner data.

Run from the repo root:
    PYTHONPATH=src python3 examples/run_feature_pipeline.py

Or after installing:
    python3 examples/run_feature_pipeline.py
"""

import sys
import textwrap
import numpy as np
import pandas as pd

# ── Allow running without installing ──────────────────────────────────────────
try:
    from sepsis_vitals import build_feature_set
    from sepsis_vitals.data_quality import (
        summarize_vitals_quality,
        check_data_contract,
        temporal_quality,
    )
    from sepsis_vitals.scores import compute_scores
except ImportError:
    print("Run with:  PYTHONPATH=src python3 examples/run_feature_pipeline.py")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    width = 62
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def synthetic_vitals(n_patients: int = 4, obs_per: int = 8, seed: int = 0) -> pd.DataFrame:
    """Generate realistic synthetic vitals with deliberate missingness."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-06-01 06:00")

    profiles = [
        {"name": "Mild presentation",   temp_bias: 0.5,  hr_bias: 15,  rr_bias: 3,  sbp_bias: -5},
        {"name": "Moderate sepsis",      temp_bias: 1.5,  hr_bias: 30,  rr_bias: 8,  sbp_bias: -20},
        {"name": "Critical (qSOFA≥2)",   temp_bias: 2.0,  hr_bias: 45,  rr_bias: 12, sbp_bias: -40},
        {"name": "Recovery trajectory",  temp_bias: -0.5, hr_bias: -5,  rr_bias: -2, sbp_bias: 10},
    ]

    for pid, profile in enumerate(profiles[:n_patients]):
        for i in range(obs_per):
            # Simulate trajectory — deterioration or recovery
            t = i / (obs_per - 1)
            direction = 1 if pid < 3 else -1  # first 3 worsen, last recovers

            temp = 37.0 + profile["temp_bias"] * t * direction + rng.normal(0, 0.2)
            hr   = 80   + profile["hr_bias"]   * t * direction + rng.normal(0, 5)
            rr   = 16   + profile["rr_bias"]   * t * direction + rng.normal(0, 1)
            sbp  = 120  + profile["sbp_bias"]  * t * direction + rng.normal(0, 5)
            spo2 = max(88, 98 - abs(profile["rr_bias"]) * t * direction * 0.3 + rng.normal(0, 1))
            gcs  = max(3, min(15, 15 - max(0, profile["hr_bias"] / 50 * t * direction) * 2 + rng.normal(0, 0.5)))

            row = {
                "patient_id":  f"PT-{pid:03d}",
                "site_id":     f"SITE-{'AB'[pid % 2]}",
                "timestamp":   base + pd.Timedelta(hours=pid * 48 + i * 2),
                "age_years":   [68, 42, 55, 71][pid],
                "temperature": round(np.clip(temp, 35, 42), 1),
                "heart_rate":  int(np.clip(hr, 40, 200)),
                "resp_rate":   int(np.clip(rr, 8, 50)),
                "sbp":         int(np.clip(sbp, 60, 200)),
                "spo2":        round(np.clip(spo2, 82, 100), 1),
                "gcs":         int(np.clip(round(gcs), 3, 15)),
            }

            # Introduce realistic missingness (SpO2 least reliable in LMIC)
            if rng.random() < 0.25:
                row["spo2"] = np.nan
            if rng.random() < 0.10:
                row["gcs"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# Trick: allow dict with colons in keys for profile definition
def _make_profile(name, temp_bias, hr_bias, rr_bias, sbp_bias):
    return dict(name=name, temp_bias=temp_bias, hr_bias=hr_bias,
                rr_bias=rr_bias, sbp_bias=sbp_bias)

# Patch profiles to use function
import types

def synthetic_vitals_fixed(n_patients: int = 4, obs_per: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-06-01 06:00")

    profiles = [
        _make_profile("Mild presentation",  0.5,  15,  3,  -5),
        _make_profile("Moderate sepsis",     1.5,  30,  8,  -20),
        _make_profile("Critical qSOFA≥2",    2.0,  45, 12,  -40),
        _make_profile("Recovery",           -0.5,  -5, -2,   10),
    ]

    for pid, profile in enumerate(profiles[:n_patients]):
        for i in range(obs_per):
            t = i / max(obs_per - 1, 1)
            direction = 1.0 if pid < 3 else -1.0

            temp = 37.0 + profile["temp_bias"] * t * direction + rng.normal(0, 0.2)
            hr   = 80   + profile["hr_bias"]   * t * direction + rng.normal(0, 5)
            rr   = 16   + profile["rr_bias"]   * t * direction + rng.normal(0, 1)
            sbp  = 120  + profile["sbp_bias"]  * t * direction + rng.normal(0, 5)
            spo2 = 98   - abs(profile["rr_bias"]) * t * direction * 0.3 + rng.normal(0, 1)
            gcs_raw = 15 - max(0, profile["hr_bias"] / 50 * t * direction) * 2 + rng.normal(0, 0.5)

            row = {
                "patient_id": f"PT-{pid:03d}",
                "site_id":    f"SITE-{'AB'[pid % 2]}",
                "timestamp":  base + pd.Timedelta(hours=pid * 48 + i * 2),
                "age_years":  [68, 42, 55, 71][pid],
                "temperature":round(float(np.clip(temp, 35, 42)), 1),
                "heart_rate": int(np.clip(hr, 40, 200)),
                "resp_rate":  int(np.clip(rr, 8, 50)),
                "sbp":        int(np.clip(sbp, 60, 200)),
                "spo2":       round(float(np.clip(spo2, 82, 100)), 1),
                "gcs":        int(np.clip(round(gcs_raw), 3, 15)),
            }

            if rng.random() < 0.25:
                row["spo2"] = np.nan
            if rng.random() < 0.10:
                row["gcs"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ── Main demo ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║          SEPSIS VITALS · Feature Pipeline Demo              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── 1. Generate synthetic data ──────────────────────────────────────────
    banner("1 · Synthetic vitals (4 patients × 8 observations)")
    raw = synthetic_vitals_fixed(n_patients=4, obs_per=8)
    print(raw[["patient_id", "timestamp", "temperature", "heart_rate",
               "resp_rate", "sbp", "spo2", "gcs"]].to_string(index=False))

    # ── 2. Data contract check ─────────────────────────────────────────────
    banner("2 · Data contract validation")
    contract = check_data_contract(raw)
    status = "✓ PASS" if contract["passed"] else "✗ FAIL"
    print(f"  Status : {status}")
    print(f"  Vitals : {contract['vitals_present']}")
    if contract["warnings"]:
        for w in contract["warnings"]:
            print(f"  ⚠  {w}")
    if contract["errors"]:
        for e in contract["errors"]:
            print(f"  ✗  {e}")

    # ── 3. Data quality summary ────────────────────────────────────────────
    banner("3 · Per-vital quality summary")
    quality = summarize_vitals_quality(raw)
    print(quality[["n_present", "completeness_pct", "median",
                   "n_hard_outliers", "n_soft_outliers"]].to_string())

    # ── 4. Temporal quality ────────────────────────────────────────────────
    banner("4 · Temporal quality")
    tq = temporal_quality(raw)
    for k, v in tq.items():
        print(f"  {k:<30} {v}")

    # ── 5. Feature engineering ─────────────────────────────────────────────
    banner("5 · Feature engineering (rolling_window=3)")
    features = build_feature_set(
        raw,
        patient_col="patient_id",
        time_col="timestamp",
        age_col="age_years",
        rolling_window=3,
        score_cols=True,
    )
    print(f"  Input shape  : {raw.shape}")
    print(f"  Output shape : {features.shape}")
    print(f"  New columns  : {features.shape[1] - raw.shape[1]}")

    # Sample of derived features for last patient observation
    last = features.groupby("patient_id").tail(1)
    print("\n  Last observation per patient (key features):")
    cols = ["patient_id", "heart_rate", "heart_rate_delta", "heart_rate_roll_mean",
            "qsofa", "sirs_count", "shock_index", "risk_level", "alert_flag"]
    print(last[[c for c in cols if c in last.columns]].to_string(index=False))

    # ── 6. Individual score examples ───────────────────────────────────────
    banner("6 · Per-row clinical scores")
    scenarios = [
        ("Normal",    {"temperature": 37.0, "heart_rate": 75, "resp_rate": 16, "sbp": 120, "spo2": 98, "gcs": 15}),
        ("Moderate",  {"temperature": 38.5, "heart_rate": 105, "resp_rate": 22, "sbp": 108, "spo2": 95, "gcs": 15}),
        ("High risk", {"temperature": 39.1, "heart_rate": 118, "resp_rate": 26, "sbp": 92, "spo2": 91, "gcs": 13}),
        ("Critical",  {"temperature": 39.8, "heart_rate": 135, "resp_rate": 32, "sbp": 78, "spo2": 87, "gcs": 11}),
    ]
    for name, vitals in scenarios:
        bundle = compute_scores(vitals)
        si_str = f"{bundle.shock_index:.2f}" if bundle.shock_index else "—"
        alert  = "⚠ ALERT" if bundle.alert_flag else "  ok"
        print(
            f"  {name:<12} qSOFA={bundle.qsofa}  SIRS={bundle.sirs_count}"
            f"  SI={si_str:<5}  NEWS2={bundle.news2_style:<3}"
            f"  → {bundle.risk_level.upper():<8} {alert}"
        )

    # ── 7. Missingness feature check ───────────────────────────────────────
    banner("7 · Missingness signals")
    miss_cols = [c for c in features.columns if "_missing" in c and "roll" not in c]
    print(f"  Missingness indicator columns ({len(miss_cols)}): {miss_cols}")
    print(f"\n  Mean missingness per vital:")
    for col in miss_cols:
        mean_miss = features[col].mean()
        bar = "█" * int(mean_miss * 20)
        print(f"    {col:<25} {mean_miss:5.1%}  {bar}")

    # ── 8. What comes next ─────────────────────────────────────────────────
    banner("Next: Phase 1 model training")
    print(textwrap.dedent("""
      1. Collect adjudicated partner data → label_col='sepsis_label'
      2. from sepsis_vitals.model_scaffold import (
             prepare_Xy, train_lgbm, evaluate_model, find_threshold,
             leave_one_site_out, ModelCard
         )
      3. X, y = prepare_Xy(features, label_col='sepsis_label')
      4. lgbm, cal = train_lgbm(X, y)
      5. metrics = evaluate_model(cal, X_val, y_val)
      6. threshold = find_threshold(cal, X_val, y_val)
      7. shap_df   = compute_shap(cal, X_val)
      8. card = ModelCard(**metrics, **threshold, features_used=list(X.columns))
      9. card.save("model_card.json")
    """))

    print("  Feature pipeline demo complete.\n")
>>>>>>> phase-0-expansion


if __name__ == "__main__":
    main()
