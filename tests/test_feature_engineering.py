import numpy as np
import pandas as pd

from sepsis_vitals.features import build_feature_set, get_feature_inventory


def make_raw_vitals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["A", "A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 10:00",
                    "2026-01-01 12:00",
                    "2026-01-01 09:00",
                    "2026-01-01 11:00",
                ]
            ),
            "temperature": [37.0, 38.5, 39.2, 36.8, np.nan],
            "heart_rate": [82, 105, 128, 78, 92],
            "resp_rate": [18, 23, 31, 16, 19],
            "sbp": [122, 99, 86, 130, 112],
            "spo2": [98, 94, 88, 99, np.nan],
            "gcs": [15, 14, 13, 15, 15],
            "age_years": [44, 44, 44, 8, 8],
        }
    )


def test_build_feature_set_adds_core_clinical_features_and_preserves_rows():
    raw = make_raw_vitals()

    engineered = build_feature_set(raw, patient_col="patient_id", time_col="timestamp")

    assert len(engineered) == len(raw)
    assert engineered.loc[1, "qsofa_score"] == 3
    assert engineered.loc[1, "qsofa_positive"] == 1
    assert engineered.loc[2, "sirs_partial_positive"] == 1
    assert engineered.loc[2, "flag_si_critical"] == 1
    assert engineered.loc[4, "temperature_missing"] == 1
    assert engineered.loc[4, "spo2_missing"] == 1
    assert engineered.loc[4, "n_vitals_missing"] == 2


def test_build_feature_set_computes_patient_ordered_deltas_and_rolling_features():
    raw = make_raw_vitals()

    engineered = build_feature_set(
        raw,
        patient_col="patient_id",
        time_col="timestamp",
        rolling_window=3,
    )

    assert engineered.loc[1, "heart_rate_delta"] == 23
    assert engineered.loc[2, "sbp_delta"] == -13
    assert engineered.loc[2, "hours_since_last"] == 2
    assert engineered.loc[2, "flag_rr_rapid_rise"] == 1
    assert engineered.loc[2, "heart_rate_roll3_mean"] == 105
    assert pd.isna(engineered.loc[3, "heart_rate_delta"])


def test_feature_inventory_documents_engineered_columns():
    engineered = build_feature_set(make_raw_vitals())

    inventory = get_feature_inventory(engineered)

    assert {"feature", "type", "clinical_meaning", "range"}.issubset(inventory.columns)
    assert "qsofa_score" in set(inventory["feature"])
    assert "shock_index" in set(inventory["feature"])
    assert "n_vitals_missing" in set(inventory["feature"])
