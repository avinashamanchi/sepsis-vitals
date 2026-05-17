"""
tests/test_features.py – Tests for the feature engineering pipeline.
tests/test_data_quality.py – Tests for the data quality auditor.
(Combined in one file for compactness.)
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from sepsis_vitals.features import build_feature_set, CORE_VITALS
from sepsis_vitals.data_quality import (
    summarize_vitals_quality,
    check_data_contract,
    temporal_quality,
    generate_quality_report,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_df(n_patients: int = 3, obs_per: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic vitals DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2024-01-01 08:00")

    for pid in range(n_patients):
        for i in range(obs_per):
            rows.append({
                "patient_id": f"PT-{pid:03d}",
                "timestamp":  base_ts + pd.Timedelta(hours=pid * 24 + i * 2),
                "age_years":  int(rng.integers(18, 80)),
                "temperature":round(float(rng.uniform(36.0, 39.5)), 1),
                "heart_rate": int(rng.integers(60, 130)),
                "resp_rate":  int(rng.integers(12, 28)),
                "sbp":        int(rng.integers(85, 150)),
                "spo2":       int(rng.integers(88, 100)),
                "gcs":        int(rng.integers(12, 16)),
            })

    return pd.DataFrame(rows)


def make_sparse_df() -> pd.DataFrame:
    """DataFrame with deliberate missingness."""
    df = make_df()
    rng = np.random.default_rng(0)
    for v in ["spo2", "gcs"]:
        mask = rng.random(len(df)) < 0.4
        df.loc[mask, v] = np.nan
    return df


# ──────────────────────────────────────────────────────────────────────────────
# build_feature_set
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureSet:

    def test_returns_dataframe(self):
        df = make_df()
        out = build_feature_set(df)
        assert isinstance(out, pd.DataFrame)

    def test_row_count_preserved(self):
        df = make_df(n_patients=3, obs_per=5)
        out = build_feature_set(df)
        assert len(out) == len(df)

    def test_missingness_indicators_added(self):
        df = make_df()
        out = build_feature_set(df)
        for v in CORE_VITALS:
            assert f"{v}_missing" in out.columns

    def test_n_vitals_missing_correct(self):
        df = make_df()
        # Remove 2 vitals entirely
        df = df.drop(columns=["spo2", "gcs"])
        out = build_feature_set(df)
        assert (out["n_vitals_missing"] == 2).all()

    def test_delta_features_added(self):
        df = make_df()
        out = build_feature_set(df)
        for v in CORE_VITALS:
            if v in df.columns:
                assert f"{v}_delta" in out.columns

    def test_first_row_per_patient_delta_is_nan(self):
        df = make_df(n_patients=2, obs_per=4)
        out = build_feature_set(df)
        first_rows = out.groupby("patient_id").head(1)
        assert first_rows["heart_rate_delta"].isna().all()

    def test_rolling_features_added(self):
        df = make_df()
        out = build_feature_set(df)
        assert "heart_rate_roll_mean" in out.columns
        assert "temperature_roll_std" in out.columns

    def test_rolling_window_clipped(self):
        df = make_df()
        # Window > 12 should be silently clipped
        out = build_feature_set(df, rolling_window=999)
        assert "heart_rate_roll_mean" in out.columns

    def test_score_cols_present(self):
        df = make_df()
        out = build_feature_set(df, score_cols=True)
        assert "qsofa" in out.columns
        assert "risk_level" in out.columns

    def test_score_cols_absent_when_disabled(self):
        df = make_df()
        out = build_feature_set(df, score_cols=False)
        assert "qsofa" not in out.columns

    def test_peds_z_scores_for_young_patients(self):
        df = make_df()
        df["age_years"] = 5  # all pediatric
        out = build_feature_set(df, age_col="age_years")
        assert "heart_rate_peds_z" in out.columns
        assert out["heart_rate_peds_z"].notna().any()

    def test_no_peds_z_for_adults(self):
        df = make_df()
        df["age_years"] = 40
        out = build_feature_set(df, age_col="age_years")
        if "heart_rate_peds_z" in out.columns:
            assert out["heart_rate_peds_z"].isna().all()

    def test_peds_skipped_when_age_col_none(self):
        df = make_df()
        out = build_feature_set(df, age_col=None)
        assert "heart_rate_peds_z" not in out.columns

    def test_episode_aggregates_warned(self):
        df = make_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_feature_set(df, include_episode_aggregates=True)
            assert any("future information" in str(warning.message) for warning in w)

    def test_episode_aggregates_present_when_enabled(self):
        df = make_df()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            out = build_feature_set(df, include_episode_aggregates=True)
        assert "heart_rate_ep_mean" in out.columns

    def test_raises_on_missing_patient_col(self):
        df = make_df().drop(columns=["patient_id"])
        with pytest.raises(ValueError, match="patient_col"):
            build_feature_set(df)

    def test_raises_on_missing_time_col(self):
        df = make_df().drop(columns=["timestamp"])
        with pytest.raises(ValueError, match="time_col"):
            build_feature_set(df)

    def test_raises_on_too_few_vitals(self):
        df = make_df()[["patient_id", "timestamp", "heart_rate", "sbp"]]
        with pytest.raises(ValueError, match="At least 3"):
            build_feature_set(df)

    def test_sorted_by_patient_time(self):
        df = make_df().sample(frac=1, random_state=7)  # shuffle
        out = build_feature_set(df)
        for pid, grp in out.groupby("patient_id"):
            assert grp["timestamp"].is_monotonic_increasing

    def test_obs_gap_min_present(self):
        df = make_df()
        out = build_feature_set(df)
        assert "obs_gap_min" in out.columns
        # First obs per patient should be NaN
        first = out.groupby("patient_id").head(1)
        assert first["obs_gap_min"].isna().all()

    def test_sparse_data_does_not_crash(self):
        df = make_sparse_df()
        out = build_feature_set(df)
        assert len(out) == len(df)


# ──────────────────────────────────────────────────────────────────────────────
# summarize_vitals_quality
# ──────────────────────────────────────────────────────────────────────────────

class TestSummarizeVitalsQuality:

    def test_returns_dataframe(self):
        df = make_df()
        out = summarize_vitals_quality(df)
        assert isinstance(out, pd.DataFrame)

    def test_index_is_vital_names(self):
        out = summarize_vitals_quality(make_df())
        assert set(out.index) == set(CORE_VITALS)

    def test_completeness_100_when_no_missings(self):
        df = make_df()
        out = summarize_vitals_quality(df)
        assert (out["completeness_pct"] == 100.0).all()

    def test_completeness_reflects_missing(self):
        df = make_df()
        df.loc[:, "temperature"] = np.nan
        out = summarize_vitals_quality(df)
        assert out.loc["temperature", "completeness_pct"] == 0.0

    def test_missing_vital_col_reported_as_zero(self):
        df = make_df().drop(columns=["gcs"])
        out = summarize_vitals_quality(df)
        assert out.loc["gcs", "n_present"] == 0

    def test_hard_outlier_detected(self):
        df = make_df()
        df.loc[0, "temperature"] = 60.0  # physically impossible
        out = summarize_vitals_quality(df)
        assert out.loc["temperature", "n_hard_outliers"] >= 1

    def test_median_reasonable(self):
        df = make_df()
        out = summarize_vitals_quality(df)
        assert 36 <= out.loc["temperature", "median"] <= 40
        assert 60 <= out.loc["heart_rate", "median"] <= 140


# ──────────────────────────────────────────────────────────────────────────────
# check_data_contract
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckDataContract:

    def test_passes_on_good_data(self):
        result = check_data_contract(make_df())
        assert result["passed"] is True
        assert result["errors"] == []

    def test_fails_on_missing_patient_col(self):
        df = make_df().drop(columns=["patient_id"])
        result = check_data_contract(df)
        assert result["passed"] is False
        assert any("patient_id" in e for e in result["errors"])

    def test_fails_on_missing_time_col(self):
        df = make_df().drop(columns=["timestamp"])
        result = check_data_contract(df)
        assert result["passed"] is False

    def test_fails_on_too_few_vitals(self):
        df = make_df()[["patient_id", "timestamp", "heart_rate", "sbp"]]
        result = check_data_contract(df)
        assert result["passed"] is False

    def test_warns_on_duplicate_rows(self):
        df = make_df()
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)
        result = check_data_contract(df)
        assert any("duplicate" in w.lower() for w in result["warnings"])

    def test_warns_on_low_completeness(self):
        df = make_df()
        df["temperature"] = np.nan
        df.loc[0, "temperature"] = 37.0  # 1 present — very low completeness
        result = check_data_contract(df)
        assert any("temperature" in e for e in result["errors"] + result["warnings"])

    def test_warns_on_hard_outliers(self):
        df = make_df()
        df.loc[0, "sbp"] = 500  # impossible
        result = check_data_contract(df)
        assert any("sbp" in w for w in result["warnings"])

    def test_vitals_present_list_correct(self):
        df = make_df().drop(columns=["gcs"])
        result = check_data_contract(df)
        assert "gcs" not in result["vitals_present"]


# ──────────────────────────────────────────────────────────────────────────────
# temporal_quality
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalQuality:

    def test_returns_dict(self):
        result = temporal_quality(make_df())
        assert isinstance(result, dict)

    def test_n_patients_correct(self):
        df = make_df(n_patients=4, obs_per=3)
        result = temporal_quality(df)
        assert result["n_patients"] == 4

    def test_n_observations_correct(self):
        df = make_df(n_patients=2, obs_per=5)
        result = temporal_quality(df)
        assert result["n_observations"] == 10

    def test_gap_min_is_120_for_2h_intervals(self):
        df = make_df(n_patients=1, obs_per=4)
        result = temporal_quality(df)
        # make_df uses 2h intervals → 120 min gaps
        assert result["gap_min_median_min"] == pytest.approx(120.0, abs=1.0)

    def test_missing_cols_returns_error(self):
        df = make_df().drop(columns=["timestamp"])
        result = temporal_quality(df)
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────────────
# generate_quality_report
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateQualityReport:

    def test_returns_dict(self):
        r = generate_quality_report(make_df())
        assert isinstance(r, dict)

    def test_has_expected_keys(self):
        r = generate_quality_report(make_df())
        for key in ("contract", "temporal", "vitals", "overall_status"):
            assert key in r

    def test_pass_on_clean_data(self):
        r = generate_quality_report(make_df(), site_id="SITE-A")
        assert r["overall_status"] == "PASS"
        assert r["site_id"] == "SITE-A"

    def test_fail_on_bad_data(self):
        df = make_df()[["patient_id", "timestamp", "heart_rate"]]  # too few vitals
        r = generate_quality_report(df)
        assert r["overall_status"] == "FAIL"
