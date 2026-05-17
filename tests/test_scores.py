"""
tests/test_scores.py – Unit tests for all clinical scoring functions.
"""

import pytest
from sepsis_vitals.scores import (
    qsofa,
    partial_sirs,
    shock_index,
    news2_style,
    uva_style,
    classify_risk,
    compute_scores,
    ScoreBundle,
)


# ──────────────────────────────────────────────────────────────────────────────
# qSOFA
# ──────────────────────────────────────────────────────────────────────────────

class TestQsofa:
    def test_zero_on_normal_vitals(self):
        score, _ = qsofa({"resp_rate": 16, "gcs": 15, "sbp": 120})
        assert score == 0

    def test_one_on_elevated_rr(self):
        score, flags = qsofa({"resp_rate": 22, "gcs": 15, "sbp": 120})
        assert score == 1
        assert flags["qsofa_rr"] is True

    def test_one_on_low_gcs(self):
        score, flags = qsofa({"resp_rate": 16, "gcs": 13, "sbp": 120})
        assert score == 1
        assert flags["qsofa_gcs"] is True

    def test_one_on_low_sbp(self):
        score, flags = qsofa({"resp_rate": 16, "gcs": 15, "sbp": 100})
        assert score == 1
        assert flags["qsofa_sbp"] is True

    def test_three_critical(self):
        score, _ = qsofa({"resp_rate": 28, "gcs": 12, "sbp": 88})
        assert score == 3

    def test_missing_vital_not_scored(self):
        # Only resp_rate provided — only that component can fire
        score, flags = qsofa({"resp_rate": 24})
        assert score == 1
        assert flags.get("qsofa_gcs") is False
        assert flags.get("qsofa_sbp") is False

    def test_rr_boundary_21_not_fired(self):
        score, _ = qsofa({"resp_rate": 21})
        assert score == 0

    def test_rr_boundary_22_fires(self):
        score, _ = qsofa({"resp_rate": 22})
        assert score == 1

    def test_sbp_boundary_100_fires(self):
        score, flags = qsofa({"sbp": 100})
        assert flags["qsofa_sbp"] is True

    def test_sbp_boundary_101_not_fired(self):
        score, flags = qsofa({"sbp": 101})
        assert flags["qsofa_sbp"] is False

    def test_empty_vitals(self):
        score, _ = qsofa({})
        assert score == 0


# ──────────────────────────────────────────────────────────────────────────────
# Partial SIRS
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialSirs:
    def test_zero_on_normal(self):
        count, _ = partial_sirs({"temperature": 37.0, "heart_rate": 80, "resp_rate": 16})
        assert count == 0

    def test_high_temp_fires(self):
        count, flags = partial_sirs({"temperature": 38.4})
        assert count == 1
        assert flags["sirs_temp"] is True

    def test_low_temp_fires(self):
        count, flags = partial_sirs({"temperature": 35.8})
        assert count == 1
        assert flags["sirs_temp"] is True

    def test_hr_boundary(self):
        _, flags = partial_sirs({"heart_rate": 91})
        assert flags["sirs_hr"] is True
        _, flags2 = partial_sirs({"heart_rate": 90})
        assert flags2["sirs_hr"] is False

    def test_rr_boundary(self):
        _, flags = partial_sirs({"resp_rate": 21})
        assert flags["sirs_rr"] is True
        _, flags2 = partial_sirs({"resp_rate": 20})
        assert flags2["sirs_rr"] is False

    def test_full_sirs(self):
        count, _ = partial_sirs({
            "temperature": 39.0, "heart_rate": 105, "resp_rate": 24
        })
        assert count == 3

    def test_empty_returns_zero(self):
        count, _ = partial_sirs({})
        assert count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Shock Index
# ──────────────────────────────────────────────────────────────────────────────

class TestShockIndex:
    def test_normal(self):
        si = shock_index({"heart_rate": 70, "sbp": 120})
        assert abs(si - 70/120) < 0.001

    def test_elevated(self):
        si = shock_index({"heart_rate": 110, "sbp": 90})
        assert si >= 1.0

    def test_missing_hr_returns_none(self):
        assert shock_index({"sbp": 120}) is None

    def test_missing_sbp_returns_none(self):
        assert shock_index({"heart_rate": 80}) is None

    def test_zero_sbp_returns_none(self):
        assert shock_index({"heart_rate": 80, "sbp": 0}) is None

    def test_rounded_to_3dp(self):
        si = shock_index({"heart_rate": 100, "sbp": 3})
        assert isinstance(si, float)
        assert len(str(si).split(".")[-1]) <= 3


# ──────────────────────────────────────────────────────────────────────────────
# NEWS2-style
# ──────────────────────────────────────────────────────────────────────────────

class TestNews2Style:
    def test_zero_on_perfect_vitals(self):
        score = news2_style({
            "resp_rate": 16, "spo2": 98, "sbp": 120,
            "heart_rate": 75, "temperature": 37.2, "gcs": 15
        })
        assert score == 0

    def test_high_rr_scores(self):
        assert news2_style({"resp_rate": 25}) == 3

    def test_low_spo2_scores(self):
        assert news2_style({"spo2": 91}) == 3

    def test_low_sbp_scores(self):
        assert news2_style({"sbp": 88}) == 3

    def test_altered_gcs_scores(self):
        assert news2_style({"gcs": 14}) == 3

    def test_low_rr_scores(self):
        assert news2_style({"resp_rate": 8}) == 3

    def test_composite_high_risk(self):
        score = news2_style({
            "resp_rate": 26, "spo2": 92, "sbp": 95,
            "heart_rate": 115, "temperature": 38.5, "gcs": 14
        })
        assert score >= 7

    def test_empty_returns_zero(self):
        assert news2_style({}) == 0


# ──────────────────────────────────────────────────────────────────────────────
# UVA-style
# ──────────────────────────────────────────────────────────────────────────────

class TestUvaStyle:
    def test_zero_on_normal(self):
        assert uva_style({"resp_rate": 16, "sbp": 120, "temperature": 37.2, "gcs": 15}) == 0

    def test_high_rr_gets_2(self):
        assert uva_style({"resp_rate": 32}) == 2

    def test_low_sbp_gets_3(self):
        assert uva_style({"sbp": 88}) == 3

    def test_low_temp_gets_2(self):
        assert uva_style({"temperature": 35.5}) == 2

    def test_low_gcs_gets_2(self):
        assert uva_style({"gcs": 12}) == 2

    def test_critical_combination(self):
        score = uva_style({
            "resp_rate": 35, "sbp": 82, "temperature": 35.0, "gcs": 10
        })
        assert score >= 9

    def test_empty_returns_zero(self):
        assert uva_style({}) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Risk classification
# ──────────────────────────────────────────────────────────────────────────────

class TestClassifyRisk:
    def test_low_all_zeros(self):
        level, alert = classify_risk(0, 0, None, 0)
        assert level == "low"
        assert alert is False

    def test_moderate_qsofa_1(self):
        level, alert = classify_risk(1, 0, None, 0)
        assert level == "moderate"
        assert alert is False

    def test_moderate_sirs_2(self):
        level, alert = classify_risk(0, 2, None, 0)
        assert level == "moderate"

    def test_high_qsofa_2(self):
        level, alert = classify_risk(2, 0, None, 0)
        assert level == "high"
        assert alert is True

    def test_high_shock_index_1(self):
        level, alert = classify_risk(0, 0, 1.05, 0)
        assert level == "high"
        assert alert is True

    def test_critical_qsofa_3(self):
        level, alert = classify_risk(3, 0, None, 0)
        assert level == "critical"
        assert alert is True

    def test_critical_news2_7(self):
        level, alert = classify_risk(0, 0, None, 7)
        assert level == "critical"
        assert alert is True

    def test_critical_shock_index_1_4(self):
        level, alert = classify_risk(0, 0, 1.4, 0)
        assert level == "critical"
        assert alert is True


# ──────────────────────────────────────────────────────────────────────────────
# compute_scores (integration)
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeScores:
    def test_returns_score_bundle(self):
        result = compute_scores({
            "temperature": 38.9,
            "heart_rate": 118,
            "resp_rate": 26,
            "sbp": 92,
            "spo2": 93,
            "gcs": 14,
        })
        assert isinstance(result, ScoreBundle)

    def test_critical_patient(self):
        result = compute_scores({
            "heart_rate": 130, "resp_rate": 30, "sbp": 80, "gcs": 12
        })
        assert result.risk_level in ("high", "critical")
        assert result.alert_flag is True

    def test_normal_patient(self):
        result = compute_scores({
            "temperature": 37.0, "heart_rate": 75, "resp_rate": 16,
            "sbp": 120, "spo2": 98, "gcs": 15
        })
        assert result.risk_level == "low"
        assert result.alert_flag is False

    def test_as_dict_has_expected_keys(self):
        d = compute_scores({"heart_rate": 80, "sbp": 120}).as_dict()
        for key in ("qsofa", "sirs_count", "shock_index", "news2_style",
                    "uva_style", "risk_level", "alert_flag"):
            assert key in d

    def test_component_flags_populated(self):
        result = compute_scores({"resp_rate": 25, "sbp": 95, "gcs": 14})
        assert "qsofa_rr" in result.component_flags
        assert result.component_flags["qsofa_rr"] is True

    def test_partial_vitals_still_scores(self):
        # Only 2 vitals — still should return a bundle
        result = compute_scores({"heart_rate": 130, "sbp": 75})
        assert isinstance(result, ScoreBundle)
        assert result.shock_index is not None
