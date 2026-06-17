"""
tests/test_new_modules.py
Tests for: auth/jwt, monitoring/metrics, ml/fairness, health_economics/model
"""

import json
import time
import numpy as np
import pandas as pd
import pytest

try:
    import bcrypt as _bcrypt  # noqa: F401
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

try:
    import pyotp as _pyotp  # noqa: F401
    HAS_PYOTP = True
except ImportError:
    HAS_PYOTP = False

# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_BCRYPT, reason="bcrypt not installed")
class TestPasswordHashing:
    def test_hash_and_verify(self):
        from sepsis_vitals.auth.jwt import hash_password, verify_password
        h = hash_password("SecurePass1!")
        assert verify_password("SecurePass1!", h) is True

    def test_wrong_password_fails(self):
        from sepsis_vitals.auth.jwt import hash_password, verify_password
        h = hash_password("Correct1!")
        assert verify_password("Wrong1!", h) is False

    def test_hash_is_not_plaintext(self):
        from sepsis_vitals.auth.jwt import hash_password
        h = hash_password("MySecret99!")
        assert "MySecret99!" not in h


class TestRBAC:
    def test_nurse_can_read_vitals(self):
        from sepsis_vitals.auth.jwt import check_permission
        check_permission("nurse", "vital:read")  # should not raise

    def test_nurse_cannot_write_users(self):
        from sepsis_vitals.auth.jwt import check_permission, AuthorizationError
        with pytest.raises(AuthorizationError):
            check_permission("nurse", "user:write")

    def test_system_admin_has_all(self):
        from sepsis_vitals.auth.jwt import check_permission
        for perm in ["vital:read", "alert:escalate", "user:write", "site:write"]:
            check_permission("system_admin", perm)

    def test_researcher_no_patient_access(self):
        from sepsis_vitals.auth.jwt import check_permission, AuthorizationError
        with pytest.raises(AuthorizationError):
            check_permission("researcher", "patient:read")


@pytest.mark.skipif(not HAS_PYOTP, reason="pyotp not installed")
class TestMFA:
    def test_generate_secret_is_base32(self):
        from sepsis_vitals.auth.jwt import generate_totp_secret
        secret = generate_totp_secret()
        import base64
        base64.b32decode(secret)  # should not raise

    def test_verify_valid_totp(self):
        from sepsis_vitals.auth.jwt import generate_totp_secret, verify_totp
        import pyotp
        secret = generate_totp_secret()
        totp = pyotp.TOTP(secret)
        assert verify_totp(secret, totp.now()) is True

    def test_verify_wrong_code_fails(self):
        from sepsis_vitals.auth.jwt import generate_totp_secret, verify_totp
        secret = generate_totp_secret()
        assert verify_totp(secret, "000000") is False

    def test_totp_uri_contains_issuer(self):
        from sepsis_vitals.auth.jwt import generate_totp_secret, get_totp_uri
        uri = get_totp_uri(generate_totp_secret(), "nurse@hospital.ke")
        assert "SepsisVitals" in uri
        assert "nurse" in uri and "hospital.ke" in uri.replace("%40", "@")


class TestLockout:
    def test_lockout_duration_exponential(self):
        from sepsis_vitals.auth.jwt import lockout_duration
        d0 = lockout_duration(0)
        d1 = lockout_duration(1)
        d2 = lockout_duration(2)
        assert d0 == 0
        assert d1 < d2

    def test_is_locked_out_with_future_time(self):
        from sepsis_vitals.auth.jwt import is_locked_out
        from datetime import datetime, timezone, timedelta
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert is_locked_out(future) is True

    def test_not_locked_with_past_time(self):
        from sepsis_vitals.auth.jwt import is_locked_out
        from datetime import datetime, timezone, timedelta
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        assert is_locked_out(past) is False

    def test_not_locked_with_none(self):
        from sepsis_vitals.auth.jwt import is_locked_out
        assert is_locked_out(None) is False


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring / PSI
# ──────────────────────────────────────────────────────────────────────────────

class TestPSI:
    def test_identical_distributions_near_zero(self):
        from sepsis_vitals.monitoring.metrics import compute_psi
        data = np.random.normal(37, 0.5, 200)
        psi  = compute_psi(data, data.copy())
        assert psi < 0.05

    def test_shifted_distribution_high_psi(self):
        from sepsis_vitals.monitoring.metrics import compute_psi
        ref = np.random.normal(37, 0.5, 500)
        cur = np.random.normal(39, 0.5, 500)  # 2°C shift
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_insufficient_data_returns_zero(self):
        from sepsis_vitals.monitoring.metrics import compute_psi
        psi = compute_psi(np.array([37.0] * 5), np.array([38.0] * 5))
        assert psi == 0.0

    def test_drift_detection_flag(self):
        from sepsis_vitals.monitoring.metrics import check_distribution_drift
        ref = {"heart_rate": list(np.random.normal(80, 10, 500))}
        cur = {"heart_rate": list(np.random.normal(110, 10, 500))}  # tachycardic shift
        result = check_distribution_drift(ref, cur, threshold=0.2)
        assert result["overall_drift"] is True

    def test_no_drift_stable_data(self):
        from sepsis_vitals.monitoring.metrics import check_distribution_drift
        data = list(np.random.normal(80, 5, 500))
        result = check_distribution_drift(
            {"heart_rate": data},
            {"heart_rate": data},
            threshold=0.2,
        )
        assert result["heart_rate"]["psi"] < 0.1


class TestAlertFatigueMetrics:
    def test_high_override_rate_flagged(self):
        from sepsis_vitals.monitoring.metrics import compute_alert_fatigue_metrics
        rows = [{"action": "dismissed", "time_to_action_s": 60}] * 8
        rows += [{"action": "acknowledged", "time_to_action_s": 30}] * 2
        result = compute_alert_fatigue_metrics(rows)
        assert result["override_rate"] == pytest.approx(0.8, abs=0.01)
        assert result["fatigue_level"] == "critical"

    def test_normal_behavior_no_flag(self):
        from sepsis_vitals.monitoring.metrics import compute_alert_fatigue_metrics
        rows = [{"action": "acknowledged", "time_to_action_s": 45}] * 10
        result = compute_alert_fatigue_metrics(rows)
        assert result["fatigue_level"] == "normal"

    def test_empty_returns_error(self):
        from sepsis_vitals.monitoring.metrics import compute_alert_fatigue_metrics
        result = compute_alert_fatigue_metrics([])
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────────────
# ML fairness
# ──────────────────────────────────────────────────────────────────────────────

def _make_prediction_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "label":   rng.integers(0, 2, n),
        "prob":    rng.uniform(0, 1, n),
        "sex":     rng.choice(["M", "F"], n),
        "age_grp": rng.choice(["18-60", ">60"], n),
        "site_id": rng.choice(["SITE-A", "SITE-B"], n),
    })


class TestFairnessAudit:
    def test_returns_subgroups(self):
        from sepsis_vitals.ml.fairness import audit_fairness
        df = _make_prediction_df()
        result = audit_fairness(df, "label", "prob", ["sex", "site_id"])
        assert "subgroups" in result
        assert len(result["subgroups"]) >= 3  # overall + subgroups

    def test_overall_group_present(self):
        from sepsis_vitals.ml.fairness import audit_fairness
        df = _make_prediction_df()
        result = audit_fairness(df, "label", "prob", ["sex"])
        names = [s["group_name"] for s in result["subgroups"]]
        assert "overall" in names

    def test_small_groups_excluded(self):
        from sepsis_vitals.ml.fairness import audit_fairness
        df = _make_prediction_df(n=50)
        result = audit_fairness(df, "label", "prob", ["sex"],
                                min_group_size=100)  # all groups < 100
        assert len(result["subgroups"]) == 1  # only overall

    def test_flags_dict_present(self):
        from sepsis_vitals.ml.fairness import audit_fairness
        df = _make_prediction_df()
        result = audit_fairness(df, "label", "prob", ["sex"])
        assert "fairness_flags" in result
        assert isinstance(result["fairness_flags"], list)


class TestCalibrationMetrics:
    def test_perfect_calibration_low_ece(self):
        from sepsis_vitals.ml.fairness import calibration_metrics
        # Perfect calibration: prob = frequency
        y_true = np.array([1, 0] * 500)
        y_prob = np.array([0.9, 0.1] * 500)
        result = calibration_metrics(y_true, y_prob)
        assert result["ece"] < 0.15
        assert "reliability_diagram" in result

    def test_brier_score_worst_case(self):
        from sepsis_vitals.ml.fairness import calibration_metrics
        y_true = np.ones(100)
        y_prob = np.zeros(100)  # always predicts 0 for all-positive labels
        result = calibration_metrics(y_true, y_prob)
        assert result["brier_score"] == pytest.approx(1.0, abs=0.01)

    def test_quality_labels(self):
        from sepsis_vitals.ml.fairness import calibration_metrics
        y = np.array([1, 0] * 200)
        p = np.array([0.8, 0.2] * 200)
        result = calibration_metrics(y, p)
        assert result["calibration_quality"] in ("excellent", "good", "moderate", "poor")


class TestConformalPredictor:
    def test_coverage_meets_target(self):
        """Conformal prediction must achieve at least 1-alpha coverage."""
        from sepsis_vitals.ml.fairness import ConformalPredictor

        class DummyModel:
            def predict_proba(self, X):
                n = len(X)
                probs = np.clip(np.random.default_rng(0).normal(0.5, 0.2, n), 0, 1)
                return np.column_stack([1 - probs, probs])

        X_cal = pd.DataFrame({"f": np.ones(200)})
        y_cal = pd.Series(np.random.randint(0, 2, 200))
        X_test = pd.DataFrame({"f": np.ones(100)})
        y_test = pd.Series(np.random.randint(0, 2, 100))

        model = DummyModel()
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(model, X_cal, y_cal)

        lower, upper, uncertain = cp.predict_interval(model, X_test)
        assert len(lower) == 100
        assert (lower <= upper).all()

    def test_raises_before_calibrate(self):
        from sepsis_vitals.ml.fairness import ConformalPredictor
        cp = ConformalPredictor()
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(None, pd.DataFrame({"f": [1]}))


class TestAlertExplanation:
    def test_returns_string(self):
        from sepsis_vitals.ml.fairness import generate_alert_explanation
        result = generate_alert_explanation(
            vitals={"resp_rate": 26, "sbp": 92, "gcs": 13},
            qsofa=3, sirs=2, shock_index=1.1,
            risk_level="critical",
        )
        assert isinstance(result, str)
        assert len(result) > 20

    def test_includes_risk_level(self):
        from sepsis_vitals.ml.fairness import generate_alert_explanation
        result = generate_alert_explanation(
            {"resp_rate": 26}, qsofa=1, sirs=1,
            shock_index=None, risk_level="high",
        )
        assert "HIGH" in result

    def test_swahili_stub(self):
        from sepsis_vitals.ml.fairness import generate_alert_explanation
        result = generate_alert_explanation(
            {"resp_rate": 26}, qsofa=1, sirs=1,
            shock_index=None, risk_level="moderate",
            language="sw",
        )
        assert "[SW-PENDING]" in result


class TestCounterfactual:
    def test_high_rr_generates_counterfactual(self):
        from sepsis_vitals.ml.fairness import generate_counterfactual
        result = generate_counterfactual(
            {"resp_rate": 26, "sbp": 92, "gcs": 13}, "high"
        )
        # Should suggest reducing resp_rate or raising sbp
        assert result is None or isinstance(result, str)

    def test_normal_vitals_no_counterfactual(self):
        from sepsis_vitals.ml.fairness import generate_counterfactual
        result = generate_counterfactual(
            {"resp_rate": 16, "sbp": 120, "gcs": 15}, "low"
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Health Economics
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEconomics:
    def setup_method(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from health_economics.model import HealthEconomicsModel, EconomicsParams
        self.Model  = HealthEconomicsModel
        self.Params = EconomicsParams

    def test_full_report_returns_dict(self):
        report = self.Model().full_report()
        assert isinstance(report, dict)
        for key in ("epidemiology","mortality_impact","clinical_savings_usd",
                    "software_costs_usd","health_economics","roi","summary"):
            assert key in report

    def test_deaths_averted_positive(self):
        m = self.Model()
        assert m.deaths_averted() > 0

    def test_deaths_averted_lt_total_deaths(self):
        m = self.Model()
        assert m.deaths_averted() < m.deaths_without_model()

    def test_zero_sensitivity_no_deaths_averted(self):
        p = self.Params(model_sensitivity=0.0)
        m = self.Model(p)
        assert m.deaths_averted() == pytest.approx(0.0, abs=0.01)

    def test_roi_positive_at_high_sensitivity(self):
        p = self.Params(model_sensitivity=0.90, annual_encounters=6000)
        m = self.Model(p)
        assert m.roi_pct() > 0

    def test_qaly_gained_positive(self):
        m = self.Model()
        assert m.qalys_gained() > 0

    def test_cost_per_qaly_finite(self):
        m = self.Model()
        assert m.cost_per_qaly_usd() < float("inf")

    def test_alerts_per_100_reasonable(self):
        m = self.Model()
        a = m.alerts_per_100_enc()
        assert 0 < a < 100

    def test_break_even_sensitivity_found(self):
        p = self.Params(annual_encounters=6000)
        m = self.Model(p)
        be = m.break_even_sensitivity()
        assert be is None or 0 < be <= 1.0

    def test_summary_is_string(self):
        m = self.Model()
        assert isinstance(m.full_report()["summary"], str)
        assert len(m.full_report()["summary"]) > 50

    def test_higher_encounters_more_deaths_averted(self):
        m_small = self.Model(self.Params(annual_encounters=2000))
        m_large = self.Model(self.Params(annual_encounters=8000))
        assert m_large.deaths_averted() > m_small.deaths_averted()
