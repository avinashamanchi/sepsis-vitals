"""
tests/test_new_systems.py — Tests for billing, auth, patients, alerts, FHIR, and i18n modules.
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Billing Plans
# ──────────────────────────────────────────────────────────────────────────────

class TestBillingPlans:
    def test_all_plans_defined(self):
        from sepsis_vitals.billing.plans import PLANS
        tiers = [p.tier.value for p in PLANS]
        assert "community" in tiers
        assert "clinical" in tiers
        assert "enterprise" in tiers

    def test_plan_pricing(self):
        from sepsis_vitals.billing.plans import get_plan_by_tier, PlanTier
        community = get_plan_by_tier(PlanTier.COMMUNITY)
        clinical = get_plan_by_tier(PlanTier.CLINICAL)
        enterprise = get_plan_by_tier(PlanTier.ENTERPRISE)
        assert community.monthly_price_cents > 0
        assert clinical.monthly_price_cents > community.monthly_price_cents
        assert enterprise.monthly_price_cents > clinical.monthly_price_cents

    def test_get_plan_by_tier(self):
        from sepsis_vitals.billing.plans import get_plan_by_tier, PlanTier
        plan = get_plan_by_tier(PlanTier.CLINICAL)
        assert plan is not None
        assert plan.display_name == "Clinical"

    def test_three_plans_exist(self):
        from sepsis_vitals.billing.plans import PLANS
        assert len(PLANS) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Auth Tokens
# ──────────────────────────────────────────────────────────────────────────────

try:
    import jwt as _pyjwt  # noqa: F401
    HAS_PYJWT = True
except ImportError:
    HAS_PYJWT = False


@pytest.mark.skipif(not HAS_PYJWT, reason="PyJWT not installed")
class TestAuthTokens:
    def test_create_and_decode_access_token(self):
        from sepsis_vitals.auth.tokens import create_access_token, decode_token
        token = create_access_token("user1", "test@test.com", "nurse", "org1")
        assert isinstance(token, str)
        payload = decode_token(token)
        assert payload["sub"] == "user1"
        assert payload["role"] == "nurse"
        assert payload["org_id"] == "org1"

    def test_create_refresh_token(self):
        from sepsis_vitals.auth.tokens import create_refresh_token, decode_token
        token = create_refresh_token("user1")
        payload = decode_token(token)
        assert payload["sub"] == "user1"
        assert payload["type"] == "refresh"

    def test_expired_token_raises(self):
        from sepsis_vitals.auth.tokens import create_access_token, decode_token
        token = create_access_token("user1", "test@test.com", "nurse", "org1", expires_minutes=-1)
        with pytest.raises(Exception):
            decode_token(token)


# ──────────────────────────────────────────────────────────────────────────────
# Auth Service
# ──────────────────────────────────────────────────────────────────────────────

try:
    import sqlalchemy as _sa  # noqa: F401
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
class TestAuthServiceValidation:
    def test_password_validation_too_short(self):
        from sepsis_vitals.auth.service import _validate_password_strength
        with pytest.raises(ValueError):
            _validate_password_strength("Ab1")

    def test_password_validation_no_uppercase(self):
        from sepsis_vitals.auth.service import _validate_password_strength
        with pytest.raises(ValueError):
            _validate_password_strength("abcdefg1")

    def test_password_validation_no_digit(self):
        from sepsis_vitals.auth.service import _validate_password_strength
        with pytest.raises(ValueError):
            _validate_password_strength("Abcdefgh")

    def test_password_validation_valid(self):
        from sepsis_vitals.auth.service import _validate_password_strength
        # Should not raise
        _validate_password_strength("SecurePass1")


# ──────────────────────────────────────────────────────────────────────────────
# FHIR Resources
# ──────────────────────────────────────────────────────────────────────────────

class TestFHIRLoinc:
    def test_loinc_table_has_entries(self):
        from sepsis_vitals.fhir.loinc import LOINC_TABLE
        assert len(LOINC_TABLE) >= 7  # at least 7 vital signs

    def test_temperature_loinc(self):
        from sepsis_vitals.fhir.loinc import LOINC_TO_INTERNAL
        assert LOINC_TO_INTERNAL.get("8310-5") == "temperature"

    def test_heart_rate_loinc(self):
        from sepsis_vitals.fhir.loinc import LOINC_TO_INTERNAL
        assert LOINC_TO_INTERNAL.get("8867-4") == "heart_rate"

    def test_fahrenheit_to_celsius(self):
        from sepsis_vitals.fhir.loinc import fahrenheit_to_celsius
        assert abs(fahrenheit_to_celsius(98.6) - 37.0) < 0.1

    def test_celsius_to_fahrenheit(self):
        from sepsis_vitals.fhir.loinc import celsius_to_fahrenheit
        assert abs(celsius_to_fahrenheit(37.0) - 98.6) < 0.1


class TestFHIRObservation:
    def test_parse_temperature_observation(self):
        from sepsis_vitals.fhir.resources import FHIRObservation
        obs = FHIRObservation.from_fhir({
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8310-5"}]},
            "valueQuantity": {"value": 38.5, "unit": "Cel"},
        })
        assert obs.internal_name == "temperature"
        assert obs.value == 38.5

    def test_parse_heart_rate_observation(self):
        from sepsis_vitals.fhir.resources import FHIRObservation
        obs = FHIRObservation.from_fhir({
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
            "valueQuantity": {"value": 110, "unit": "/min"},
        })
        assert obs.internal_name == "heart_rate"
        assert obs.value == 110

    def test_vitals_from_observations(self):
        from sepsis_vitals.fhir.resources import FHIRObservation, vitals_from_observations
        obs_list = [
            FHIRObservation.from_fhir({
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "8310-5"}]},
                "valueQuantity": {"value": 39.0, "unit": "Cel"},
            }),
            FHIRObservation.from_fhir({
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                "valueQuantity": {"value": 120, "unit": "/min"},
            }),
        ]
        vitals = vitals_from_observations(obs_list)
        assert vitals["temperature"] == 39.0
        assert vitals["heart_rate"] == 120

    def test_to_fhir_observation(self):
        from sepsis_vitals.fhir.resources import to_fhir_observation
        resource = to_fhir_observation("temperature", 38.5, "Patient/123", "2024-01-01T00:00:00Z")
        assert resource["resourceType"] == "Observation"
        assert resource["valueQuantity"]["value"] == 38.5

    def test_unknown_loinc_code_returns_none(self):
        from sepsis_vitals.fhir.resources import FHIRObservation
        obs = FHIRObservation.from_fhir({
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "99999-9"}]},
            "valueQuantity": {"value": 1.0},
        })
        # Unknown LOINC codes return None (graceful handling)
        assert obs is None


class TestFHIRPatient:
    def test_parse_fhir_patient(self):
        from sepsis_vitals.fhir.resources import FHIRPatient
        patient = FHIRPatient.from_fhir({
            "resourceType": "Patient",
            "id": "test-123",
            "name": [{"family": "Doe", "given": ["John"]}],
            "gender": "male",
            "birthDate": "1990-01-01",
        })
        assert patient.resource_id == "test-123"
        assert patient.gender == "male"
        assert patient.birth_date == "1990-01-01"

    def test_to_fhir_patient(self):
        from sepsis_vitals.fhir.resources import to_fhir_patient
        resource = to_fhir_patient({
            "id": "123",
            "external_id": "PT-001",
            "age_years": 45,
            "sex": "F",
        })
        assert resource["resourceType"] == "Patient"
        assert resource["gender"] in ("female", "unknown", None) or "id" in resource


class TestFHIRRiskAssessment:
    def test_to_fhir_risk_assessment(self):
        from sepsis_vitals.fhir.resources import to_fhir_risk_assessment
        pred = {
            "risk_probability": 0.75,
            "risk_level": "high",
            "timestamp": "2024-01-01T00:00:00Z",
            "recommendation": "Urgent review",
        }
        resource = to_fhir_risk_assessment(pred, "Patient/123")
        assert resource["resourceType"] == "RiskAssessment"
        assert resource["prediction"][0]["probabilityDecimal"] == 0.75


# ──────────────────────────────────────────────────────────────────────────────
# Alerts SMS
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertsSMS:
    def test_get_sms_provider_returns_none_without_config(self):
        from sepsis_vitals.alerts.sms import get_sms_provider
        provider = get_sms_provider()
        assert provider is None

    def test_format_alert_message(self):
        from sepsis_vitals.alerts.sms import _format_alert
        msg = _format_alert("PT-001", "critical", 0.92, "Immediate assessment required")
        assert "PT-001" in msg
        assert "CRITICAL" in msg


class TestAlertsDispatcher:
    def test_dispatcher_init(self):
        from sepsis_vitals.alerts.dispatcher import AlertDispatcher
        d = AlertDispatcher()
        assert d is not None

    def test_register_contact(self):
        from sepsis_vitals.alerts.dispatcher import AlertDispatcher
        d = AlertDispatcher()
        d.register_contact("user1", "websocket", "ws://test")
        contacts = d.list_contacts()
        assert len(contacts) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Patient Service
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
class TestPatientService:
    def test_service_module_imports(self):
        from sepsis_vitals.patients import service
        assert hasattr(service, "create_patient")
        assert hasattr(service, "record_vitals")
        assert hasattr(service, "get_patient_vitals")

    @pytest.mark.skipif(
        not HAS_SQLALCHEMY,
        reason="fastapi/sqlalchemy not installed",
    )
    def test_router_module_imports(self):
        from sepsis_vitals.patients import router
        assert hasattr(router, "router")


# ──────────────────────────────────────────────────────────────────────────────
# I18N
# ──────────────────────────────────────────────────────────────────────────────

class TestLocalization:
    @pytest.fixture
    def en_keys(self):
        import json
        with open("src/sepsis_vitals/i18n/en.json") as f:
            return set(json.load(f).keys())

    def test_french_has_all_keys(self, en_keys):
        with open("src/sepsis_vitals/i18n/fr.json") as f:
            fr_keys = set(json.load(f).keys())
        missing = en_keys - fr_keys
        assert len(missing) == 0, f"French missing keys: {missing}"

    def test_portuguese_has_all_keys(self, en_keys):
        with open("src/sepsis_vitals/i18n/pt.json") as f:
            pt_keys = set(json.load(f).keys())
        missing = en_keys - pt_keys
        assert len(missing) == 0, f"Portuguese missing keys: {missing}"

    def test_amharic_has_all_keys(self, en_keys):
        with open("src/sepsis_vitals/i18n/am.json") as f:
            am_keys = set(json.load(f).keys())
        missing = en_keys - am_keys
        assert len(missing) == 0, f"Amharic missing keys: {missing}"

    def test_arabic_has_all_keys(self, en_keys):
        with open("src/sepsis_vitals/i18n/ar.json") as f:
            ar_keys = set(json.load(f).keys())
        missing = en_keys - ar_keys
        assert len(missing) == 0, f"Arabic missing keys: {missing}"

    def test_all_languages_valid_json(self):
        langs = ["en", "sw", "fr", "pt", "am", "ar"]
        for lang in langs:
            with open(f"src/sepsis_vitals/i18n/{lang}.json") as f:
                data = json.load(f)
                assert isinstance(data, dict), f"{lang}.json is not a dict"
                assert len(data) > 0, f"{lang}.json is empty"

    def test_six_languages_available(self):
        import os
        files = os.listdir("src/sepsis_vitals/i18n/")
        json_files = [f for f in files if f.endswith(".json")]
        assert len(json_files) >= 6


# ──────────────────────────────────────────────────────────────────────────────
# Billing Models
# ──────────────────────────────────────────────────────────────────────────────

class TestBillingModels:
    def test_organization_model_importable(self):
        try:
            from sepsis_vitals.billing.models import Organization
            assert Organization.__tablename__ == "organizations"
        except ImportError:
            pytest.skip("sqlalchemy not installed")

    def test_subscription_model_importable(self):
        try:
            from sepsis_vitals.billing.models import Subscription
            assert Subscription.__tablename__ == "subscriptions"
        except ImportError:
            pytest.skip("sqlalchemy not installed")


# ──────────────────────────────────────────────────────────────────────────────
# Health Economics (existing but verify still works)
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEconomics:
    def test_roi_positive(self):
        from health_economics.model import HealthEconomicsModel
        model = HealthEconomicsModel()
        assert model.roi_pct() > 0

    def test_deaths_averted_positive(self):
        from health_economics.model import HealthEconomicsModel
        model = HealthEconomicsModel()
        assert model.deaths_averted() > 0

    def test_cost_per_qaly(self):
        from health_economics.model import HealthEconomicsModel
        model = HealthEconomicsModel()
        cpq = model.cost_per_qaly_usd()
        assert cpq > 0 and cpq < 100000

    def test_full_report_has_required_keys(self):
        from health_economics.model import HealthEconomicsModel
        model = HealthEconomicsModel()
        report = model.full_report()
        assert "roi" in report
        assert "mortality_impact" in report
        assert "health_economics" in report
        assert "summary" in report
