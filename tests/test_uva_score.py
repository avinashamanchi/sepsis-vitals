"""
tests/test_uva_score.py — the published Universal Vital Assessment score.

Expected values come from Moore CC et al., BMJ Glob Health 2017;2:e000344
(doi:10.1136/bmjgh-2017-000344): Table 5 (points) and Table 7 (bands).
"""

from __future__ import annotations

import pytest

NORMAL = {"temperature": 37.0, "heart_rate": 80, "resp_rate": 16, "sbp": 120, "spo2": 97, "gcs": 15}


@pytest.mark.parametrize("change, points", [
    ({"temperature": 36.0}, 0), ({"temperature": 35.9}, 2),
    ({"heart_rate": 119}, 0), ({"heart_rate": 120}, 1),
    ({"resp_rate": 29}, 0), ({"resp_rate": 30}, 1),
    ({"sbp": 90}, 0), ({"sbp": 89}, 1),
    ({"spo2": 92}, 0), ({"spo2": 91}, 2),
    ({"gcs": 15}, 0), ({"gcs": 14}, 4),
])
def test_each_published_cut_off(change, points):
    from sepsis_vitals.scores import uva_score

    assert uva_score({**NORMAL, **change}).score == points


def test_hiv_positive_adds_two_and_unknown_scores_zero():
    from sepsis_vitals.scores import uva_score

    assert uva_score(NORMAL, hiv_positive=True).score == 2
    assert uva_score(NORMAL, hiv_positive=False).score == 0
    unknown = uva_score(NORMAL)
    assert unknown.score == 0 and unknown.hiv_status == "no/unknown"


def test_maximum_is_13_and_bands_follow_table_7():
    from sepsis_vitals.scores import uva_score

    worst = {"temperature": 35.0, "heart_rate": 140, "resp_rate": 35, "sbp": 80, "spo2": 85, "gcs": 10}
    assert uva_score(worst, hiv_positive=True).score == 13
    assert [uva_score({**NORMAL, **v}).risk_band for v in (
        {}, {"heart_rate": 125}, {"temperature": 35.5}, {"temperature": 35.5, "resp_rate": 31, "sbp": 85},
        {"gcs": 14}, {"gcs": 14, "heart_rate": 125},
    )] == ["low", "low", "medium", "medium", "medium", "high"]


def test_missing_inputs_are_listed_not_hidden():
    from sepsis_vitals.scores import uva_score

    result = uva_score({"heart_rate": 125, "resp_rate": 32})
    assert result.score == 2 and not result.complete
    assert result.missing_inputs == ["temperature", "sbp", "spo2", "gcs"]


def test_bundle_reports_uva_without_changing_the_rule_alert():
    from sepsis_vitals.scores import compute_scores

    bundle = compute_scores({**NORMAL, "gcs": 14, "hiv_positive": True})
    assert bundle.uva.score == 6 and bundle.uva.risk_band == "high"
    as_dict = bundle.as_dict()
    assert as_dict["uva"] == 6 and as_dict["uva_risk_band"] == "high"
    assert as_dict["uva_style"] == 0  # legacy project score, unchanged


def test_score_endpoint_returns_the_published_uva(monkeypatch):
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api

    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", False)
    api.app.dependency_overrides[api.check_rate_limit] = lambda: None
    try:
        with TestClient(api.app) as client:
            body = client.post("/score", json={"heart_rate": 125, "resp_rate": 31, "sbp": 85,
                                               "hiv_positive": True}).json()
    finally:
        api.app.dependency_overrides.clear()
    assert body["uva"] == 5 and body["uva_risk_band"] == "high"
    assert body["uva_missing_inputs"] == ["temperature", "spo2", "gcs"]
    assert "bmjgh-2017-000344" in body["uva_reference"]
    assert "mortality" in body["uva_outcome"]


def test_hiv_status_is_accepted_only_by_the_score_endpoint(monkeypatch):
    """HIV status must never reach /predict, its audit record, or the copilot."""
    from fastapi.testclient import TestClient

    import sepsis_vitals.api as api
    from sepsis_vitals.schemas import CopilotRequest, PredictRequest, VitalsInput

    assert "hiv_positive" not in VitalsInput.model_fields
    for model in (PredictRequest, CopilotRequest):
        assert "hiv_positive" not in model.model_fields["vitals"].annotation.model_fields
    monkeypatch.setattr("sepsis_vitals.dependencies._auth_enabled", False)
    for dep in (api.check_rate_limit, api.check_ml_rate_limit):
        api.app.dependency_overrides[dep] = lambda: None
    try:
        with TestClient(api.app) as client:
            resp = client.post("/predict", json={"patient_id": "synthetic-hiv", "vitals": {
                "heart_rate": 125, "resp_rate": 31, "sbp": 85, "hiv_positive": True}})
    finally:
        api.app.dependency_overrides.clear()
    assert resp.status_code == 422
