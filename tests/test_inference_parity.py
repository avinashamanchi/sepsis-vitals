"""
tests/test_inference_parity.py — live inference must build exactly the
features the model was trained and evaluated on.

Regression for train/serve skew: the predictor used to score every request
as a first observation (no deltas, rolling std or observation gap), which on
the synthetic audit lowered AUROC 0.90 -> 0.82 and halved predicted risk.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None or not (MODELS / "sepsis_model.joblib").exists(),
    reason="model artifact or scikit-learn not available",
)


@pytest.fixture(scope="module")
def predictor():
    from sepsis_vitals.ml.predictor import SepsisPredictor

    p = SepsisPredictor(str(MODELS))
    p.load()
    return p


SEQUENCE = [
    ("2026-01-01 02:00", dict(temperature=37.0, heart_rate=88, resp_rate=16, sbp=124, dbp=78, spo2=97, gcs=15, map=93, lactate=1.1)),
    ("2026-01-01 06:00", dict(temperature=37.9, heart_rate=104, resp_rate=20, sbp=108, dbp=66, spo2=95, gcs=15, map=80)),
    ("2026-01-01 09:30", dict(temperature=38.4, heart_rate=112, resp_rate=23, sbp=101, dbp=62, spo2=94, gcs=15, map=75, lactate=2.0)),
    ("2026-01-01 13:00", dict(temperature=38.8, heart_rate=121, resp_rate=26, sbp=94, dbp=58, spo2=92, gcs=14, map=70, lactate=2.9)),
]
COMORBIDITIES = dict(has_hypertension=1, has_diabetes=0, has_ckd=0, has_copd=0, has_heart_failure=0)


def _training_features(predictor) -> np.ndarray:
    from sepsis_vitals.ml.trainer import prepare_features

    rows = []
    for ts, vitals in SEQUENCE:
        row = {c: vitals.get(c, np.nan) for c in predictor._INPUT_COLUMNS}
        rows.append({**row, "timestamp": pd.Timestamp(ts), "patient_id": "p", "age_years": 71, **COMORBIDITIES})
    frame = pd.DataFrame(rows)
    for col in ("on_supplemental_o2", "spo2_scale2"):
        frame[col] = frame[col].fillna(False).astype(bool)
    feats, _ = prepare_features(frame)
    return np.array([float(feats.iloc[-1].get(n, np.nan)) for n in predictor.feature_names])


def test_history_features_match_training_pipeline(predictor):
    history = [{"timestamp": ts, **v} for ts, v in SEQUENCE[:-1]]
    ts, current = SEQUENCE[-1]
    live = predictor._build_feature_vector(current, 71, COMORBIDITIES, history, ts)[0]
    np.testing.assert_allclose(live, _training_features(predictor), equal_nan=True)


def test_history_features_are_actually_used(predictor):
    """Deltas and the observation gap must be populated, not NaN."""
    history = [{"timestamp": ts, **v} for ts, v in SEQUENCE[:-1]]
    ts, current = SEQUENCE[-1]
    live = dict(zip(predictor.feature_names,
                    predictor._build_feature_vector(current, 71, COMORBIDITIES, history, ts)[0]))
    assert live["heart_rate_delta"] == pytest.approx(121 - 112)
    assert live["obs_gap_min"] == pytest.approx(210.0)
    assert not np.isnan(live["heart_rate_roll_std"])


def test_future_history_is_ignored(predictor):
    """Observations at or after the current time must not leak into features."""
    ts, current = SEQUENCE[1]
    leaky = [{"timestamp": t, **v} for t, v in SEQUENCE]  # includes later rows
    clean = [{"timestamp": t, **v} for t, v in SEQUENCE[:1]]
    a = predictor._build_feature_vector(current, 71, COMORBIDITIES, leaky, ts)[0]
    b = predictor._build_feature_vector(current, 71, COMORBIDITIES, clean, ts)[0]
    np.testing.assert_allclose(a, b, equal_nan=True)
