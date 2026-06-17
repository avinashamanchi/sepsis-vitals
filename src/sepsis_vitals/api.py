"""
sepsis_vitals.api — FastAPI application with ML prediction and security layers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sepsis_vitals import __version__
from sepsis_vitals.scores import compute_scores

app = FastAPI(
    title="Sepsis Vitals API",
    version=__version__,
    description="AI-powered vitals-only sepsis prediction for low-resource hospitals",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Lazy-loaded predictor singleton
_predictor = None


def _get_predictor():
    """Lazy-load the sepsis predictor."""
    global _predictor
    if _predictor is None:
        from sepsis_vitals.ml.predictor import SepsisPredictor
        _predictor = SepsisPredictor()
        try:
            _predictor.load()
        except FileNotFoundError:
            _predictor = None
            return None
    return _predictor


@app.get("/health")
async def health():
    predictor = _get_predictor()
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": time.time(),
        "model_loaded": predictor is not None,
        "model_name": predictor.metadata["model_name"] if predictor and predictor.metadata else None,
    }


@app.post("/score")
async def score_vitals(request: Request):
    vitals = await request.json()
    result = compute_scores(vitals)
    return result.as_dict()


@app.post("/predict")
async def predict_sepsis(request: Request):
    """Generate ML-powered sepsis risk prediction.

    Request body:
        {
            "vitals": {"temperature": 38.5, "heart_rate": 110, ...},
            "patient_id": "PT-001",
            "age_years": 65,
            "comorbidities": {"has_hypertension": 1, "has_diabetes": 0, ...}
        }

    Returns full prediction with risk probability, explanations, and recommendations.
    """
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python -m sepsis_vitals.train' first.",
        )

    data = await request.json()
    vitals = data.get("vitals", data)
    patient_id = data.get("patient_id", "unknown")
    age_years = data.get("age_years")
    comorbidities = data.get("comorbidities")

    prediction = predictor.predict(
        vitals=vitals,
        patient_id=patient_id,
        age_years=age_years,
        comorbidities=comorbidities,
    )
    return prediction.to_dict()


@app.post("/predict/batch")
async def predict_batch(request: Request):
    """Batch prediction for multiple patients."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    data = await request.json()
    patients = data.get("patients", [])

    results = []
    for patient in patients:
        prediction = predictor.predict(
            vitals=patient.get("vitals", {}),
            patient_id=patient.get("patient_id", "unknown"),
            age_years=patient.get("age_years"),
            comorbidities=patient.get("comorbidities"),
        )
        results.append(prediction.to_dict())

    return {"predictions": results, "count": len(results)}


@app.get("/patient/{patient_id}/trend")
async def patient_trend(patient_id: str):
    """Get risk trend for a monitored patient."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    trend = predictor.get_patient_trend(patient_id)
    if trend is None:
        raise HTTPException(status_code=404, detail=f"No data for patient {patient_id}")

    return trend


@app.get("/model/info")
async def model_info():
    """Return model metadata and performance metrics."""
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "model_name": predictor.metadata["model_name"],
        "version": predictor.metadata["version"],
        "is_calibrated": predictor.metadata.get("is_calibrated", False),
        "feature_count": len(predictor.feature_names),
        "metrics": predictor.metadata.get("metrics", {}),
        "feature_importance": dict(list(
            predictor.metadata.get("feature_importance", {}).items()
        )[:15]),
    }


@app.get("/metrics")
async def metrics():
    return JSONResponse(
        content={"info": "Prometheus metrics endpoint — connect Prometheus scraper"},
        media_type="text/plain",
    )
