"""
sepsis_vitals.api — FastAPI application with security layers.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sepsis_vitals import __version__
from sepsis_vitals.scores import compute_scores

app = FastAPI(
    title="Sepsis Vitals API",
    version=__version__,
    description="Vitals-only sepsis prediction for low-resource hospitals",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__, "timestamp": time.time()}


@app.post("/score")
async def score_vitals(request: Request):
    vitals = await request.json()
    result = compute_scores(vitals)
    return result.as_dict()


@app.get("/metrics")
async def metrics():
    return JSONResponse(
        content={"info": "Prometheus metrics endpoint — connect Prometheus scraper"},
        media_type="text/plain",
    )
