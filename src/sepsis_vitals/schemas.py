"""
sepsis_vitals.schemas — request and response models for the HTTP API.

Shared by ``sepsis_vitals.api`` and the endpoint modules in
``sepsis_vitals.routes``. ``sepsis_vitals.api`` re-exports every model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class VitalsInput(BaseModel):
    temperature: Optional[float] = Field(None, ge=25.0, le=45.0, description="Body temperature in °C")
    heart_rate: Optional[float] = Field(None, ge=0, le=350, description="Heart rate in bpm")
    resp_rate: Optional[float] = Field(None, ge=0, le=80, description="Respiratory rate /min")
    sbp: Optional[float] = Field(None, ge=30, le=300, description="Systolic blood pressure mmHg")
    dbp: Optional[float] = Field(None, ge=20, le=200, description="Diastolic blood pressure mmHg")
    spo2: Optional[float] = Field(None, ge=0, le=100, description="Oxygen saturation %")
    gcs: Optional[float] = Field(None, ge=3, le=15, description="Glasgow Coma Scale")
    map: Optional[float] = Field(None, ge=20, le=200, description="Mean arterial pressure mmHg")
    lactate: Optional[float] = Field(None, ge=0, le=30, description="Serum lactate mmol/L")
    wbc: Optional[float] = Field(None, ge=0, le=100, description="White blood cell count x10^9/L")
    procalcitonin: Optional[float] = Field(None, ge=0, le=200, description="Procalcitonin ng/mL")
    on_supplemental_o2: Optional[bool] = Field(
        None, description="Receiving supplemental oxygen (NEWS2 adds 2 points)"
    )
    spo2_scale2: Optional[bool] = Field(
        None, description="Use NEWS2 SpO2 Scale 2 (prescribed 88-92% target only)"
    )


_NEWS2_FLAGS = ("on_supplemental_o2", "spo2_scale2")


def _count_measurements(vitals: Dict[str, Any]) -> int:
    """Number of measured values, excluding NEWS2 context flags."""
    return sum(1 for k in vitals if k not in _NEWS2_FLAGS)


class ComorbidityInput(BaseModel):
    has_hypertension: int = Field(0, ge=0, le=1)
    has_diabetes: int = Field(0, ge=0, le=1)
    has_ckd: int = Field(0, ge=0, le=1)
    has_copd: int = Field(0, ge=0, le=1)
    has_heart_failure: int = Field(0, ge=0, le=1)


class PredictRequest(BaseModel):
    vitals: VitalsInput
    patient_id: str = Field("unknown", max_length=100)
    age_years: Optional[int] = Field(None, ge=0, le=120)
    comorbidities: Optional[ComorbidityInput] = None


class BatchPredictRequest(BaseModel):
    patients: List[PredictRequest] = Field(..., max_length=10)


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class PredictionResponse(BaseModel):
    """Research output of the development model and the rule-based scores.

    ``risk_level`` is an ordinal category, not a probability: the higher of
    ``rule_risk_level`` (from the NEWS2-style/qSOFA/SIRS/shock-index scores)
    and ``model_risk_level`` (from ``risk_probability``). It exists so the
    model can never lower what the rules flag. ``risk_probability`` is the
    model output on synthetic development data; it is not a calibrated
    probability for patients. ``clinical_use`` is always "not-permitted".
    """

    patient_id: str
    timestamp: str
    risk_probability: float = Field(..., description="Development-model output; not calibrated for patients")
    risk_level: str = Field(..., description="Higher of rule_risk_level and model_risk_level (ordinal, not a probability)")
    confidence_interval: ConfidenceInterval
    alert: bool = Field(..., description="Rule alert, or model level high/critical, or model output above its alert cut-off")
    clinical_scores: Dict[str, Any]
    top_risk_factors: List[Dict[str, Any]]
    recommendation: str
    model: Dict[str, str]
    rule_risk_level: str = Field(..., description="Level from the rule-based scores alone")
    model_risk_level: str = Field(..., description="Level from the model output alone")
    provenance: Dict[str, Any]
    research_only: bool = True
    clinical_use: str = "not-permitted"
    validation_status: str = "Synthetic development baseline; no clinical validation"
    intended_use: str = "Retrospective research and prospective silent-mode evaluation"


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    model_loaded: bool
    model_name: Optional[str]
    auth_enabled: bool
    websocket_connections: int


# Known, clinically unreviewed gaps in the NEWS2-style score. Returned with
# every score so the output is never read as a complete NEWS2 assessment.
# Resolving them needs an approved clinical specification (PROJECT_REVIEW.md C7).
NEWS2_LIMITATIONS = [
    "Consciousness is approximated from GCS (<15 scores 3); ACVPU and new confusion are not assessed.",
    "The NEWS2 single-parameter red score (any parameter scoring 3) is not evaluated "
    "and does not raise the risk level.",
    "Supplemental oxygen and SpO2 Scale 2 are scored only when the caller supplies them.",
]


class ScoreResponse(BaseModel):
    qsofa: int
    sirs_count: int
    news2_style: int
    shock_index: Optional[float]
    uva: int
    risk_level: str
    alert_flag: bool
    explanations: List[str]
    news2_limitations: List[str] = NEWS2_LIMITATIONS


class CopilotRequest(BaseModel):
    vitals: VitalsInput
    patient_id: str = Field("unknown", max_length=100)
    age_years: Optional[int] = Field(None, ge=0, le=120)
    comorbidities: Optional[ComorbidityInput] = None
    question: Optional[str] = Field(None, max_length=500, description="Optional clinical question")


class CopilotResponse(BaseModel):
    analysis: str
    risk_level: str
    key_concerns: List[str]
    suggested_actions: List[str]
    disclaimer: str


# Monitor / simulator request models
class MonitorRegisterRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=100, description="Patient identifier")
    demographics: Optional[Dict[str, Any]] = Field(None, description="Patient demographics")
    comorbidities: Optional[Dict[str, Any]] = Field(None, description="Patient comorbidities")


class SimulatorWardRequest(BaseModel):
    n_patients: int = Field(8, ge=1, le=50, description="Number of patients")
    speed: int = Field(360, ge=1, le=3600, description="Simulation speed multiplier")
    sepsis_count: int = Field(2, ge=0, le=50, description="Number of sepsis patients")
    seed: int = Field(42, ge=0, description="Random seed")


class SimulatorReplayRequest(BaseModel):
    subject_id: Optional[str] = Field(None, max_length=100, description="MIMIC subject ID or 'random'")
    speed: int = Field(720, ge=1, le=3600, description="Replay speed multiplier")
    sepsis_only: bool = Field(False, description="Only select sepsis cases")
