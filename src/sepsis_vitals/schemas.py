"""
sepsis_vitals.schemas — request and response models for the HTTP API.

Shared by ``sepsis_vitals.api`` and the endpoint modules in
``sepsis_vitals.routes``. ``sepsis_vitals.api`` re-exports every model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class VitalsInput(BaseModel):
    # Unknown fields are an error, not silently dropped: a misspelt or
    # unsupported measurement must never be scored as "missing".
    model_config = ConfigDict(extra="forbid")

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
    hiv_positive: Optional[bool] = Field(
        None,
        description=(
            "Known HIV-positive serostatus, used only by the published UVA score "
            "(true adds 2 points; false or omitted scores as 'no/unknown'). Sensitive: "
            "send only where local policy allows."
        ),
    )


_NEWS2_FLAGS = ("on_supplemental_o2", "spo2_scale2", "hiv_positive")  # context, not measurements


def _count_measurements(vitals: Dict[str, Any]) -> int:
    """Number of measured values, excluding NEWS2 context flags."""
    return sum(1 for k in vitals if k not in _NEWS2_FLAGS)


class ComorbidityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_hypertension: int = Field(0, ge=0, le=1)
    has_diabetes: int = Field(0, ge=0, le=1)
    has_ckd: int = Field(0, ge=0, le=1)
    has_copd: int = Field(0, ge=0, le=1)
    has_heart_failure: int = Field(0, ge=0, le=1)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vitals: VitalsInput
    patient_id: str = Field("unknown", max_length=100)
    age_years: Optional[int] = Field(None, ge=0, le=120)
    comorbidities: Optional[ComorbidityInput] = None


class BatchPredictRequest(BaseModel):
    patients: List[PredictRequest] = Field(..., max_length=10)


# Units accepted by the time-series contract. Values must already be in the
# canonical unit; the declaration is checked, never converted (FHIR ingest
# converts UCUM units). A caller sending Fahrenheit or kPa gets a 422.
CANONICAL_UNITS: Dict[str, tuple] = {
    "temperature": ("Cel", {"Cel", "°C", "C", "degC"}),
    "heart_rate": ("/min", {"/min", "bpm", "beats/min", "{beats}/min"}),
    "resp_rate": ("/min", {"/min", "breaths/min", "{breaths}/min"}),
    "sbp": ("mm[Hg]", {"mm[Hg]", "mmHg"}),
    "dbp": ("mm[Hg]", {"mm[Hg]", "mmHg"}),
    "map": ("mm[Hg]", {"mm[Hg]", "mmHg"}),
    "spo2": ("%", {"%"}),
    "gcs": ("{score}", {"{score}", "score"}),
    "lactate": ("mmol/L", {"mmol/L"}),
    "wbc": ("10*9/L", {"10*9/L", "10^9/L", "x10^9/L", "×10⁹/L"}),
    "procalcitonin": ("ng/mL", {"ng/mL"}),
}

# Engineering limits, not clinical ones: clock skew tolerated for the newest
# observation, and payload size. Only the last few observations affect the
# model's features (SepsisPredictor._HISTORY_ROWS); older ones are recorded
# with the request but do not change the score.
MAX_CLOCK_SKEW = timedelta(minutes=5)
MAX_OBSERVATIONS = 48


class TimedObservation(VitalsInput):
    timestamp: datetime = Field(..., description="Observation time with a UTC offset (ISO 8601)")
    patient_id: Optional[str] = Field(
        None, max_length=100, description="Optional; must equal the request's patient_id"
    )


class TimeseriesPredictRequest(BaseModel):
    """One patient's observations, oldest first; the last one is scored."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(..., min_length=1, max_length=100)
    age_years: Optional[int] = Field(None, ge=0, le=120)
    comorbidities: Optional[ComorbidityInput] = None
    units: Optional[Dict[str, str]] = Field(
        None, description="Optional unit declaration per field; must match the canonical units"
    )
    observations: List[TimedObservation] = Field(..., min_length=1, max_length=MAX_OBSERVATIONS)

    @field_validator("units")
    @classmethod
    def _units_are_canonical(cls, units: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        for name, unit in (units or {}).items():
            if name not in CANONICAL_UNITS:
                raise ValueError(f"units: unknown field {name!r}")
            canonical, accepted = CANONICAL_UNITS[name]
            if unit not in accepted:
                raise ValueError(
                    f"units: {name} in {unit!r} is not accepted; send values in {canonical}"
                )
        return units

    @model_validator(mode="after")
    def _consistent_series(self) -> "TimeseriesPredictRequest":
        previous: Optional[datetime] = None
        for i, obs in enumerate(self.observations):
            if obs.timestamp.tzinfo is None or obs.timestamp.utcoffset() is None:
                raise ValueError(f"observations[{i}].timestamp must include a UTC offset")
            if obs.patient_id is not None and obs.patient_id != self.patient_id:
                raise ValueError(f"observations[{i}].patient_id does not match patient_id")
            if previous is not None:
                if obs.timestamp == previous:
                    raise ValueError(f"observations[{i}] duplicates the timestamp of the one before")
                if obs.timestamp < previous:
                    raise ValueError("observations must be in ascending time order")
            previous = obs.timestamp
        newest = self.observations[-1]
        if newest.timestamp > datetime.now(timezone.utc) + MAX_CLOCK_SKEW:
            raise ValueError("the newest observation is in the future")
        measured = {k: v for k, v in newest.model_dump().items()
                    if v is not None and k not in ("timestamp", "patient_id")}
        if _count_measurements(measured) < 3:
            raise ValueError("the newest observation needs at least 3 measurements")
        return self


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
    observed_at: Optional[str] = Field(None, description="Time of the scored observation (UTC)")
    risk_probability: float = Field(..., description="Development-model output; not calibrated for patients")
    risk_level: str = Field(..., description="Higher of rule_risk_level and model_risk_level (ordinal, not a probability)")
    confidence_interval: Optional[ConfidenceInterval] = Field(
        None, description="Always null unless the artifact supports a validated interval; see uncertainty"
    )
    uncertainty: Dict[str, Any] = Field(default_factory=dict)
    alert: bool = Field(..., description="Rule alert, or model level high/critical, or model output above its alert cut-off")
    clinical_scores: Dict[str, Any]
    top_risk_factors: List[Dict[str, Any]]
    recommendation: str
    model: Dict[str, str]
    rule_risk_level: str = Field(..., description="Level from the rule-based scores alone")
    model_risk_level: str = Field(..., description="Level from the model output alone")
    provenance: Dict[str, Any]
    explanation: Dict[str, Any] = Field(default_factory=dict, description="Method behind top_risk_factors")
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    threshold_policy: Dict[str, Any] = Field(default_factory=dict, description="Cut-offs in force (placeholders)")
    versions: Dict[str, Any] = Field(default_factory=dict)
    decision_support_notice: str = ""
    request_id: str = ""
    audit_record_id: str = Field("", description="The append-only record written before this response")
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
    uva: int = Field(..., description="Published UVA score (0-13); see uva_reference")
    uva_risk_band: str = Field(..., description="low 0-1, medium 2-4, high >4 (published bands)")
    uva_missing_inputs: List[str] = Field(default_factory=list,
                                          description="UVA items not measured; they added no points")
    uva_outcome: str = "in-hospital mortality, adult inpatients, sub-Saharan Africa (derivation cohort)"
    uva_reference: str = ""
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
