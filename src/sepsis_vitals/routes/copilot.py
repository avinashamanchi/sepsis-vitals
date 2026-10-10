"""
sepsis_vitals.routes.copilot

Endpoints moved out of sepsis_vitals.api (behaviour unchanged). They are
declared on this module's ``router``, which ``sepsis_vitals.api`` includes;
shared state is read from the api module per request, so patching
``sepsis_vitals.api`` in tests still works.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sepsis_vitals.dependencies import _copilot_limiter, check_rate_limit, verify_auth
from sepsis_vitals.schemas import CopilotRequest, CopilotResponse
from sepsis_vitals.scores import compute_scores
from sepsis_vitals.security import sanitise_string

router = APIRouter()
logger = logging.getLogger("sepsis_vitals.api")


def _core():
    """The application module, for runtime state (model, monitor, metrics).

    Looked up per request, never at import: this module does not import
    ``sepsis_vitals.api``, so it can be imported first, alone or in any order.
    """
    from sepsis_vitals import api

    return api

# ---------------------------------------------------------------------------
# AI Clinical Copilot (Anthropic-powered)
# ---------------------------------------------------------------------------

# The copilot is frozen by default until clinical validation and a human-factors
# review establish that it adds value without unsafe automation bias.
_copilot_enabled = os.getenv("SEPSIS_ENABLE_COPILOT", "false").lower() == "true"
# Enterprise LLM feature gate — separate opt-in, requires signed BAA.
_enterprise_llm_enabled = os.getenv("SEPSIS_ENTERPRISE_LLM", "false").lower() == "true"


def _deidentify_vitals(vitals: dict) -> dict:
    """Strip any patient-identifying information before sending to external LLM.

    Only numeric clinical measurements are sent. No names, MRNs, DOBs, or
    free-text fields cross the boundary.
    """
    safe_keys = {
        "temperature", "heart_rate", "resp_rate", "sbp", "dbp", "spo2",
        "gcs", "map", "lactate", "wbc", "procalcitonin",
    }
    return {k: v for k, v in vitals.items() if k in safe_keys}


@router.post("/copilot", response_model=CopilotResponse, dependencies=[Depends(check_rate_limit)])
async def clinical_copilot(body: CopilotRequest, user: Dict = Depends(verify_auth)):
    """Research-only observation summary.

    Disabled by default. Enabling it requires an explicit feature flag; enabling
    external LLM processing additionally requires a signed BAA and separate flag.
    """
    if not _copilot_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "Copilot is frozen for this investigational release pending "
                "clinical validation and human-factors review."
            ),
        )

    copilot_key = f"copilot:{user.get('user', user.get('email', 'anon'))}"
    if not _copilot_limiter.allow(copilot_key):
        raise HTTPException(status_code=429, detail="Copilot rate limit exceeded. Max 1 request per 2 seconds.")

    _core()._metrics["copilot_calls_total"] += 1

    vitals_dict = {k: v for k, v in body.vitals.model_dump().items() if v is not None}
    scores = compute_scores(vitals_dict)
    scores_dict = scores.as_dict()

    # Get ML prediction if model loaded
    ml_risk = None
    predictor = await asyncio.to_thread(_core()._get_predictor)
    if predictor:
        comorbidities = body.comorbidities.model_dump() if body.comorbidities else None
        pred = await asyncio.to_thread(
            predictor.predict,
            vitals=vitals_dict,
            patient_id=body.patient_id,
            age_years=body.age_years,
            comorbidities=comorbidities,
        )
        ml_risk = pred.to_dict()

    # LLM copilot: ONLY available under enterprise flag with BAA
    if _enterprise_llm_enabled:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                # Sanitise and check for prompt injection before LLM call
                safe_question = None
                if body.question:
                    from sepsis_vitals.security import PromptInjectionError, check_prompt_injection
                    try:
                        check_prompt_injection(body.question)
                    except PromptInjectionError:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid input detected in clinical question.",
                        )
                    safe_question = sanitise_string(body.question, max_length=500)
                safe_vitals = _deidentify_vitals(vitals_dict)
                analysis = await _anthropic_copilot(
                    safe_vitals, scores_dict, ml_risk, body.age_years, safe_question
                )
                return analysis
            except Exception:
                logger.warning("LLM copilot failed, falling back to rule-based", exc_info=True)

    # Default: deterministic rule-based analysis (legally safe, no hallucination risk)
    return _rule_based_copilot(vitals_dict, scores_dict, ml_risk, body.age_years)


async def _anthropic_copilot(
    vitals: dict, scores: dict, ml_risk: Optional[dict],
    age: Optional[int], question: Optional[str],
) -> CopilotResponse:
    """Call Anthropic Claude for clinical analysis."""
    import anthropic

    client = anthropic.Anthropic()

    risk_info = ""
    if ml_risk:
        risk_info = f"""
ML Model Prediction:
- Risk probability: {ml_risk['risk_probability']:.1%}
- Risk level: {ml_risk['risk_level']}
- Top risk factors: {json.dumps(ml_risk.get('top_risk_factors', [])[:3])}
"""

    prompt = f"""You summarize observations for an investigational sepsis-model validation study. Do not diagnose, prescribe, recommend treatment, or claim clinical benefit. Identify only the supplied score criteria, unusual measurements, missing data, and questions for a designated study reviewer.

Patient vitals: {json.dumps(vitals)}
Age: {age if age else 'Unknown'}
Clinical scores: qSOFA={scores.get('qsofa',0)}/3, SIRS={scores.get('sirs_count',0)}/3, NEWS2={scores.get('news2_style',0)}, Shock Index={scores.get('shock_index','N/A')}
Risk level: {scores.get('risk_level', 'unknown')}
{risk_info}
{f'Clinical question: {question}' if question else ''}

Respond in this exact JSON format:
{{
  "analysis": "2-3 sentence research observation summary",
  "risk_level": "low|moderate|high|critical",
  "key_concerns": ["concern1", "concern2"],
  "suggested_actions": ["data verification step", "study review step"]
}}

Be concise and precise. Suggested actions must be limited to data verification,
documentation, or review under the study protocol."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = getattr(message.content[0], "text", "")
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Enterprise LLM returned no text response")
    response_text = response_text.strip()
    # Extract JSON from response
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    parsed = json.loads(response_text)

    return CopilotResponse(
        analysis=parsed.get("analysis", "Analysis unavailable."),
        risk_level=parsed.get("risk_level", scores.get("risk_level", "unknown")),
        key_concerns=parsed.get("key_concerns", []),
        suggested_actions=parsed.get("suggested_actions", []),
        disclaimer=(
            "Investigational research summary. Not for diagnosis or treatment; "
            "review only under the approved study protocol."
        ),
    )


def _rule_based_copilot(
    vitals: dict, scores: dict, ml_risk: Optional[dict], age: Optional[int],
) -> CopilotResponse:
    """Produce a non-treatment research summary when external LLM use is off."""
    concerns: List[str] = []
    risk_level = scores.get("risk_level", "low")

    temp = vitals.get("temperature")
    if temp and (temp > 38.3 or temp < 36.0):
        concerns.append(f"Temperature ({temp}°C) meets an encoded score criterion.")

    hr = vitals.get("heart_rate")
    if hr and hr > 100:
        concerns.append(f"Heart rate ({hr} bpm) is above the encoded reference range.")
    elif hr and hr < 50:
        concerns.append(f"Heart rate ({hr} bpm) is below the encoded reference range.")

    rr = vitals.get("resp_rate")
    if rr and rr > 22:
        concerns.append(f"Respiratory rate ({rr}/min) meets the qSOFA criterion.")

    sbp = vitals.get("sbp")
    if sbp and sbp <= 100:
        concerns.append(f"Systolic blood pressure ({sbp} mmHg) meets the qSOFA criterion.")

    spo2 = vitals.get("spo2")
    if spo2 and spo2 < 94:
        concerns.append(f"SpO2 ({spo2}%) is below the encoded reference range.")

    gcs = vitals.get("gcs")
    if gcs and gcs < 15:
        concerns.append(f"GCS ({gcs}/15) meets the qSOFA criterion.")

    lactate = vitals.get("lactate")
    if lactate is not None and lactate >= 2.0:
        concerns.append(f"Lactate ({lactate} mmol/L) meets an encoded risk criterion.")

    qsofa = scores.get("qsofa", 0)
    sirs = scores.get("sirs_count", 0)
    ml_prob = ml_risk["risk_probability"] if ml_risk else None
    if ml_prob is not None:
        concerns.append(
            f"The unvalidated development model produced a {ml_prob:.0%} output."
        )

    if not concerns:
        concerns.append("No encoded score criteria fired in the supplied observations.")

    analysis = (
        f"Research summary: qSOFA {qsofa}/3 and SIRS {sirs}/3. "
        f"The encoded risk category is {risk_level}; this is not a diagnosis."
    )

    return CopilotResponse(
        analysis=analysis,
        risk_level=risk_level,
        key_concerns=concerns[:5],
        suggested_actions=[
            "Verify observation values, timestamps, units, and data source.",
            "Record reviewer feedback under the approved validation protocol.",
        ],
        disclaimer=(
            "Investigational research summary. Not for diagnosis or treatment."
        ),
    )


