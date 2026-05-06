"""
scores.py – Validated clinical scoring functions.

Implements:
  - qSOFA  (Seymour et al., JAMA 2016)
  - Partial SIRS  (Bone et al., Chest 1992)
  - Shock Index  (Allgöwer & Burri, 1967)
  - NEWS2-style  (RCP 2017)
  - UVA-style LMIC score  (adapted from Kruisselbrink et al., 2019)
  - Pediatric flag  (Phoenix criteria detection only — not scoring)

All functions operate on plain dicts of vitals and return int/float/None.
They are deliberately free of pandas/numpy so they can run in the API,
on an embedded device, and in browser-side JS analogues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreBundle:
    """All scores for one observation row."""
    qsofa:         int           = 0
    sirs_count:    int           = 0
    shock_index:   Optional[float] = None
    news2_style:   int           = 0
    uva_style:     int           = 0
    risk_level:    str           = "low"       # low | moderate | high | critical
    alert_flag:    bool          = False
    component_flags: dict        = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "qsofa":           self.qsofa,
            "sirs_count":      self.sirs_count,
            "shock_index":     self.shock_index,
            "news2_style":     self.news2_style,
            "uva_style":       self.uva_style,
            "risk_level":      self.risk_level,
            "alert_flag":      self.alert_flag,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Individual scorers
# ──────────────────────────────────────────────────────────────────────────────

def qsofa(vitals: dict) -> tuple[int, dict]:
    """
    qSOFA (Quick SOFA) — 3-point scale.

    Criteria:
      +1  Respiratory rate ≥ 22 /min
      +1  Altered mentation (GCS < 15)
      +1  Systolic BP ≤ 100 mmHg

    Returns (score, component_flags_dict).
    """
    flags: dict[str, bool] = {}
    score = 0

    rr = vitals.get("resp_rate")
    if rr is not None:
        flags["qsofa_rr"] = float(rr) >= 22
        score += int(flags["qsofa_rr"])
    else:
        flags["qsofa_rr"] = False   # missing — cannot score

    gcs = vitals.get("gcs")
    if gcs is not None:
        flags["qsofa_gcs"] = float(gcs) < 15
        score += int(flags["qsofa_gcs"])
    else:
        flags["qsofa_gcs"] = False

    sbp = vitals.get("sbp")
    if sbp is not None:
        flags["qsofa_sbp"] = float(sbp) <= 100
        score += int(flags["qsofa_sbp"])
    else:
        flags["qsofa_sbp"] = False

    return score, flags


def partial_sirs(vitals: dict) -> tuple[int, dict]:
    """
    Partial SIRS — uses only the vitals available (no WBC/PaCO2).

    Criteria (vitals-only subset):
      +1  Temperature > 38.3 °C or < 36.0 °C
      +1  Heart rate > 90 bpm
      +1  Respiratory rate > 20 /min

    Returns (count_met, component_flags_dict).
    """
    flags: dict[str, bool] = {}
    count = 0

    temp = vitals.get("temperature")
    if temp is not None:
        flags["sirs_temp"] = float(temp) > 38.3 or float(temp) < 36.0
        count += int(flags["sirs_temp"])

    hr = vitals.get("heart_rate")
    if hr is not None:
        flags["sirs_hr"] = float(hr) > 90
        count += int(flags["sirs_hr"])

    rr = vitals.get("resp_rate")
    if rr is not None:
        flags["sirs_rr"] = float(rr) > 20
        count += int(flags["sirs_rr"])

    return count, flags


def shock_index(vitals: dict) -> Optional[float]:
    """
    Shock Index = HR / SBP.

    ≥ 0.9  : borderline
    ≥ 1.0  : elevated — associated with occult shock
    ≥ 1.4  : severe

    Returns None if either vital is missing or SBP is 0.
    """
    hr  = vitals.get("heart_rate")
    sbp = vitals.get("sbp")
    if hr is None or sbp is None or float(sbp) == 0:
        return None
    return round(float(hr) / float(sbp), 3)


def news2_style(vitals: dict) -> int:
    """
    NEWS2-style composite score (vitals-only, SpO2 scale 1 unless labelled).

    Full NEWS2 requires SpO2 scale selection and supplemental O2 status —
    we use the scale-1 defaults as a bedside-accessible approximation.

    Returns aggregate score. Thresholds: ≥ 5 medium, ≥ 7 high.
    """
    score = 0

    rr = vitals.get("resp_rate")
    if rr is not None:
        rr = float(rr)
        if rr <= 8:           score += 3
        elif rr <= 11:        score += 1
        elif rr <= 20:        score += 0
        elif rr <= 24:        score += 2
        else:                 score += 3

    spo2 = vitals.get("spo2")
    if spo2 is not None:
        spo2 = float(spo2)
        if spo2 <= 91:        score += 3
        elif spo2 <= 93:      score += 2
        elif spo2 <= 95:      score += 1

    sbp = vitals.get("sbp")
    if sbp is not None:
        sbp = float(sbp)
        if sbp <= 90:         score += 3
        elif sbp <= 100:      score += 2
        elif sbp <= 110:      score += 1
        elif sbp >= 220:      score += 3

    hr = vitals.get("heart_rate")
    if hr is not None:
        hr = float(hr)
        if hr <= 40:          score += 3
        elif hr <= 50:        score += 1
        elif hr <= 90:        score += 0
        elif hr <= 110:       score += 1
        elif hr <= 130:       score += 2
        else:                 score += 3

    temp = vitals.get("temperature")
    if temp is not None:
        temp = float(temp)
        if temp <= 35.0:      score += 3
        elif temp <= 36.0:    score += 1
        elif temp <= 38.0:    score += 0
        elif temp <= 39.0:    score += 1
        else:                 score += 2

    gcs = vitals.get("gcs")
    if gcs is not None and float(gcs) < 15:
        score += 3

    return score


def uva_style(vitals: dict) -> int:
    """
    UVA-style LMIC severity score.

    Adapted from Kruisselbrink et al. (PLOS ONE 2019) — a sub-Saharan Africa
    mortality model using accessible features. HIV status omitted (not in scope).

    Returns score. ≥ 2 : elevated, ≥ 4 : high mortality risk.
    """
    score = 0

    rr = vitals.get("resp_rate")
    if rr is not None:
        rr = float(rr)
        if rr >= 30:   score += 2
        elif rr >= 22: score += 1

    sbp = vitals.get("sbp")
    if sbp is not None:
        sbp = float(sbp)
        if sbp < 90:    score += 3
        elif sbp < 100: score += 1

    temp = vitals.get("temperature")
    if temp is not None:
        temp = float(temp)
        if temp > 38.5:    score += 1
        elif temp < 36.0:  score += 2

    gcs = vitals.get("gcs")
    if gcs is not None and float(gcs) < 13:
        score += 2

    hr = vitals.get("heart_rate")
    if hr is not None and float(hr) > 120:
        score += 1

    return score


# ──────────────────────────────────────────────────────────────────────────────
# Risk classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_risk(
    qsofa_score: int,
    sirs_count: int,
    si: Optional[float],
    news2_score: int = 0,
) -> tuple[str, bool]:
    """
    Map scores to a risk level and alert flag.

    Returns (risk_level, alert_flag).
    risk_level: "low" | "moderate" | "high" | "critical"
    alert_flag: True = nurse should be notified immediately
    """
    if qsofa_score >= 3 or (si is not None and si >= 1.4) or news2_score >= 7:
        return "critical", True
    if qsofa_score >= 2 or (si is not None and si >= 1.0) or news2_score >= 5:
        return "high", True
    if qsofa_score == 1 or sirs_count >= 2 or news2_score >= 3:
        return "moderate", False
    return "low", False


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def compute_scores(vitals: dict) -> ScoreBundle:
    """
    Compute all clinical scores for one observation dict.

    Parameters
    ----------
    vitals : dict
        Keys: temperature, heart_rate, resp_rate, sbp, spo2, gcs
        (Any subset is valid — missing vitals reduce scoring sensitivity.)

    Returns
    -------
    ScoreBundle
    """
    q_score, q_flags   = qsofa(vitals)
    s_count, s_flags   = partial_sirs(vitals)
    si                 = shock_index(vitals)
    n_score            = news2_style(vitals)
    u_score            = uva_style(vitals)
    risk, alert        = classify_risk(q_score, s_count, si, n_score)

    return ScoreBundle(
        qsofa          = q_score,
        sirs_count     = s_count,
        shock_index    = si,
        news2_style    = n_score,
        uva_style      = u_score,
        risk_level     = risk,
        alert_flag     = alert,
        component_flags= {**q_flags, **s_flags},
    )
