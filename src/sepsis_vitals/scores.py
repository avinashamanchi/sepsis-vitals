"""
sepsis_vitals/scores.py – Clinical scoring functions for sepsis screening.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def qsofa(vitals: dict) -> tuple[int, dict]:
    """Quick SOFA score. Returns (score, flags_dict)."""
    score = 0
    flags: dict[str, bool] = {}

    # Respiratory rate >= 22
    if "resp_rate" in vitals:
        fired = vitals["resp_rate"] >= 22
        flags["qsofa_rr"] = fired
        if fired:
            score += 1
    else:
        flags["qsofa_rr"] = False

    # GCS <= 13
    if "gcs" in vitals:
        fired = vitals["gcs"] <= 13
        flags["qsofa_gcs"] = fired
        if fired:
            score += 1
    else:
        flags["qsofa_gcs"] = False

    # SBP <= 100
    if "sbp" in vitals:
        fired = vitals["sbp"] <= 100
        flags["qsofa_sbp"] = fired
        if fired:
            score += 1
    else:
        flags["qsofa_sbp"] = False

    return score, flags


def partial_sirs(vitals: dict) -> tuple[int, dict]:
    """Partial SIRS criteria (temperature, heart rate, respiratory rate).
    Returns (count, flags_dict).
    """
    count = 0
    flags: dict[str, bool] = {}

    # Temperature > 38.3 or < 36
    if "temperature" in vitals:
        temp = vitals["temperature"]
        fired = temp > 38.3 or temp < 36.0
        flags["sirs_temp"] = fired
        if fired:
            count += 1
    else:
        flags["sirs_temp"] = False

    # Heart rate > 90
    if "heart_rate" in vitals:
        fired = vitals["heart_rate"] > 90
        flags["sirs_hr"] = fired
        if fired:
            count += 1
    else:
        flags["sirs_hr"] = False

    # Respiratory rate > 20
    if "resp_rate" in vitals:
        fired = vitals["resp_rate"] > 20
        flags["sirs_rr"] = fired
        if fired:
            count += 1
    else:
        flags["sirs_rr"] = False

    return count, flags


def shock_index(vitals: dict) -> float | None:
    """Shock index = HR / SBP, rounded to 3 decimal places.
    Returns None if HR or SBP is missing, or SBP is 0.
    """
    hr = vitals.get("heart_rate")
    sbp = vitals.get("sbp")

    if hr is None or sbp is None or sbp == 0:
        return None

    return round(hr / sbp, 3)


def news2_style(vitals: dict) -> int:
    """NEWS2-style aggregate score. Returns total score."""
    total = 0

    # Respiratory rate
    if "resp_rate" in vitals:
        rr = vitals["resp_rate"]
        if rr <= 8:
            total += 3
        elif rr <= 11:
            total += 1
        elif rr <= 20:
            total += 0
        elif rr <= 24:
            total += 2
        else:  # >= 25
            total += 3

    # SpO2
    if "spo2" in vitals:
        spo2 = vitals["spo2"]
        if spo2 <= 91:
            total += 3
        elif spo2 <= 93:
            total += 2
        elif spo2 <= 95:
            total += 1
        else:  # >= 96
            total += 0

    # SBP
    if "sbp" in vitals:
        sbp = vitals["sbp"]
        if sbp <= 90:
            total += 3
        elif sbp <= 100:
            total += 2
        elif sbp <= 110:
            total += 1
        elif sbp <= 219:
            total += 0
        else:  # >= 220
            total += 3

    # Heart rate
    if "heart_rate" in vitals:
        hr = vitals["heart_rate"]
        if hr <= 40:
            total += 3
        elif hr <= 50:
            total += 1
        elif hr <= 90:
            total += 0
        elif hr <= 110:
            total += 1
        elif hr <= 130:
            total += 2
        else:  # >= 131
            total += 3

    # Temperature
    if "temperature" in vitals:
        temp = vitals["temperature"]
        if temp <= 35.0:
            total += 3
        elif temp <= 36.0:
            total += 1
        elif temp <= 38.0:
            total += 0
        elif temp <= 39.0:
            total += 1
        else:  # >= 39.1
            total += 2

    # GCS
    if "gcs" in vitals:
        gcs = vitals["gcs"]
        if gcs < 15:
            total += 3
        # gcs == 15 -> 0

    return total


def uva_style(vitals: dict) -> int:
    """UVA/Kruisselbrink-style screening score. Returns total score."""
    total = 0

    # Respiratory rate: < 10 or > 29 -> 2
    if "resp_rate" in vitals:
        rr = vitals["resp_rate"]
        if rr < 10 or rr > 29:
            total += 2

    # SBP: < 90 -> 3
    if "sbp" in vitals:
        sbp = vitals["sbp"]
        if sbp < 90:
            total += 3

    # Temperature: < 36.0 -> 2
    if "temperature" in vitals:
        temp = vitals["temperature"]
        if temp < 36.0:
            total += 2

    # GCS: < 14 -> 2
    if "gcs" in vitals:
        gcs = vitals["gcs"]
        if gcs < 14:
            total += 2

    return total


def classify_risk(
    qsofa_score: int,
    sirs: int,
    si: float | None,
    news2: int,
    lactate: float | None = None,
) -> tuple[str, bool]:
    """Classify sepsis risk level based on composite scores and labs.
    Returns (level, alert_flag).
    """
    si_val = si if si is not None else 0.0

    # Critical — lactate >= 4 mmol/L is a septic shock criterion (SSC 2021)
    if qsofa_score >= 3 or news2 >= 7 or si_val >= 1.3:
        return ("critical", True)
    if lactate is not None and lactate >= 4.0:
        return ("critical", True)

    # High — lactate >= 2 mmol/L indicates tissue hypoperfusion
    if qsofa_score >= 2 or si_val >= 1.0 or news2 >= 5:
        return ("high", True)
    if lactate is not None and lactate >= 2.0:
        return ("high", True)

    # Moderate
    if qsofa_score >= 1 or sirs >= 2:
        return ("moderate", False)

    # Low
    return ("low", False)


@dataclass
class ScoreBundle:
    """Aggregated result of all scoring functions."""

    qsofa: int
    sirs_count: int
    shock_index: float | None
    news2_style: int
    uva_style: int
    risk_level: str
    alert_flag: bool
    component_flags: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "qsofa": self.qsofa,
            "sirs_count": self.sirs_count,
            "shock_index": self.shock_index,
            "news2_style": self.news2_style,
            "uva_style": self.uva_style,
            "risk_level": self.risk_level,
            "alert_flag": self.alert_flag,
        }


def compute_scores(vitals: dict) -> ScoreBundle:
    """Orchestrate all scoring functions and return a ScoreBundle."""
    q_score, q_flags = qsofa(vitals)
    s_count, s_flags = partial_sirs(vitals)
    si = shock_index(vitals)
    n2 = news2_style(vitals)
    uva = uva_style(vitals)

    lactate = vitals.get("lactate")
    level, alert = classify_risk(q_score, s_count, si, n2, lactate=lactate)

    # Merge all component flags
    component_flags = {**q_flags, **s_flags}

    return ScoreBundle(
        qsofa=q_score,
        sirs_count=s_count,
        shock_index=si,
        news2_style=n2,
        uva_style=uva,
        risk_level=level,
        alert_flag=alert,
        component_flags=component_flags,
    )
