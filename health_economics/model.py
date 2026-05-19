"""
Health economics model for sepsis early-warning system cost-effectiveness analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EconomicsParams:
    annual_encounters: int = 4000
    sepsis_prevalence: float = 0.15
    mortality_rate: float = 0.25
    model_sensitivity: float = 0.85
    specificity: float = 0.80
    relative_risk_reduction: float = 0.30
    cost_per_sepsis_death: float = 50000
    software_annual_cost: float = 25000
    qaly_per_death_averted: float = 8.0
    discount_rate: float = 0.03


class HealthEconomicsModel:
    def __init__(self, params: EconomicsParams = None):
        if params is None:
            params = EconomicsParams()
        self.params = params

    def deaths_without_model(self) -> float:
        p = self.params
        return p.annual_encounters * p.sepsis_prevalence * p.mortality_rate

    def deaths_averted(self) -> float:
        p = self.params
        detected_sepsis_cases = p.annual_encounters * p.sepsis_prevalence * p.model_sensitivity
        return detected_sepsis_cases * p.mortality_rate * p.relative_risk_reduction

    def roi_pct(self) -> float:
        p = self.params
        clinical_savings = self.deaths_averted() * p.cost_per_sepsis_death
        return ((clinical_savings - p.software_annual_cost) / p.software_annual_cost) * 100

    def qalys_gained(self) -> float:
        return self.deaths_averted() * self.params.qaly_per_death_averted

    def cost_per_qaly_usd(self) -> float:
        qalys = self.qalys_gained()
        if qalys == 0:
            return float("inf")
        return self.params.software_annual_cost / qalys

    def alerts_per_100_enc(self) -> float:
        p = self.params
        tp_rate = p.sepsis_prevalence * p.model_sensitivity
        fp_rate = (1 - p.sepsis_prevalence) * (1 - p.specificity)
        return (tp_rate + fp_rate) * 100

    def break_even_sensitivity(self) -> Optional[float]:
        """Binary search for the minimum sensitivity where ROI >= 0."""
        original_sensitivity = self.params.model_sensitivity
        lo, hi = 0.0, 1.0

        # First check if even perfect sensitivity can't break even
        self.params.model_sensitivity = 1.0
        if self.roi_pct() < 0:
            self.params.model_sensitivity = original_sensitivity
            return None

        # Check if sensitivity=0 already breaks even (shouldn't happen, but handle it)
        self.params.model_sensitivity = 0.0
        if self.roi_pct() >= 0:
            self.params.model_sensitivity = original_sensitivity
            return 0.0

        # Binary search
        for _ in range(100):
            mid = (lo + hi) / 2
            self.params.model_sensitivity = mid
            if self.roi_pct() >= 0:
                hi = mid
            else:
                lo = mid

        self.params.model_sensitivity = original_sensitivity
        return hi

    def full_report(self) -> dict:
        p = self.params
        da = self.deaths_averted()
        dwm = self.deaths_without_model()
        clinical_savings = da * p.cost_per_sepsis_death
        roi = self.roi_pct()
        qalys = self.qalys_gained()
        cpq = self.cost_per_qaly_usd()
        alerts = self.alerts_per_100_enc()
        be = self.break_even_sensitivity()

        summary = (
            f"With {p.annual_encounters} annual encounters and {p.sepsis_prevalence:.0%} sepsis prevalence, "
            f"the model averts {da:.1f} deaths per year (from {dwm:.1f} baseline deaths). "
            f"ROI is {roi:.1f}% with {qalys:.1f} QALYs gained at ${cpq:,.0f}/QALY."
        )

        return {
            "epidemiology": {
                "annual_encounters": p.annual_encounters,
                "sepsis_prevalence": p.sepsis_prevalence,
                "sepsis_cases": p.annual_encounters * p.sepsis_prevalence,
                "deaths_without_model": dwm,
            },
            "mortality_impact": {
                "deaths_averted": da,
                "deaths_without_model": dwm,
                "relative_risk_reduction": p.relative_risk_reduction,
            },
            "clinical_savings_usd": clinical_savings,
            "software_costs_usd": p.software_annual_cost,
            "health_economics": {
                "qalys_gained": qalys,
                "cost_per_qaly_usd": cpq,
                "discount_rate": p.discount_rate,
            },
            "roi": {
                "roi_pct": roi,
                "break_even_sensitivity": be,
                "alerts_per_100_encounters": alerts,
            },
            "summary": summary,
        }
