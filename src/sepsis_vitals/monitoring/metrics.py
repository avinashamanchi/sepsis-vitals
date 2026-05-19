"""Monitoring metrics: Population Stability Index and alert fatigue analysis."""

from __future__ import annotations

import numpy as np
from statistics import median


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    buckets: int = 10,
) -> float:
    """Compute Population Stability Index between reference and current arrays.

    Returns 0.0 if either array has fewer than 10 elements.
    """
    if len(reference) < 10 or len(current) < 10:
        return 0.0

    # Compute breakpoints from the reference distribution (deciles)
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    # Ensure the outer edges capture everything
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Convert to proportions
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Replace zeros with a small epsilon to avoid log(0) and division by zero
    eps = 1e-8
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_distribution_drift(
    ref_data: dict,
    cur_data: dict,
    threshold: float = 0.2,
) -> dict:
    """Check distribution drift for each vital using PSI.

    Parameters
    ----------
    ref_data : dict mapping vital names to lists of values
    cur_data : dict mapping vital names to lists of values
    threshold : PSI threshold above which drift is flagged

    Returns
    -------
    dict with per-vital results (each containing a "psi" key) and an
    "overall_drift" boolean (True if any vital's PSI exceeds threshold).
    """
    results: dict = {}
    any_drift = False

    for vital in ref_data:
        ref_arr = np.asarray(ref_data[vital], dtype=float)
        cur_arr = np.asarray(cur_data[vital], dtype=float)
        psi_value = compute_psi(ref_arr, cur_arr)
        drifted = psi_value > threshold
        results[vital] = {"psi": psi_value, "drift": drifted}
        if drifted:
            any_drift = True

    results["overall_drift"] = any_drift
    return results


def compute_alert_fatigue_metrics(rows: list[dict]) -> dict:
    """Compute alert fatigue metrics from a list of alert action rows.

    Each row must have keys "action" (str) and "time_to_action_s" (float).

    Returns a dict with override_rate, median_response_time_s, and fatigue_level,
    or {"error": ...} if the input list is empty.
    """
    if not rows:
        return {"error": "No alert data provided"}

    total = len(rows)
    dismissed_count = sum(1 for r in rows if r["action"] == "dismissed")
    override_rate = dismissed_count / total

    times = [r["time_to_action_s"] for r in rows]
    median_response = median(times)

    if override_rate >= 0.7:
        fatigue_level = "critical"
    elif override_rate >= 0.5:
        fatigue_level = "elevated"
    else:
        fatigue_level = "normal"

    return {
        "override_rate": override_rate,
        "median_response_time_s": median_response,
        "fatigue_level": fatigue_level,
    }
