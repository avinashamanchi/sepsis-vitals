"""Threshold policy and decision-support notice attached to every risk result.

The cut-offs below are engineering placeholders. They exist so the prototype
runs end to end. They were not derived from patient data and no clinician
has approved them; setting real ones is a clinical-governance decision
(``compliance/clinical_requirements_needed.md``). Every result records the
policy id and a digest of its exact values, so the cut-offs that produced a
result can always be recovered, including after they change.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

PLACEHOLDER_STATUS = "engineering-placeholder: not clinically derived or approved"

# Model-output bands and the alert cut-off used when the artifact carries no
# operating points of its own.
PLACEHOLDER_BANDS = {"moderate": 0.25, "high": 0.50, "critical": 0.75}
PLACEHOLDER_ALERT_CUTOFF = 0.6

ALERT_RULE = (
    "alert if the rule-based scores raise an alert, or the combined level is "
    "high or critical, or the model output exceeds alert_cutoff"
)

DECISION_SUPPORT_NOTICE = (
    "Decision support for research evaluation only. This is not a diagnosis and "
    "must not be used to guide patient care. The model was developed on synthetic "
    "data and has no clinical validation."
)


def _digest(policy: Dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def threshold_policy(dual_thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """The active policy, with its id and a SHA-256 digest of its values."""
    if dual_thresholds:
        policy: Dict[str, Any] = {
            "id": "artifact-operating-points-v1",
            "status": PLACEHOLDER_STATUS,
            "model_bands": {
                "moderate": dual_thresholds["on_demand"]["threshold"],
                "high": dual_thresholds["continuous"]["threshold"],
                "critical": dual_thresholds["continuous"]["threshold"] + 0.15,
            },
            "alert_cutoff": PLACEHOLDER_ALERT_CUTOFF,
            "alert_rule": ALERT_RULE,
        }
    else:
        policy = {
            "id": "placeholder-bands-v1",
            "status": PLACEHOLDER_STATUS,
            "model_bands": dict(PLACEHOLDER_BANDS),
            "alert_cutoff": PLACEHOLDER_ALERT_CUTOFF,
            "alert_rule": ALERT_RULE,
        }
    return {**policy, "sha256": _digest(policy)}


def model_level(prob: float, policy: Dict[str, Any]) -> str:
    """Band of the model output alone under ``policy``."""
    bands = policy["model_bands"]
    if prob >= bands["critical"]:
        return "critical"
    if prob >= bands["high"]:
        return "high"
    if prob >= bands["moderate"]:
        return "moderate"
    return "low"
