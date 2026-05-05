"""
sepsis_vitals
=============
Vitals-only sepsis prediction tooling for low-resource hospitals.

Public API::

    from sepsis_vitals import build_feature_set
    from sepsis_vitals.data_quality import summarize_vitals_quality
    from sepsis_vitals.scores import compute_scores
"""

from .features import build_feature_set
from .scores import compute_scores

__all__ = ["build_feature_set", "compute_scores"]
__version__ = "0.2.0"
