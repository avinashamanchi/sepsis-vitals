"""
sepsis_vitals.bundles
~~~~~~~~~~~~~~~~~~~~~~
Hour-1 Sepsis Bundle tracking.

Turns the static "start the bundle" recommendation into a stateful,
time-boxed clinical workflow: a running clock against the 60-minute
antibiotic target, per-task completion timestamps, and automatic
compliance scoring aligned with the Surviving Sepsis Campaign
Hour-1 Bundle (Levy et al., Crit Care Med 2018; SSC 2021 update).
"""

from sepsis_vitals.bundles.protocol import (  # noqa: F401
    HOUR1_BUNDLE,
    BundleTaskSpec,
    get_task_spec,
    list_task_keys,
)
