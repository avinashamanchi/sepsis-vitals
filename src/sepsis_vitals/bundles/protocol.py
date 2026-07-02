"""
sepsis_vitals.bundles.protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Declarative definition of the Surviving Sepsis Campaign **Hour-1 Bundle**.

The Hour-1 Bundle (Levy, Evans & Rhodes, *Crit Care Med* 2018) collapses the
former 3- and 6-hour bundles into a single set of interventions to be
*initiated* within one hour of sepsis recognition:

1. Measure lactate; re-measure if initial lactate > 2 mmol/L.
2. Obtain blood cultures **before** administering antibiotics.
3. Administer broad-spectrum antibiotics.
4. Begin rapid crystalloid (30 mL/kg) for hypotension or lactate >= 4.
5. Apply vasopressors if hypotensive during/after resuscitation to keep
   MAP >= 65 mmHg.

This module contains **no state** -- it is the immutable clinical protocol
that :mod:`sepsis_vitals.bundles.service` instantiates per patient.  Keeping
it declarative means the bundle can be localised, versioned, or swapped for a
paediatric protocol without touching the service or persistence layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


PROTOCOL_VERSION = "ssc-hour1-2021"


@dataclass(frozen=True)
class BundleTaskSpec:
    """Immutable specification of a single Hour-1 bundle task.

    Attributes
    ----------
    key:
        Stable machine identifier (used in the API and DB).
    label:
        Human-readable task name (i18n key lives in the frontend locales).
    target_minutes:
        Minutes from bundle start within which the task should be *initiated*.
        ``60`` for the core Hour-1 interventions.
    order:
        Suggested display / execution order.
    critical:
        Whether omission materially breaks bundle compliance.  All five core
        SSC tasks are critical; documentation tasks may not be.
    depends_on:
        Optional key of a task that should be completed first.  Blood cultures
        depend on nothing but antibiotics *should* follow cultures, so
        ``antibiotics`` depends on ``blood_cultures`` for ordering guidance
        (a soft dependency -- never a hard block, since clinical judgment wins).
    conditional:
        Human-readable trigger describing when the task applies.  ``None``
        means always applicable.
    """

    key: str
    label: str
    target_minutes: int = 60
    order: int = 0
    critical: bool = True
    depends_on: Optional[str] = None
    conditional: Optional[str] = None


@dataclass(frozen=True)
class BundleProtocol:
    """A named, versioned ordered collection of :class:`BundleTaskSpec`."""

    version: str
    name: str
    tasks: List[BundleTaskSpec] = field(default_factory=list)

    def task_map(self) -> Dict[str, BundleTaskSpec]:
        return {t.key: t for t in self.tasks}


HOUR1_BUNDLE = BundleProtocol(
    version=PROTOCOL_VERSION,
    name="Surviving Sepsis Campaign Hour-1 Bundle",
    tasks=[
        BundleTaskSpec(
            key="lactate_initial",
            label="Measure initial serum lactate",
            target_minutes=60,
            order=1,
        ),
        BundleTaskSpec(
            key="blood_cultures",
            label="Obtain blood cultures before antibiotics",
            target_minutes=60,
            order=2,
        ),
        BundleTaskSpec(
            key="antibiotics",
            label="Administer broad-spectrum antibiotics",
            target_minutes=60,
            order=3,
            depends_on="blood_cultures",
        ),
        BundleTaskSpec(
            key="fluids",
            label="Begin 30 mL/kg crystalloid",
            target_minutes=60,
            order=4,
            conditional="Hypotension (MAP < 65) or lactate >= 4 mmol/L",
        ),
        BundleTaskSpec(
            key="vasopressors",
            label="Start vasopressors to keep MAP >= 65",
            target_minutes=60,
            order=5,
            conditional="Hypotensive during or after fluid resuscitation",
        ),
        BundleTaskSpec(
            key="lactate_repeat",
            label="Re-measure lactate",
            target_minutes=180,
            order=6,
            critical=False,
            depends_on="lactate_initial",
            conditional="Initial lactate > 2 mmol/L",
        ),
    ],
)


_TASK_MAP: Dict[str, BundleTaskSpec] = HOUR1_BUNDLE.task_map()


def get_task_spec(key: str) -> BundleTaskSpec:
    """Return the :class:`BundleTaskSpec` for *key*.

    Raises
    ------
    KeyError
        If *key* is not part of the active protocol.
    """
    if key not in _TASK_MAP:
        raise KeyError(f"Unknown bundle task '{key}'")
    return _TASK_MAP[key]


def list_task_keys() -> List[str]:
    """Return the ordered list of task keys in the active protocol."""
    return [t.key for t in sorted(HOUR1_BUNDLE.tasks, key=lambda s: s.order)]


def applicable_tasks(
    *,
    lactate: Optional[float] = None,
    map_pressure: Optional[float] = None,
    sbp: Optional[int] = None,
) -> List[BundleTaskSpec]:
    """Return the subset of protocol tasks that apply given current context.

    Conditional tasks (fluids, vasopressors, repeat lactate) are only
    surfaced when their clinical trigger is met.  Unconditional tasks are
    always included.  When the relevant value is unknown we *include* the
    task (fail-open) so the clinician is never silently steered away from a
    life-saving intervention.
    """
    out: List[BundleTaskSpec] = []
    hypotensive = _is_hypotensive(map_pressure, sbp)

    for spec in sorted(HOUR1_BUNDLE.tasks, key=lambda s: s.order):
        if spec.key == "fluids":
            if hypotensive or lactate is None or lactate >= 4.0:
                out.append(spec)
        elif spec.key == "vasopressors":
            if hypotensive or map_pressure is None:
                out.append(spec)
        elif spec.key == "lactate_repeat":
            if lactate is None or lactate > 2.0:
                out.append(spec)
        else:
            out.append(spec)
    return out


def _is_hypotensive(
    map_pressure: Optional[float], sbp: Optional[int]
) -> bool:
    if map_pressure is not None and map_pressure < 65:
        return True
    if sbp is not None and sbp < 90:
        return True
    return False
