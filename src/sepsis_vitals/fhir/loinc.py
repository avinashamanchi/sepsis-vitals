"""
sepsis_vitals.fhir.loinc -- LOINC code mappings for FHIR vital sign observations.

Maps LOINC codes to internal vital sign names used by compute_scores() and
provides unit conversion helpers for common measurement systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# LOINC <-> internal vital name mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoincEntry:
    """Metadata for a single LOINC-coded vital sign."""

    code: str
    display: str
    internal_name: str
    unit: str
    ucum_unit: str  # UCUM unit code used in FHIR valueQuantity
    system: str = "http://loinc.org"


# Canonical table of supported vital signs.
LOINC_TABLE: tuple[LoincEntry, ...] = (
    LoincEntry(
        code="8310-5",
        display="Body temperature",
        internal_name="temperature",
        unit="Cel",
        ucum_unit="Cel",
    ),
    LoincEntry(
        code="8867-4",
        display="Heart rate",
        internal_name="heart_rate",
        unit="bpm",
        ucum_unit="/min",
    ),
    LoincEntry(
        code="9279-1",
        display="Respiratory rate",
        internal_name="resp_rate",
        unit="breaths/min",
        ucum_unit="/min",
    ),
    LoincEntry(
        code="8480-6",
        display="Systolic blood pressure",
        internal_name="sbp",
        unit="mmHg",
        ucum_unit="mm[Hg]",
    ),
    LoincEntry(
        code="8462-4",
        display="Diastolic blood pressure",
        internal_name="dbp",
        unit="mmHg",
        ucum_unit="mm[Hg]",
    ),
    LoincEntry(
        code="2708-6",
        display="Oxygen saturation in Arterial blood",
        internal_name="spo2",
        unit="%",
        ucum_unit="%",
    ),
    LoincEntry(
        code="9269-2",
        display="Glasgow coma score total",
        internal_name="gcs",
        unit="{score}",
        ucum_unit="{score}",
    ),
)

# Forward lookup: LOINC code -> LoincEntry
LOINC_TO_ENTRY: dict[str, LoincEntry] = {e.code: e for e in LOINC_TABLE}

# Forward lookup: LOINC code -> internal vital name
LOINC_TO_INTERNAL: dict[str, str] = {e.code: e.internal_name for e in LOINC_TABLE}

# Reverse lookup: internal vital name -> LoincEntry
INTERNAL_TO_ENTRY: dict[str, LoincEntry] = {e.internal_name: e for e in LOINC_TABLE}

# Reverse lookup: internal vital name -> LOINC code
INTERNAL_TO_LOINC: dict[str, str] = {e.internal_name: e.code for e in LOINC_TABLE}

# Set of all recognised LOINC codes for quick membership tests
SUPPORTED_LOINC_CODES: frozenset[str] = frozenset(LOINC_TO_INTERNAL)


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius, rounded to 1 decimal place."""
    return round((f - 32.0) * 5.0 / 9.0, 1)


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit, rounded to 1 decimal place."""
    return round(c * 9.0 / 5.0 + 32.0, 1)


def kpa_to_mmhg(kpa: float) -> float:
    """Convert kilopascals to millimetres of mercury."""
    return round(kpa * 7.50062, 1)


def mmhg_to_kpa(mmhg: float) -> float:
    """Convert millimetres of mercury to kilopascals."""
    return round(mmhg / 7.50062, 2)


def fraction_to_percent(frac: float) -> float:
    """Convert a 0-1 fraction (e.g. 0.97) to a percentage (97.0)."""
    return round(frac * 100.0, 1)


# ---------------------------------------------------------------------------
# Normalisation: convert a raw observation value to internal units
# ---------------------------------------------------------------------------

# Map of (LOINC code, source UCUM unit) -> conversion function.
# If the pair is absent the value is assumed to already be in canonical units.
_UNIT_CONVERTERS: dict[tuple[str, str], Callable[[float], float]] = {
    # Temperature: Fahrenheit -> Celsius
    ("8310-5", "[degF]"): fahrenheit_to_celsius,
    ("8310-5", "degF"): fahrenheit_to_celsius,
    # Blood pressure: kPa -> mmHg
    ("8480-6", "kPa"): kpa_to_mmhg,
    ("8462-4", "kPa"): kpa_to_mmhg,
    # SpO2: fraction -> percent
    ("2708-6", "1"): fraction_to_percent,
}


def normalise_value(loinc_code: str, value: float, source_unit: str | None) -> float:
    """Convert *value* to the canonical internal unit for *loinc_code*.

    If *source_unit* matches the canonical UCUM unit (or is ``None``), the
    value is returned unchanged.  Otherwise an appropriate converter is
    applied if one exists; a ``ValueError`` is raised when the unit is
    unrecognised.
    """
    entry = LOINC_TO_ENTRY.get(loinc_code)
    if entry is None:
        raise ValueError(f"Unsupported LOINC code: {loinc_code}")

    if source_unit is None or source_unit == entry.ucum_unit:
        return value

    converter = _UNIT_CONVERTERS.get((loinc_code, source_unit))
    if converter is not None:
        return converter(value)

    raise ValueError(
        f"Cannot convert unit '{source_unit}' to '{entry.ucum_unit}' "
        f"for LOINC {loinc_code} ({entry.display})"
    )
