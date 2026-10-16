"""
sepsis_vitals.fhir.ingest_models

Shared types for vitals ingestion: LOINC maps, VitalsReading and HL7Message.
Split out of sepsis_vitals.fhir.listener, which re-exports every name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

# Keep the original logger name so existing log routing is unchanged.
logger = logging.getLogger("sepsis_vitals.fhir.listener")

# ---------------------------------------------------------------------------
# LOINC code -> internal vital sign name mapping
# ---------------------------------------------------------------------------

LOINC_VITAL_MAP: Dict[str, str] = {
    "8310-5": "temperature",       # Body temperature
    "8867-4": "heart_rate",        # Heart rate
    "9279-1": "resp_rate",         # Respiratory rate
    "8480-6": "sbp",              # Systolic blood pressure
    "8462-4": "dbp",              # Diastolic blood pressure
    "2708-6": "spo2",             # Oxygen saturation
    "9269-2": "gcs",              # Glasgow coma scale total
    "8478-0": "map",              # Mean arterial pressure
    "2524-7": "lactate",          # Lactate [Moles/volume]
    "6690-2": "wbc",              # Leukocytes [#/volume]
    "33959-8": "procalcitonin",   # Procalcitonin [Mass/volume]
}

# Also support common display names (lowercased for matching)
DISPLAY_NAME_MAP: Dict[str, str] = {
    "body temperature": "temperature",
    "heart rate": "heart_rate",
    "respiratory rate": "resp_rate",
    "systolic blood pressure": "sbp",
    "diastolic blood pressure": "dbp",
    "oxygen saturation": "spo2",
    "glasgow coma scale": "gcs",
    "mean arterial pressure": "map",
    "lactate": "lactate",
    "leukocytes": "wbc",
    "procalcitonin": "procalcitonin",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VitalsReading:
    """A single vitals reading extracted from an HL7/FHIR message."""

    patient_id: str
    timestamp: str
    vitals: Dict[str, float]
    source: str  # "hl7v2" or "fhir_r4"
    raw_message_id: Optional[str] = None

    def __repr__(self) -> str:
        vital_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(self.vitals.items())
        )
        return (
            f"VitalsReading(patient={self.patient_id!r}, "
            f"source={self.source!r}, {vital_summary})"
        )


@dataclass
class HL7Message:
    """Parsed HL7v2 message."""

    segments: List[List[str]]
    message_type: str = ""
    patient_id: str = ""
    timestamp: str = ""

    def get_segments(self, segment_type: str) -> List[List[str]]:
        """Return all segments matching *segment_type* (e.g. ``"OBX"``)."""
        return [seg for seg in self.segments if seg and seg[0] == segment_type]

    def get_segment(self, segment_type: str) -> Optional[List[str]]:
        """Return the first segment matching *segment_type*, or ``None``."""
        for seg in self.segments:
            if seg and seg[0] == segment_type:
                return seg
        return None

    @property
    def message_control_id(self) -> str:
        """Extract MSH-10 (message control ID) for ACK correlation."""
        msh = self.get_segment("MSH")
        if msh and len(msh) > 10:
            return msh[10]
        return ""

    @property
    def sending_application(self) -> str:
        """Extract MSH-3 (sending application)."""
        msh = self.get_segment("MSH")
        if msh and len(msh) > 3:
            return msh[3]
        return ""

    @property
    def sending_facility(self) -> str:
        """Extract MSH-4 (sending facility)."""
        msh = self.get_segment("MSH")
        if msh and len(msh) > 4:
            return msh[4]
        return ""


