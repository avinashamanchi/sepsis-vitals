"""
sepsis_vitals.fhir.observation_parser

FHIR R4 Observation parsing into VitalsReading objects.
Split out of sepsis_vitals.fhir.listener, which re-exports every name.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sepsis_vitals.fhir.ingest_models import DISPLAY_NAME_MAP, LOINC_VITAL_MAP, VitalsReading

# Keep the original logger name so existing log routing is unchanged.
logger = logging.getLogger("sepsis_vitals.fhir.listener")

# ---------------------------------------------------------------------------
# FHIR R4 Observation Parser
# ---------------------------------------------------------------------------


class FHIRObservationParser:
    """Parse FHIR R4 Observation resources for vital sign extraction.

    This parser is intentionally lightweight and stdlib-only.  It operates
    on plain ``dict`` objects (parsed JSON) rather than requiring a FHIR
    library.  For full FHIR resource handling (serialisation, validation,
    patient parsing), see ``sepsis_vitals.fhir.resources``.
    """

    def parse_bundle(self, bundle: dict) -> List[VitalsReading]:
        """Parse a FHIR Bundle of Observation resources.

        Handles Bundle types ``transaction``, ``batch``, ``collection``,
        and ``searchset``.  Observations are grouped by patient reference
        and effective datetime to produce one ``VitalsReading`` per
        patient-timestamp pair.

        Parameters
        ----------
        bundle : dict
            A FHIR Bundle resource as a parsed JSON dict.

        Returns
        -------
        list[VitalsReading]
            One reading per distinct patient/timestamp combination.
        """
        resource_type = bundle.get("resourceType", "")
        if resource_type != "Bundle":
            logger.warning(
                "Expected resourceType 'Bundle', got %r; "
                "attempting to parse as single Observation",
                resource_type,
            )
            single = self._parse_single_to_reading(bundle)
            return [single] if single else []

        entries = bundle.get("entry", [])
        if not entries:
            logger.info("Empty FHIR Bundle received (no entries)")
            return []

        # Group observations by (patient_id, effective_datetime) so that
        # a bundle with multiple vitals for the same patient at the same
        # time produces one VitalsReading.
        groups: Dict[tuple, Dict[str, float]] = {}
        group_meta: Dict[tuple, str] = {}  # key -> patient_id

        for entry in entries:
            resource = entry.get("resource", entry)
            if resource.get("resourceType") != "Observation":
                continue

            patient_id = self._extract_patient_id(resource)
            effective = resource.get(
                "effectiveDateTime",
                resource.get("effectiveInstant", ""),
            )

            vital = self.parse_observation(resource, patient_id=patient_id)
            if vital is None:
                continue

            key = (patient_id, effective)
            if key not in groups:
                groups[key] = {}
                group_meta[key] = patient_id
            groups[key].update(vital)

        readings: List[VitalsReading] = []
        bundle_id = bundle.get("id")

        for (patient_id, effective), vitals in groups.items():
            if not vitals:
                continue
            timestamp = effective or datetime.now(timezone.utc).isoformat()
            readings.append(
                VitalsReading(
                    patient_id=patient_id,
                    timestamp=timestamp,
                    vitals=vitals,
                    source="fhir_r4",
                    raw_message_id=bundle_id,
                )
            )

        logger.info(
            "Parsed FHIR Bundle: %d entries -> %d readings (%d vitals total)",
            len(entries),
            len(readings),
            sum(len(r.vitals) for r in readings),
        )
        return readings

    def parse_observation(
        self,
        obs: dict,
        patient_id: str = "unknown",
    ) -> Optional[Dict[str, float]]:
        """Parse a single FHIR Observation resource into a vital sign dict.

        Parameters
        ----------
        obs : dict
            A FHIR Observation resource as a parsed JSON dict.
        patient_id : str
            Fallback patient ID if the Observation does not carry a
            ``subject`` reference.

        Returns
        -------
        dict[str, float] | None
            A single-entry dict mapping the internal vital name to its
            numeric value, or ``None`` if the Observation is not a
            recognised vital sign.
        """
        if obs.get("resourceType") != "Observation":
            return None

        # Resolve LOINC code from coding array
        vital_name = self._resolve_vital_name(obs)
        if vital_name is None:
            return None

        # Extract numeric value
        value = self._extract_value(obs)
        if value is None:
            logger.debug(
                "No numeric value in Observation %s for %s",
                obs.get("id", "?"), vital_name,
            )
            return None

        return {vital_name: value}

    # -- private helpers ---------------------------------------------------

    def _parse_single_to_reading(
        self, resource: dict
    ) -> Optional[VitalsReading]:
        """Attempt to parse a non-Bundle resource as a single Observation."""
        if resource.get("resourceType") != "Observation":
            return None
        patient_id = self._extract_patient_id(resource)
        vital = self.parse_observation(resource, patient_id=patient_id)
        if vital is None:
            return None
        effective = resource.get(
            "effectiveDateTime",
            resource.get("effectiveInstant", ""),
        )
        return VitalsReading(
            patient_id=patient_id,
            timestamp=effective or datetime.now(timezone.utc).isoformat(),
            vitals=vital,
            source="fhir_r4",
            raw_message_id=resource.get("id"),
        )

    @staticmethod
    def _extract_patient_id(obs: dict) -> str:
        """Extract patient ID from Observation.subject.reference."""
        subject = obs.get("subject", {})
        reference = subject.get("reference", "")
        if reference:
            # "Patient/12345" -> "12345"
            return reference.split("/")[-1] if "/" in reference else reference
        return "unknown"

    @staticmethod
    def _resolve_vital_name(obs: dict) -> Optional[str]:
        """Resolve the internal vital name from Observation.code.coding."""
        codings = obs.get("code", {}).get("coding", [])
        for coding in codings:
            code = coding.get("code", "")
            system = coding.get("system", "")

            # Primary lookup: LOINC code
            if system == "http://loinc.org" or not system:
                vital_name = LOINC_VITAL_MAP.get(code)
                if vital_name is not None:
                    return vital_name

        # Fallback: match on display text
        code_text = obs.get("code", {}).get("text", "")
        if code_text:
            vital_name = DISPLAY_NAME_MAP.get(code_text.lower())
            if vital_name is not None:
                return vital_name

        # Try display fields in coding array
        for coding in codings:
            display = coding.get("display", "")
            if display:
                vital_name = DISPLAY_NAME_MAP.get(display.lower())
                if vital_name is not None:
                    return vital_name

        return None

    @staticmethod
    def _extract_value(obs: dict) -> Optional[float]:
        """Extract a numeric value from the Observation.

        Checks ``valueQuantity.value``, ``valueInteger``, and
        ``valueDecimal`` in order.  Also handles component-based
        observations (e.g. blood pressure with systolic/diastolic
        components).
        """
        # Primary: valueQuantity
        vq = obs.get("valueQuantity")
        if vq is not None:
            val = vq.get("value")
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

        # Scalar value types
        for key in ("valueDecimal", "valueInteger", "valueString"):
            if key in obs:
                try:
                    return float(obs[key])
                except (ValueError, TypeError):
                    pass

        return None


