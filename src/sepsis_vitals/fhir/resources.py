"""
sepsis_vitals.fhir.resources -- FHIR R4 resource parsing and serialisation.

Pure-Python helpers that convert between HL7 FHIR R4 JSON resources and the
internal data structures used by sepsis-vitals.  No external FHIR libraries
are required -- all operations are dict/JSON manipulation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sepsis_vitals.fhir.loinc import (
    INTERNAL_TO_ENTRY,
    LOINC_TO_ENTRY,
    SUPPORTED_LOINC_CODES,
    normalise_value,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FHIR_CONTENT_TYPE = "application/fhir+json"
LOINC_SYSTEM = "http://loinc.org"
SNOMED_SYSTEM = "http://snomed.info/sct"
FHIR_RISK_PROBABILITY_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/risk-probability"
)


def _new_id() -> str:
    """Generate a random UUID string for resource ids."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# OperationOutcome builder (FHIR error responses)
# ---------------------------------------------------------------------------


def operation_outcome(
    severity: str,
    code: str,
    diagnostics: str,
    *,
    http_status: int = 400,
) -> dict[str, Any]:
    """Build a FHIR OperationOutcome resource dict.

    Parameters
    ----------
    severity : str
        ``fatal | error | warning | information``
    code : str
        FHIR issue-type code, e.g. ``invalid``, ``not-found``, ``exception``.
    diagnostics : str
        Human-readable diagnostic message.
    http_status : int
        Suggested HTTP status code (carried in ``_http_status`` key for the
        router to pick up; not part of the FHIR payload itself).
    """
    return {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": severity,
                "code": code,
                "diagnostics": diagnostics,
            }
        ],
        "_http_status": http_status,
    }


# ---------------------------------------------------------------------------
# FHIRPatient -- inbound parsing
# ---------------------------------------------------------------------------


@dataclass
class FHIRPatient:
    """Parsed representation of a FHIR Patient resource."""

    resource_id: str
    external_id: str | None = None
    family_name: str | None = None
    given_name: str | None = None
    birth_date: str | None = None
    gender: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- factories --------------------------------------------------------

    @classmethod
    def from_fhir(cls, resource: dict[str, Any]) -> FHIRPatient:
        """Parse a FHIR Patient JSON dict into a ``FHIRPatient``.

        Raises ``ValueError`` if *resourceType* is not ``Patient``.
        """
        if resource.get("resourceType") != "Patient":
            raise ValueError(
                f"Expected resourceType 'Patient', "
                f"got '{resource.get('resourceType')}'"
            )

        resource_id = resource.get("id", _new_id())

        # Extract MRN / external identifier
        external_id: str | None = None
        for ident in resource.get("identifier", []):
            if ident.get("type", {}).get("coding", [{}])[0].get("code") == "MR":
                external_id = ident.get("value")
                break
        if external_id is None:
            # Fallback: use first identifier value
            identifiers = resource.get("identifier", [])
            if identifiers:
                external_id = identifiers[0].get("value")

        # Name
        family_name: str | None = None
        given_name: str | None = None
        names = resource.get("name", [])
        if names:
            family_name = names[0].get("family")
            given_parts = names[0].get("given", [])
            if given_parts:
                given_name = " ".join(given_parts)

        return cls(
            resource_id=resource_id,
            external_id=external_id or resource_id,
            family_name=family_name,
            given_name=given_name,
            birth_date=resource.get("birthDate"),
            gender=resource.get("gender"),
            raw=resource,
        )

    # -- conversions ------------------------------------------------------

    def to_internal(self, site_id: str = "fhir") -> dict[str, Any]:
        """Convert to the dict expected by the internal Patient model."""
        sex_map = {"male": "M", "female": "F", "other": "U", "unknown": "U"}
        age: int | None = None
        if self.birth_date:
            try:
                bd = datetime.strptime(self.birth_date, "%Y-%m-%d")
                today = datetime.now(timezone.utc)
                age = (
                    today.year
                    - bd.year
                    - ((today.month, today.day) < (bd.month, bd.day))
                )
            except ValueError:
                pass

        ext_id = self.external_id or self.resource_id
        from sepsis_vitals.security import compute_blind_index

        return {
            "external_id": ext_id,
            "external_id_hash": compute_blind_index(ext_id),
            "site_id": site_id,
            "age_years": age,
            "sex": sex_map.get(self.gender or "", "U"),
        }


# ---------------------------------------------------------------------------
# FHIRObservation -- inbound parsing
# ---------------------------------------------------------------------------


@dataclass
class FHIRObservation:
    """Parsed representation of a FHIR Observation resource for vital signs."""

    resource_id: str
    loinc_code: str
    internal_name: str
    value: float
    unit: str | None = None
    effective_datetime: str | None = None
    patient_reference: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_fhir(cls, resource: dict[str, Any]) -> FHIRObservation | None:
        """Parse a FHIR Observation resource.

        Returns ``None`` if the observation does not contain a recognised
        LOINC-coded vital sign.  Raises ``ValueError`` on structural issues.
        """
        if resource.get("resourceType") != "Observation":
            raise ValueError(
                f"Expected resourceType 'Observation', "
                f"got '{resource.get('resourceType')}'"
            )

        # Locate LOINC code
        loinc_code: str | None = None
        for coding in resource.get("code", {}).get("coding", []):
            if coding.get("system") == LOINC_SYSTEM:
                candidate = coding.get("code", "")
                if candidate in SUPPORTED_LOINC_CODES:
                    loinc_code = candidate
                    break

        if loinc_code is None:
            return None  # not a vital sign we track

        entry = LOINC_TO_ENTRY[loinc_code]

        # Extract value
        value: float | None = None
        source_unit: str | None = None

        vq = resource.get("valueQuantity")
        if vq is not None:
            value = vq.get("value")
            source_unit = vq.get("unit") or vq.get("code")
        else:
            # GCS may arrive as a plain valueInteger or valueDecimal
            for key in ("valueDecimal", "valueInteger"):
                if key in resource:
                    value = float(resource[key])
                    break

        if value is None:
            # Check for component-based value (e.g. BP components)
            for comp in resource.get("component", []):
                for coding in comp.get("code", {}).get("coding", []):
                    if coding.get("code") == loinc_code:
                        cvq = comp.get("valueQuantity", {})
                        value = cvq.get("value")
                        source_unit = cvq.get("unit") or cvq.get("code")
                        break
                if value is not None:
                    break

        if value is None:
            raise ValueError(
                f"Observation {resource.get('id', '?')} for LOINC "
                f"{loinc_code} has no extractable numeric value"
            )

        # Normalise units
        normalised = normalise_value(loinc_code, float(value), source_unit)

        # Patient reference
        patient_ref: str | None = None
        subj = resource.get("subject", {})
        ref_str = subj.get("reference", "")
        if ref_str:
            patient_ref = ref_str.split("/")[-1] if "/" in ref_str else ref_str

        return cls(
            resource_id=resource.get("id", _new_id()),
            loinc_code=loinc_code,
            internal_name=entry.internal_name,
            value=normalised,
            unit=entry.ucum_unit,
            effective_datetime=resource.get("effectiveDateTime"),
            patient_reference=patient_ref,
            raw=resource,
        )


# ---------------------------------------------------------------------------
# FHIRBundle -- inbound parsing
# ---------------------------------------------------------------------------


@dataclass
class FHIRBundle:
    """Parsed representation of a FHIR Bundle containing Patients and Observations."""

    patients: list[FHIRPatient] = field(default_factory=list)
    observations: list[FHIRObservation] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_fhir(cls, resource: dict[str, Any]) -> FHIRBundle:
        """Parse a FHIR Bundle.

        Iterates over ``entry`` elements and collects ``Patient`` and
        ``Observation`` resources.  Unknown resource types are silently
        skipped.
        """
        if resource.get("resourceType") != "Bundle":
            raise ValueError(
                f"Expected resourceType 'Bundle', "
                f"got '{resource.get('resourceType')}'"
            )

        patients: list[FHIRPatient] = []
        observations: list[FHIRObservation] = []

        for entry in resource.get("entry", []):
            res = entry.get("resource")
            if res is None:
                continue

            rt = res.get("resourceType")
            if rt == "Patient":
                patients.append(FHIRPatient.from_fhir(res))
            elif rt == "Observation":
                obs = FHIRObservation.from_fhir(res)
                if obs is not None:
                    observations.append(obs)

        return cls(patients=patients, observations=observations, raw=resource)


# ---------------------------------------------------------------------------
# Outbound serialisation helpers
# ---------------------------------------------------------------------------


def to_fhir_patient(internal: dict[str, Any]) -> dict[str, Any]:
    """Serialize an internal patient dict to a FHIR Patient resource.

    The *internal* dict is expected to carry the keys produced by
    ``db.Patient`` column names (``id``, ``external_id``, ``age_years``,
    ``sex``, ``site_id``, etc.).
    """
    sex_map = {"M": "male", "F": "female", "U": "unknown"}

    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": str(internal.get("id", _new_id())),
        "meta": {
            "profile": [
                "http://hl7.org/fhir/StructureDefinition/Patient"
            ],
        },
    }

    ext_id = internal.get("external_id")
    if ext_id:
        resource["identifier"] = [
            {
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                            "code": "MR",
                            "display": "Medical Record Number",
                        }
                    ]
                },
                "value": str(ext_id),
            }
        ]

    gender = sex_map.get(internal.get("sex", "U"), "unknown")
    resource["gender"] = gender

    age = internal.get("age_years")
    if age is not None:
        now = datetime.now(timezone.utc)
        approx_year = now.year - age
        resource["birthDate"] = f"{approx_year}-01-01"

    return resource


def to_fhir_observation(
    vital_name: str,
    value: float,
    patient_ref: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Serialize a single vital sign to a FHIR Observation resource.

    Parameters
    ----------
    vital_name : str
        Internal vital name (e.g. ``"heart_rate"``).
    value : float
        Numeric measurement value in canonical units.
    patient_ref : str
        FHIR-style patient reference, e.g. ``"Patient/abc-123"``.
    timestamp : str | None
        ISO-8601 effective date-time.  Defaults to *now* if omitted.
    """
    entry = INTERNAL_TO_ENTRY.get(vital_name)
    if entry is None:
        raise ValueError(f"Unknown internal vital name: {vital_name}")

    obs_id = _new_id()
    effective = timestamp or _now_iso()

    # Ensure patient_ref is in "Patient/<id>" form
    if not patient_ref.startswith("Patient/"):
        patient_ref = f"Patient/{patient_ref}"

    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_id,
        "meta": {
            "profile": [
                "http://hl7.org/fhir/StructureDefinition/vitalsigns"
            ],
        },
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": entry.code,
                    "display": entry.display,
                }
            ],
            "text": entry.display,
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": effective,
        "valueQuantity": {
            "value": value,
            "unit": entry.unit,
            "system": "http://unitsofmeasure.org",
            "code": entry.ucum_unit,
        },
    }

    return resource


def to_fhir_risk_assessment(
    prediction: dict[str, Any],
    patient_ref: str,
) -> dict[str, Any]:
    """Serialize a sepsis prediction dict to a FHIR RiskAssessment resource.

    Parameters
    ----------
    prediction : dict
        Output of ``ScoreBundle.as_dict()`` (or a superset with ML fields).
        Expected keys: ``risk_level``, ``qsofa``, ``sirs_count``,
        ``news2_style``, ``shock_index``, ``alert_flag``.
    patient_ref : str
        FHIR patient reference string.
    """
    ra_id = _new_id()

    if not patient_ref.startswith("Patient/"):
        patient_ref = f"Patient/{patient_ref}"

    risk_level = prediction.get("risk_level", "low")

    # Map internal risk level -> FHIR probability coding
    probability_map: dict[str, dict[str, str]] = {
        "low": {"code": "low", "display": "Low likelihood"},
        "moderate": {"code": "moderate", "display": "Moderate likelihood"},
        "high": {"code": "high", "display": "High likelihood"},
        "critical": {"code": "certain", "display": "Certain"},
    }
    prob_coding = probability_map.get(
        risk_level, {"code": "moderate", "display": "Moderate likelihood"}
    )

    # Qualitative risk probability
    risk_probability: float | None = prediction.get("risk_probability")
    probability_decimal = risk_probability if risk_probability is not None else {
        "low": 0.1,
        "moderate": 0.35,
        "high": 0.7,
        "critical": 0.95,
    }.get(risk_level, 0.5)

    outcome_text = (
        f"Sepsis risk: {risk_level.upper()}. "
        f"qSOFA={prediction.get('qsofa', 'N/A')}, "
        f"SIRS={prediction.get('sirs_count', 'N/A')}, "
        f"NEWS2={prediction.get('news2_style', 'N/A')}, "
        f"SI={prediction.get('shock_index', 'N/A')}"
    )

    resource: dict[str, Any] = {
        "resourceType": "RiskAssessment",
        "id": ra_id,
        "meta": {
            "profile": [
                "http://hl7.org/fhir/StructureDefinition/RiskAssessment"
            ],
        },
        "status": "final",
        "subject": {"reference": patient_ref},
        "occurrenceDateTime": _now_iso(),
        "condition": {
            "reference": "#sepsis-suspicion",
            "display": "Suspected Sepsis",
        },
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": SNOMED_SYSTEM,
                            "code": "91302008",
                            "display": "Sepsis",
                        }
                    ],
                    "text": outcome_text,
                },
                "probabilityDecimal": probability_decimal,
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": FHIR_RISK_PROBABILITY_SYSTEM,
                            **prob_coding,
                        }
                    ]
                },
            }
        ],
        "mitigation": (
            "Alert triggered -- immediate clinical review recommended."
            if prediction.get("alert_flag")
            else "Continue routine monitoring per protocol."
        ),
    }

    # Attach clinical scores as extensions
    extensions: list[dict[str, Any]] = []
    score_fields = [
        ("qsofa", "qSOFA Score"),
        ("sirs_count", "SIRS Criteria Met"),
        ("news2_style", "NEWS2-style Score"),
        ("shock_index", "Shock Index"),
        ("uva_style", "UVA-style Score"),
    ]
    for key, label in score_fields:
        val = prediction.get(key)
        if val is not None:
            extensions.append(
                {
                    "url": f"http://sepsis-vitals.example.org/fhir/StructureDefinition/{key}",
                    "valueDecimal": val,
                }
            )
    if extensions:
        resource["extension"] = extensions

    return resource


# ---------------------------------------------------------------------------
# vitals_from_observations -- aggregate observations for compute_scores()
# ---------------------------------------------------------------------------


def vitals_from_observations(
    observations: list[FHIRObservation],
) -> dict[str, float]:
    """Collapse a list of parsed observations into a flat vitals dict.

    When multiple observations exist for the same vital sign, the *last* one
    (by list order, assumed chronological) wins.
    """
    vitals: dict[str, float] = {}
    for obs in observations:
        vitals[obs.internal_name] = obs.value
    return vitals
