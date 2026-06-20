"""
sepsis_vitals.fhir.listener
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Background listener for automatic vital sign ingestion from hospital systems.

Supports:
- HL7v2 MLLP: Listens on a TCP socket for ORU^R01 observation messages
- FHIR R4 webhook: Accepts POST requests with FHIR Observation bundles

Eliminates the "manual entry problem" -- vitals flow directly from bedside
monitors and EHR systems without nurse double-documentation.

Both ingestion paths normalise observations into ``VitalsReading`` objects and
feed them through a single ``VitalsIngestionQueue``, keeping downstream
processing uniform regardless of wire format.

Usage::

    queue = VitalsIngestionQueue()
    queue.register_handler(my_callback)

    mllp = MLLPServer(port=2575, queue=queue)
    webhook = FHIRWebhookServer(port=8090, queue=queue)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(queue.process_loop())
        tg.create_task(mllp.start())
        tg.create_task(webhook.start())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# HL7v2 Parser
# ---------------------------------------------------------------------------


class HL7Parser:
    """Parse HL7v2 messages (specifically ORU^R01 observation results).

    HL7v2 messages use a pipe-delimited format with segments separated by
    carriage returns.  The MSH segment is special: MSH-1 is the field
    separator character (``|``) and MSH-2 contains encoding characters
    (typically ``^~\\&``).

    This parser handles the ORU^R01 message type, which carries clinical
    observation results from laboratory and bedside instruments.
    """

    SEGMENT_SEP = "\r"
    FIELD_SEP = "|"
    COMPONENT_SEP = "^"

    def parse(self, raw: str) -> HL7Message:
        """Parse a raw HL7v2 message string into an ``HL7Message``.

        Parameters
        ----------
        raw : str
            The raw HL7 message text.  Segment separators may be ``\\r``,
            ``\\n``, or ``\\r\\n``; all are accepted.

        Returns
        -------
        HL7Message
            Structured representation of the message.

        Raises
        ------
        ValueError
            If the message does not start with an ``MSH`` segment.
        """
        # Normalise line endings -- HL7 standard uses \r but TCP streams
        # and test fixtures may use \n or \r\n.
        normalised = raw.strip()
        normalised = normalised.replace("\r\n", "\r").replace("\n", "\r")

        raw_segments = normalised.split(self.SEGMENT_SEP)
        segments: List[List[str]] = []

        for raw_seg in raw_segments:
            raw_seg = raw_seg.strip()
            if not raw_seg:
                continue

            if raw_seg.startswith("MSH"):
                # MSH is special: MSH-1 *is* the field separator, so the
                # first ``|`` after ``MSH`` is the separator declaration,
                # not a field delimiter in the usual sense.  We prepend a
                # synthetic "MSH" element so field indexing stays consistent
                # with the HL7 spec (MSH-1 = "|", MSH-2 = encoding chars,
                # MSH-3 = sending application, etc.).
                parts = raw_seg.split(self.FIELD_SEP)
                # parts[0] is "MSH", parts[1] is encoding chars (MSH-2),
                # but MSH-1 (the pipe) is consumed by split.  Re-insert it.
                segments.append(
                    ["MSH", self.FIELD_SEP] + parts[1:]
                )
            else:
                segments.append(raw_seg.split(self.FIELD_SEP))

        if not segments or segments[0][0] != "MSH":
            raise ValueError(
                "Invalid HL7 message: does not start with MSH segment"
            )

        msg = HL7Message(segments=segments)

        # Extract message type from MSH-9 (e.g. "ORU^R01")
        msh = segments[0]
        if len(msh) > 9:
            msg.message_type = msh[9]  # MSH-9
            # MSH indices after our reconstruction:
            # 0=MSH, 1="|", 2=encoding chars, 3=sending app, ...
            # But in the standard split MSH-9 is at index 9.
            # Let's recalculate: after split on "|":
            #   parts = ["MSH", encoding_chars, sending_app, sending_fac,
            #            recv_app, recv_fac, datetime, security,
            #            msg_type, msg_control_id, ...]
            # After re-insert: ["MSH", "|", encoding_chars, sending_app, ...]
            # So MSH-9 (message type) is at index 9.

        # Extract timestamp from MSH-7
        if len(msh) > 7:
            msg.timestamp = self._parse_hl7_datetime(msh[7])

        # Extract patient ID from PID-3
        pid = msg.get_segment("PID")
        if pid and len(pid) > 3:
            # PID-3 may have components: ID^check_digit^...
            pid3 = pid[3]
            msg.patient_id = pid3.split(self.COMPONENT_SEP)[0]

        return msg

    def extract_vitals(self, msg: HL7Message) -> VitalsReading:
        """Extract vital signs from OBX segments of an ORU message.

        OBX segment format::

            OBX|set_id|value_type|observation_id^text^coding_system|sub_id|value|units|ref_range|abnormal_flags|probability|nature|status|...

        The observation_id field (OBX-3) contains the LOINC code as the
        first component.

        Parameters
        ----------
        msg : HL7Message
            A parsed HL7 message (should be ORU^R01).

        Returns
        -------
        VitalsReading
            Extracted vital signs with patient and timestamp metadata.
        """
        vitals: Dict[str, float] = {}

        for obx in msg.get_segments("OBX"):
            if len(obx) < 6:
                logger.debug(
                    "Skipping OBX segment with fewer than 6 fields: %s", obx
                )
                continue

            # OBX-2: value type (NM = numeric, ST = string, etc.)
            value_type = obx[2] if len(obx) > 2 else ""

            # OBX-3: observation identifier (LOINC code^display^coding system)
            obs_id_field = obx[3] if len(obx) > 3 else ""
            components = obs_id_field.split(self.COMPONENT_SEP)
            loinc_code = components[0] if components else ""
            display_text = components[1] if len(components) > 1 else ""

            # OBX-5: observation value
            raw_value = obx[5] if len(obx) > 5 else ""

            # OBX-11: observation result status (F=final, C=corrected,
            # P=preliminary, R=entered in error, etc.)
            # In the 0-indexed split array, OBX-11 is at index 11.
            # Treat empty/missing status as Final -- most bedside monitors
            # and interfaces only send final results.
            _ACCEPTED_STATUSES = {"F", "C", "R", ""}
            obs_status = obx[11].strip() if len(obx) > 11 else ""

            if obs_status not in _ACCEPTED_STATUSES:
                logger.debug(
                    "Skipping OBX with non-final status %r for %s",
                    obs_status, loinc_code,
                )
                continue

            # Resolve vital name from LOINC code
            vital_name = LOINC_VITAL_MAP.get(loinc_code)

            # Fall back to display text matching if LOINC code not found
            if vital_name is None and display_text:
                vital_name = DISPLAY_NAME_MAP.get(display_text.lower())

            if vital_name is None:
                logger.debug(
                    "Unrecognised observation: LOINC=%r, display=%r",
                    loinc_code, display_text,
                )
                continue

            # Parse numeric value
            if value_type == "NM" or not value_type:
                try:
                    numeric_value = float(raw_value)
                except (ValueError, TypeError):
                    logger.warning(
                        "Non-numeric value %r for %s (LOINC %s)",
                        raw_value, vital_name, loinc_code,
                    )
                    continue
            else:
                logger.debug(
                    "Skipping non-numeric OBX (type=%s) for %s",
                    value_type, vital_name,
                )
                continue

            vitals[vital_name] = numeric_value

        timestamp = msg.timestamp or datetime.now(timezone.utc).isoformat()

        return VitalsReading(
            patient_id=msg.patient_id or "unknown",
            timestamp=timestamp,
            vitals=vitals,
            source="hl7v2",
            raw_message_id=msg.message_control_id or None,
        )

    def build_ack(self, msg: HL7Message, ack_code: str = "AA") -> str:
        """Build an ACK message for the received HL7 message.

        Parameters
        ----------
        msg : HL7Message
            The original message being acknowledged.
        ack_code : str
            ``AA`` = application accept, ``AE`` = application error,
            ``AR`` = application reject.

        Returns
        -------
        str
            A properly formatted HL7 ACK message.
        """
        now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        ack_control_id = str(uuid.uuid4())[:8]

        # MSH: respond with our own sending app/facility, reference the
        # original message control ID in MSA.
        msh = self.FIELD_SEP.join([
            "MSH",
            "^~\\&",                      # MSH-2: encoding characters
            "SEPSIS_VITALS",              # MSH-3: sending application
            "SEPSIS_FACILITY",            # MSH-4: sending facility
            msg.sending_application,      # MSH-5: receiving application
            msg.sending_facility,         # MSH-6: receiving facility
            now,                          # MSH-7: date/time of message
            "",                           # MSH-8: security
            "ACK^R01",                    # MSH-9: message type
            ack_control_id,               # MSH-10: message control ID
            "P",                          # MSH-11: processing ID
            "2.5.1",                      # MSH-12: version ID
        ])

        # MSA: acknowledgment segment
        msa = self.FIELD_SEP.join([
            "MSA",
            ack_code,                     # MSA-1: acknowledgment code
            msg.message_control_id,       # MSA-2: message control ID being ACKed
        ])

        return f"{msh}\r{msa}"

    # -- private helpers ---------------------------------------------------

    @staticmethod
    def _parse_hl7_datetime(raw: str) -> str:
        """Convert an HL7 datetime string to ISO-8601.

        HL7 datetimes have the format ``YYYYMMDDHHMMSS[.S[S[S[S]]]][+/-ZZZZ]``.
        We parse what we can and return an ISO-8601 string.
        """
        if not raw:
            return datetime.now(timezone.utc).isoformat()

        # Strip timezone suffix for parsing; re-attach later
        tz_match = re.search(r"([+-]\d{4})$", raw)
        tz_suffix = ""
        date_part = raw
        if tz_match:
            tz_suffix = tz_match.group(1)
            date_part = raw[: tz_match.start()]

        # Remove fractional seconds for simpler parsing
        dot_idx = date_part.find(".")
        if dot_idx != -1:
            date_part = date_part[:dot_idx]

        # Try progressively shorter formats
        formats = [
            ("%Y%m%d%H%M%S", 14),
            ("%Y%m%d%H%M", 12),
            ("%Y%m%d%H", 10),
            ("%Y%m%d", 8),
        ]

        for fmt, expected_len in formats:
            if len(date_part) >= expected_len:
                try:
                    dt = datetime.strptime(
                        date_part[:expected_len], fmt
                    )
                    if tz_suffix:
                        # Convert "+0500" -> "+05:00"
                        tz_iso = f"{tz_suffix[:3]}:{tz_suffix[3:]}"
                        return dt.isoformat() + tz_iso
                    return dt.replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    continue

        # If all else fails, return the raw value
        logger.warning("Could not parse HL7 datetime: %r", raw)
        return raw


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


# ---------------------------------------------------------------------------
# Vitals Ingestion Queue
# ---------------------------------------------------------------------------


class VitalsIngestionQueue:
    """Async queue for incoming vitals readings.

    Provides a single fan-out point: readings arrive from any ingestion
    source (MLLP, FHIR webhook, manual entry) and are dispatched to all
    registered handler callbacks.

    The queue is bounded to prevent unbounded memory growth if downstream
    processing stalls.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._queue: asyncio.Queue[VitalsReading] = asyncio.Queue(
            maxsize=max_size
        )
        self._handlers: List[Callable] = []
        self._running = False
        self._stats = {
            "received": 0,
            "processed": 0,
            "errors": 0,
            "dropped": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of queue processing statistics."""
        return dict(self._stats)

    @property
    def pending(self) -> int:
        """Number of readings waiting to be processed."""
        return self._queue.qsize()

    def register_handler(self, handler: Callable) -> None:
        """Register a callback for new vitals readings.

        Handlers are invoked in registration order for each reading.
        A handler may be a coroutine function (``async def``) or a plain
        callable.

        Parameters
        ----------
        handler : callable
            Receives a single ``VitalsReading`` argument.
        """
        self._handlers.append(handler)
        logger.info(
            "Registered vitals handler: %s (total: %d)",
            getattr(handler, "__name__", repr(handler)),
            len(self._handlers),
        )

    async def put(self, reading: VitalsReading) -> None:
        """Add a reading to the queue.

        If the queue is full, the reading is dropped and a warning is
        logged.  This prevents backpressure from blocking the ingestion
        server, which must remain responsive to send HL7 ACKs.
        """
        try:
            self._queue.put_nowait(reading)
            self._stats["received"] += 1
            logger.debug(
                "Queued vitals for patient %s (%d vitals, source=%s)",
                reading.patient_id,
                len(reading.vitals),
                reading.source,
            )
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            logger.warning(
                "Vitals queue full (%d items) -- dropped reading for "
                "patient %s.  Consider increasing queue size or adding "
                "more processing capacity.",
                self._queue.maxsize,
                reading.patient_id,
            )

    async def process_loop(self) -> None:
        """Background loop that processes queued readings.

        Runs indefinitely, pulling readings from the queue and dispatching
        them to all registered handlers.  Errors in individual handlers are
        logged but do not stop the loop or affect other handlers.
        """
        self._running = True
        logger.info(
            "Vitals ingestion queue started (max_size=%d, handlers=%d)",
            self._queue.maxsize,
            len(self._handlers),
        )

        try:
            while self._running:
                try:
                    reading = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Periodic check of _running flag
                    continue

                await self._dispatch(reading)
                self._queue.task_done()
        except asyncio.CancelledError:
            logger.info("Vitals ingestion queue shutting down")
            # Drain remaining items
            await self._drain()
            raise

    async def stop(self) -> None:
        """Signal the process loop to stop after draining the queue."""
        self._running = False
        logger.info("Vitals ingestion queue stop requested")

    async def _dispatch(self, reading: VitalsReading) -> None:
        """Dispatch a reading to all registered handlers."""
        for handler in self._handlers:
            try:
                result = handler(reading)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                self._stats["errors"] += 1
                logger.exception(
                    "Error in vitals handler %s for patient %s",
                    getattr(handler, "__name__", repr(handler)),
                    reading.patient_id,
                )
            else:
                self._stats["processed"] += 1

    async def _drain(self) -> None:
        """Process any remaining items in the queue."""
        drained = 0
        while not self._queue.empty():
            try:
                reading = self._queue.get_nowait()
                await self._dispatch(reading)
                self._queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info("Drained %d remaining readings from queue", drained)


# ---------------------------------------------------------------------------
# MLLP Server (HL7v2 over TCP)
# ---------------------------------------------------------------------------


class MLLPServer:
    """Minimal Lower Layer Protocol server for HL7v2 message reception.

    MLLP is the standard transport for HL7v2 messages over TCP.  Each
    message is wrapped in framing bytes:

    - Start block: ``\\x0b`` (vertical tab / VT)
    - End block: ``\\x1c\\x0d`` (file separator + carriage return)

    The server listens for connections, extracts HL7 messages from the
    MLLP framing, parses ORU^R01 observations, and feeds extracted vitals
    into the ``VitalsIngestionQueue``.

    Parameters
    ----------
    host : str
        Bind address.  Defaults to ``"0.0.0.0"`` (all interfaces).
    port : int
        TCP port.  The HL7 MLLP default is 2575.
    queue : VitalsIngestionQueue | None
        Queue for extracted vitals.  If ``None``, a new queue is created.
    """

    START_BLOCK = b"\x0b"
    END_BLOCK = b"\x1c\x0d"

    # Maximum message size: 1 MB (HL7 messages rarely exceed a few KB,
    # but lab result batches can be larger).
    MAX_MESSAGE_SIZE = 1024 * 1024

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 2575,
        queue: Optional[VitalsIngestionQueue] = None,
        tls_cert: Optional[str] = None,
        tls_key: Optional[str] = None,
        tls_ca: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.queue = queue or VitalsIngestionQueue()
        self._parser = HL7Parser()
        self._server: Optional[asyncio.AbstractServer] = None
        self._active_connections: int = 0
        # TLS configuration — read from params or environment
        self._tls_cert = tls_cert or os.environ.get("MLLP_TLS_CERT")
        self._tls_key = tls_key or os.environ.get("MLLP_TLS_KEY")
        self._tls_ca = tls_ca or os.environ.get("MLLP_TLS_CA")
        self._stats = {
            "connections": 0,
            "messages_received": 0,
            "messages_accepted": 0,
            "messages_rejected": 0,
            "parse_errors": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of server statistics."""
        return dict(self._stats)

    async def start(self) -> None:
        """Start the MLLP server.

        This coroutine starts the TCP server and serves connections until
        ``stop()`` is called or the task is cancelled.  When TLS certificate
        and key paths are configured (via constructor args or MLLP_TLS_CERT /
        MLLP_TLS_KEY environment variables), the server wraps connections in
        TLS to satisfy HIPAA transit encryption requirements.
        """
        ssl_ctx = self._create_ssl_context()

        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port, ssl=ssl_ctx,
        )

        proto = "MLLP+TLS" if ssl_ctx else "MLLP (plaintext)"
        addrs = [
            str(sock.getsockname()) for sock in self._server.sockets
        ]
        logger.info(
            "%s server started on %s (HL7v2 ingestion active)", proto, addrs
        )
        if not ssl_ctx and os.environ.get("SEPSIS_ENV") == "production":
            logger.warning(
                "MLLP running WITHOUT TLS in production. "
                "Transmitting PHI over unencrypted TCP violates HIPAA "
                "Security Rule §164.312(e)(1). Set MLLP_TLS_CERT and "
                "MLLP_TLS_KEY or route traffic through a TLS tunnel."
            )

        async with self._server:
            await self._server.serve_forever()

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Build an SSL context if TLS cert/key are configured.

        Returns ``None`` when TLS is not configured, causing the server
        to fall back to plaintext TCP.
        """
        if not self._tls_cert or not self._tls_key:
            return None

        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(self._tls_cert, self._tls_key)

            if self._tls_ca:
                # Mutual TLS: require client certificate
                ctx.load_verify_locations(self._tls_ca)
                ctx.verify_mode = ssl.CERT_REQUIRED
                logger.info(
                    "MLLP TLS: mutual authentication enabled (CA: %s)",
                    self._tls_ca,
                )
            else:
                ctx.verify_mode = ssl.CERT_NONE

            logger.info(
                "MLLP TLS enabled (cert: %s, min version: TLSv1.2)",
                self._tls_cert,
            )
            return ctx
        except Exception:
            logger.exception("Failed to create MLLP TLS context")
            raise

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming MLLP connection.

        A single TCP connection may carry multiple HL7 messages (persistent
        connection).  Each message is individually ACKed.
        """
        peer = writer.get_extra_info("peername", ("unknown", 0))
        self._active_connections += 1
        self._stats["connections"] += 1
        logger.info(
            "MLLP connection from %s:%s (active: %d)",
            peer[0], peer[1], self._active_connections,
        )

        try:
            buffer = b""
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        reader.read(4096), timeout=300.0
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "MLLP connection from %s:%s timed out", peer[0], peer[1]
                    )
                    break

                if not chunk:
                    break  # Connection closed

                buffer += chunk

                # Process all complete messages in the buffer
                while True:
                    start_idx = buffer.find(self.START_BLOCK)
                    if start_idx == -1:
                        # No start block found; discard leading garbage
                        buffer = b""
                        break

                    end_idx = buffer.find(self.END_BLOCK, start_idx)
                    if end_idx == -1:
                        # Incomplete message; wait for more data
                        if len(buffer) > self.MAX_MESSAGE_SIZE:
                            logger.error(
                                "MLLP buffer exceeded %d bytes from %s:%s; "
                                "resetting buffer",
                                self.MAX_MESSAGE_SIZE, peer[0], peer[1],
                            )
                            buffer = b""
                        break

                    # Extract the HL7 message between framing bytes
                    msg_bytes = buffer[start_idx + 1 : end_idx]
                    buffer = buffer[end_idx + len(self.END_BLOCK) :]

                    self._stats["messages_received"] += 1

                    # Process the message and send ACK
                    ack = await self._process_message(msg_bytes, peer)
                    if ack:
                        # Wrap ACK in MLLP framing
                        framed_ack = (
                            self.START_BLOCK
                            + ack.encode("utf-8")
                            + self.END_BLOCK
                        )
                        writer.write(framed_ack)
                        await writer.drain()

        except ConnectionResetError:
            logger.info(
                "MLLP connection reset by %s:%s", peer[0], peer[1]
            )
        except Exception:
            logger.exception(
                "Unexpected error handling MLLP connection from %s:%s",
                peer[0], peer[1],
            )
        finally:
            self._active_connections -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(
                "MLLP connection from %s:%s closed (active: %d)",
                peer[0], peer[1], self._active_connections,
            )

    async def _process_message(
        self,
        msg_bytes: bytes,
        peer: tuple,
    ) -> Optional[str]:
        """Parse an HL7 message and queue extracted vitals.

        Returns the ACK message string, or ``None`` if no ACK should be
        sent.
        """
        try:
            raw = msg_bytes.decode("utf-8", errors="replace")
        except Exception:
            logger.error(
                "Failed to decode MLLP message from %s:%s",
                peer[0], peer[1],
            )
            self._stats["parse_errors"] += 1
            return None

        try:
            msg = self._parser.parse(raw)
        except ValueError as exc:
            logger.warning(
                "Failed to parse HL7 message from %s:%s: %s",
                peer[0], peer[1], exc,
            )
            self._stats["parse_errors"] += 1
            return None

        # Only process ORU (observation result) messages
        msg_type_code = msg.message_type.split(
            HL7Parser.COMPONENT_SEP
        )[0] if msg.message_type else ""

        if msg_type_code != "ORU":
            logger.info(
                "Ignoring non-ORU message type %r from %s:%s",
                msg.message_type, peer[0], peer[1],
            )
            self._stats["messages_rejected"] += 1
            # Still ACK it so the sender doesn't retry
            return self._parser.build_ack(msg, ack_code="AA")

        # Extract vitals from OBX segments
        try:
            reading = self._parser.extract_vitals(msg)
        except Exception:
            logger.exception(
                "Error extracting vitals from ORU message (control_id=%s)",
                msg.message_control_id,
            )
            self._stats["parse_errors"] += 1
            return self._parser.build_ack(msg, ack_code="AE")

        if not reading.vitals:
            logger.info(
                "ORU message %s contained no recognised vital signs",
                msg.message_control_id,
            )
            self._stats["messages_accepted"] += 1
            return self._parser.build_ack(msg, ack_code="AA")

        # Queue for downstream processing
        await self.queue.put(reading)
        self._stats["messages_accepted"] += 1

        logger.info(
            "Accepted ORU message %s: patient=%s, vitals=%s",
            msg.message_control_id,
            reading.patient_id,
            list(reading.vitals.keys()),
        )
        return self._parser.build_ack(msg, ack_code="AA")

    async def stop(self) -> None:
        """Stop the MLLP server gracefully.

        Closes the listening socket and waits for active connections to
        drain.
        """
        if self._server is not None:
            logger.info(
                "Stopping MLLP server (active connections: %d)",
                self._active_connections,
            )
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("MLLP server stopped")


# ---------------------------------------------------------------------------
# FHIR R4 Webhook Handler
# ---------------------------------------------------------------------------


class FHIRWebhookHandler:
    """Handles incoming FHIR R4 webhook POSTs.

    This handler processes FHIR Observation and Bundle resources received
    via HTTP webhook, extracts vital signs, and queues them for
    processing.

    It can be used in two ways:

    1. **Standalone** via ``FHIRWebhookServer`` -- runs its own HTTP server
       using ``asyncio`` and stdlib only (no framework needed).
    2. **Mounted** in a FastAPI/Starlette application -- call
       ``handle_observation()`` from your route handler.

    Parameters
    ----------
    queue : VitalsIngestionQueue | None
        Queue for extracted vitals.  If ``None``, a new queue is created.
    """

    def __init__(self, queue: Optional[VitalsIngestionQueue] = None) -> None:
        self.queue = queue or VitalsIngestionQueue()
        self._parser = FHIRObservationParser()
        self._stats = {
            "requests": 0,
            "observations_processed": 0,
            "bundles_processed": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of handler statistics."""
        return dict(self._stats)

    async def handle_observation(self, body: dict) -> dict:
        """Process a FHIR Observation or Bundle.  Returns acknowledgment.

        Automatically detects whether the input is a single Observation
        or a Bundle and dispatches accordingly.

        Parameters
        ----------
        body : dict
            Parsed JSON body of the incoming request.

        Returns
        -------
        dict
            Acknowledgment response suitable for JSON serialisation.
        """
        self._stats["requests"] += 1
        resource_type = body.get("resourceType", "")

        try:
            if resource_type == "Bundle":
                return await self.handle_bundle(body)
            elif resource_type == "Observation":
                return await self._handle_single_observation(body)
            else:
                self._stats["errors"] += 1
                return {
                    "status": "error",
                    "message": (
                        f"Unsupported resourceType: {resource_type!r}. "
                        f"Expected 'Observation' or 'Bundle'."
                    ),
                }
        except Exception as exc:
            self._stats["errors"] += 1
            logger.exception("Error processing FHIR webhook payload")
            return {
                "status": "error",
                "message": f"Internal processing error: {exc}",
            }

    async def handle_bundle(self, bundle: dict) -> dict:
        """Process a FHIR Bundle containing Observations.

        Parameters
        ----------
        bundle : dict
            A FHIR Bundle resource as a parsed JSON dict.

        Returns
        -------
        dict
            Acknowledgment with the count of processed readings.
        """
        readings = self._parser.parse_bundle(bundle)
        self._stats["bundles_processed"] += 1

        queued = 0
        total_vitals = 0
        for reading in readings:
            await self.queue.put(reading)
            queued += 1
            total_vitals += len(reading.vitals)

        self._stats["observations_processed"] += total_vitals

        logger.info(
            "FHIR webhook Bundle: %d readings queued (%d vitals)",
            queued, total_vitals,
        )
        return {
            "status": "accepted",
            "readings_queued": queued,
            "vitals_extracted": total_vitals,
            "bundle_id": bundle.get("id"),
        }

    async def _handle_single_observation(self, obs: dict) -> dict:
        """Process a single FHIR Observation resource."""
        patient_id = FHIRObservationParser._extract_patient_id(obs)
        vital = self._parser.parse_observation(obs, patient_id=patient_id)

        if vital is None:
            return {
                "status": "ignored",
                "message": "Observation does not contain a recognised vital sign.",
                "observation_id": obs.get("id"),
            }

        effective = obs.get(
            "effectiveDateTime",
            obs.get("effectiveInstant", ""),
        )
        reading = VitalsReading(
            patient_id=patient_id,
            timestamp=effective or datetime.now(timezone.utc).isoformat(),
            vitals=vital,
            source="fhir_r4",
            raw_message_id=obs.get("id"),
        )
        await self.queue.put(reading)
        self._stats["observations_processed"] += 1

        logger.info(
            "FHIR webhook Observation: patient=%s, vital=%s",
            patient_id, vital,
        )
        return {
            "status": "accepted",
            "readings_queued": 1,
            "vitals_extracted": len(vital),
            "observation_id": obs.get("id"),
        }


# ---------------------------------------------------------------------------
# Standalone FHIR Webhook HTTP Server (stdlib only)
# ---------------------------------------------------------------------------


class FHIRWebhookServer:
    """Minimal HTTP server for receiving FHIR webhook POSTs.

    Built entirely on ``asyncio`` -- no framework dependency.  Handles
    only ``POST`` requests to the webhook endpoint.  For production use
    with TLS, authentication, and routing, mount ``FHIRWebhookHandler``
    in a FastAPI application instead.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        TCP port.
    path : str
        URL path for the webhook endpoint.
    queue : VitalsIngestionQueue | None
        Queue for extracted vitals.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8090,
        path: str = "/fhir/webhook",
        queue: Optional[VitalsIngestionQueue] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.path = path
        self._handler = FHIRWebhookHandler(queue=queue)
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    def queue(self) -> VitalsIngestionQueue:
        return self._handler.queue

    async def start(self) -> None:
        """Start the HTTP webhook server."""
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )

        addrs = [
            str(sock.getsockname()) for sock in self._server.sockets
        ]
        logger.info(
            "FHIR webhook server started on %s (path: %s)", addrs, self.path
        )

        async with self._server:
            await self._server.serve_forever()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection."""
        peer = writer.get_extra_info("peername", ("unknown", 0))

        try:
            # Read request line and headers
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=30.0
            )
            if not request_line:
                return

            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ")
            if len(parts) < 2:
                await self._send_response(
                    writer, 400, {"error": "Malformed request"}
                )
                return

            method = parts[0]
            path = parts[1]

            # Read headers
            headers: Dict[str, str] = {}
            while True:
                header_line = await asyncio.wait_for(
                    reader.readline(), timeout=10.0
                )
                decoded = header_line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    break
                if ":" in decoded:
                    key, value = decoded.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            # Route the request
            if method == "POST" and path == self.path:
                await self._handle_post(reader, writer, headers, peer)
            elif method == "GET" and path == "/health":
                await self._send_response(
                    writer, 200, {"status": "ok", "service": "fhir-webhook"}
                )
            else:
                await self._send_response(
                    writer, 404, {"error": f"Not found: {method} {path}"}
                )

        except asyncio.TimeoutError:
            logger.debug(
                "HTTP connection from %s:%s timed out", peer[0], peer[1]
            )
        except ConnectionResetError:
            pass
        except Exception:
            logger.exception(
                "Error handling HTTP connection from %s:%s", peer[0], peer[1]
            )
            try:
                await self._send_response(
                    writer, 500, {"error": "Internal server error"}
                )
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_post(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: Dict[str, str],
        peer: tuple,
    ) -> None:
        """Handle a POST request to the webhook endpoint."""
        content_length_str = headers.get("content-length", "0")
        try:
            content_length = int(content_length_str)
        except ValueError:
            await self._send_response(
                writer, 400, {"error": "Invalid Content-Length header"}
            )
            return

        if content_length > 10 * 1024 * 1024:  # 10 MB limit
            await self._send_response(
                writer, 413, {"error": "Payload too large (max 10 MB)"}
            )
            return

        if content_length > 0:
            body_bytes = await asyncio.wait_for(
                reader.readexactly(content_length), timeout=30.0
            )
        else:
            # Try to read available data if content-length missing
            body_bytes = await asyncio.wait_for(
                reader.read(1024 * 1024), timeout=5.0
            )

        if not body_bytes:
            await self._send_response(
                writer, 400, {"error": "Empty request body"}
            )
            return

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            await self._send_response(
                writer, 400, {"error": f"Invalid JSON: {exc}"}
            )
            return

        result = await self._handler.handle_observation(body)

        status_code = 202 if result.get("status") == "accepted" else 400
        await self._send_response(writer, status_code, result)

    @staticmethod
    async def _send_response(
        writer: asyncio.StreamWriter,
        status_code: int,
        body: dict,
    ) -> None:
        """Send an HTTP response with JSON body."""
        status_messages = {
            200: "OK",
            202: "Accepted",
            400: "Bad Request",
            404: "Not Found",
            413: "Payload Too Large",
            500: "Internal Server Error",
        }
        status_text = status_messages.get(status_code, "Unknown")
        body_bytes = json.dumps(body).encode("utf-8")

        response = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(response.encode("utf-8") + body_bytes)
        await writer.drain()

    async def stop(self) -> None:
        """Stop the HTTP webhook server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("FHIR webhook server stopped")
