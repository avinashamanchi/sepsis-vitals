"""
sepsis_vitals.fhir.hl7

HL7v2 ORU^R01 parsing into VitalsReading objects.
Split out of sepsis_vitals.fhir.listener, which re-exports every name.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List
from sepsis_vitals.fhir.ingest_models import DISPLAY_NAME_MAP, HL7Message, LOINC_VITAL_MAP, VitalsReading

# Keep the original logger name so existing log routing is unchanged.
logger = logging.getLogger("sepsis_vitals.fhir.listener")

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


