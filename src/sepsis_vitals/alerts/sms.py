"""
sepsis_vitals.alerts.sms — SMS delivery providers for sepsis alerts.

Supports Twilio and Africa's Talking, the two most widely available
SMS gateways in Sub-Saharan Africa and South Asia.  Provider selection
is automatic based on which environment variables are configured.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# E.164 phone-number validation
# ---------------------------------------------------------------------------

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def _validate_e164(number: str) -> str:
    """Return *number* if it matches E.164 format, otherwise raise."""
    number = number.strip()
    if not _E164_RE.match(number):
        raise ValueError(
            f"Phone number {number!r} is not valid E.164 format "
            "(must start with '+' followed by country code and digits)."
        )
    return number


# ---------------------------------------------------------------------------
# SMS alert message formatter
# ---------------------------------------------------------------------------

_ALERT_TEMPLATE = (
    "SEPSIS ALERT [{risk_level}]\n"
    "Patient: {patient_id}\n"
    "Risk: {risk_probability:.0%}\n"
    "Action: {recommendation}\n"
    "Do NOT ignore. Assess patient now."
)


def _format_alert(
    patient_id: str,
    risk_level: str,
    risk_probability: float,
    recommendation: str,
) -> str:
    """Build a short, clear SMS alert body (stays under 160 chars when possible)."""
    return _ALERT_TEMPLATE.format(
        risk_level=risk_level.upper(),
        patient_id=patient_id,
        risk_probability=risk_probability,
        recommendation=recommendation[:80],
    )


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class SMSProvider(Protocol):
    """Structural interface every SMS backend must satisfy."""

    def send(self, to_number: str, message: str) -> dict[str, Any]: ...

    def send_sepsis_alert(
        self,
        to_number: str,
        patient_id: str,
        risk_level: str,
        risk_probability: float,
        recommendation: str,
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Twilio
# ---------------------------------------------------------------------------


class TwilioSMS:
    """SMS delivery via Twilio REST API.

    Requires environment variables:
    - ``TWILIO_ACCOUNT_SID``
    - ``TWILIO_AUTH_TOKEN``
    - ``TWILIO_FROM_NUMBER`` (E.164 format)
    """

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
    ) -> None:
        self.account_sid = account_sid or os.environ["TWILIO_ACCOUNT_SID"]
        self.auth_token = auth_token or os.environ["TWILIO_AUTH_TOKEN"]
        self.from_number = _validate_e164(
            from_number or os.environ["TWILIO_FROM_NUMBER"]
        )
        self._client: Any = None

    # Lazy import keeps ``twilio`` optional at install time.
    def _get_client(self) -> Any:
        if self._client is None:
            from twilio.rest import Client  # type: ignore[import-untyped]

            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    def send(self, to_number: str, message: str) -> dict[str, Any]:
        """Send a plain SMS.  Returns ``{"sid": ..., "status": ...}``."""
        to_number = _validate_e164(to_number)
        try:
            client = self._get_client()
            msg = client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number,
            )
            logger.info("Twilio SMS sent sid=%s to=%s", msg.sid, to_number)
            return {"sid": msg.sid, "status": msg.status}
        except Exception as exc:
            logger.error("Twilio SMS failed to=%s: %s", to_number, exc)
            return {"sid": None, "status": "failed", "error": str(exc)}

    def send_sepsis_alert(
        self,
        to_number: str,
        patient_id: str,
        risk_level: str,
        risk_probability: float,
        recommendation: str,
    ) -> dict[str, Any]:
        """Format and send a sepsis alert SMS."""
        body = _format_alert(patient_id, risk_level, risk_probability, recommendation)
        return self.send(to_number, body)


# ---------------------------------------------------------------------------
# Africa's Talking
# ---------------------------------------------------------------------------


class AfricasTalkingSMS:
    """SMS delivery via Africa's Talking gateway.

    Popular across Sub-Saharan Africa with competitive per-SMS rates.

    Requires environment variables:
    - ``AT_USERNAME``
    - ``AT_API_KEY``
    - ``AT_SENDER_ID`` (alphanumeric sender ID or shortcode)
    """

    def __init__(
        self,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
        sender_id: Optional[str] = None,
    ) -> None:
        self.username = username or os.environ["AT_USERNAME"]
        self.api_key = api_key or os.environ["AT_API_KEY"]
        self.sender_id = sender_id or os.environ.get("AT_SENDER_ID", "")
        self._sms: Any = None

    def _get_sms_service(self) -> Any:
        if self._sms is None:
            import africastalking  # type: ignore[import-untyped]

            africastalking.initialize(self.username, self.api_key)
            self._sms = africastalking.SMS
        return self._sms

    def send(self, to_number: str, message: str) -> dict[str, Any]:
        """Send a plain SMS.  Returns provider response dict."""
        to_number = _validate_e164(to_number)
        try:
            sms = self._get_sms_service()
            kwargs: dict[str, Any] = {
                "message": message,
                "recipients": [to_number],
            }
            if self.sender_id:
                kwargs["sender_id"] = self.sender_id
            response = sms.send(**kwargs)
            logger.info(
                "AfricasTalking SMS sent to=%s status=%s",
                to_number,
                response,
            )
            # Normalise response to match Twilio shape for callers.
            recipients = response.get("SMSMessageData", {}).get("Recipients", [])
            if recipients:
                r = recipients[0]
                return {
                    "sid": r.get("messageId"),
                    "status": r.get("status", "unknown"),
                    "cost": r.get("cost"),
                }
            return {"sid": None, "status": "sent", "raw": response}
        except Exception as exc:
            logger.error("AfricasTalking SMS failed to=%s: %s", to_number, exc)
            return {"sid": None, "status": "failed", "error": str(exc)}

    def send_sepsis_alert(
        self,
        to_number: str,
        patient_id: str,
        risk_level: str,
        risk_probability: float,
        recommendation: str,
    ) -> dict[str, Any]:
        """Format and send a sepsis alert SMS."""
        body = _format_alert(patient_id, risk_level, risk_probability, recommendation)
        return self.send(to_number, body)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_sms_provider() -> Optional[SMSProvider]:
    """Return whichever SMS provider has its env vars configured, or ``None``.

    Twilio is preferred when both providers are configured because it has
    wider global coverage (useful for South-Asian deployments).
    """
    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
        try:
            return TwilioSMS()
        except (ValueError, KeyError) as exc:
            logger.warning("Twilio env vars present but invalid: %s", exc)

    if os.getenv("AT_USERNAME") and os.getenv("AT_API_KEY"):
        try:
            return AfricasTalkingSMS()
        except (ValueError, KeyError) as exc:
            logger.warning("AfricasTalking env vars present but invalid: %s", exc)

    logger.info("No SMS provider configured (set TWILIO_* or AT_* env vars).")
    return None
