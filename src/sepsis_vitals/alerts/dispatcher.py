"""
sepsis_vitals.alerts.dispatcher — Central alert dispatch engine.

Discovers available notification channels (SMS, Web Push, WebSocket)
at startup and fans out every alert to the contacts registered for it.

Dispatch is non-blocking: alerts are sent in a background thread so
the API endpoint that triggered the alert returns immediately.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel constants
# ---------------------------------------------------------------------------

CHANNEL_SMS = "sms"
CHANNEL_PUSH = "push"
CHANNEL_WEBSOCKET = "websocket"
VALID_CHANNELS = frozenset({CHANNEL_SMS, CHANNEL_PUSH, CHANNEL_WEBSOCKET})


# ---------------------------------------------------------------------------
# Contact record
# ---------------------------------------------------------------------------


class Contact:
    """A notification destination for a specific user."""

    __slots__ = ("id", "user_id", "channel", "destination", "created_at")

    def __init__(
        self, user_id: str, channel: str, destination: str
    ) -> None:
        if channel not in VALID_CHANNELS:
            raise ValueError(
                f"Invalid channel {channel!r}. "
                f"Must be one of {sorted(VALID_CHANNELS)}."
            )
        self.id: str = uuid.uuid4().hex
        self.user_id = user_id
        self.channel = channel
        self.destination = destination
        self.created_at: datetime = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "channel": self.channel,
            "destination": self.destination,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Delivery record (kept for /alerts/history)
# ---------------------------------------------------------------------------


class DeliveryRecord:
    """Immutable record of a single delivery attempt."""

    __slots__ = (
        "id",
        "alert_id",
        "channel",
        "destination",
        "status",
        "error",
        "timestamp",
    )

    def __init__(
        self,
        alert_id: str,
        channel: str,
        destination: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        self.id: str = uuid.uuid4().hex
        self.alert_id = alert_id
        self.channel = channel
        self.destination = destination
        self.status = status
        self.error = error
        self.timestamp: datetime = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "alert_id": self.alert_id,
            "channel": self.channel,
            "destination": self.destination,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.error:
            result["error"] = self.error
        return result


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------


def _format_sms(alert: dict[str, Any]) -> str:
    """Compact text for SMS (aim for <160 chars)."""
    risk = alert.get("risk_level", "unknown").upper()
    patient = alert.get("patient_id", "?")
    prob = alert.get("risk_probability")
    rec = alert.get("recommendation", "Assess patient now.")
    prob_str = f" ({prob:.0%})" if prob is not None else ""
    return (
        f"SEPSIS ALERT [{risk}]\n"
        f"Pt: {patient}{prob_str}\n"
        f"{rec[:80]}\n"
        f"Respond immediately."
    )


def _format_push(alert: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Return (title, body, data) for a web-push notification."""
    risk = alert.get("risk_level", "unknown").upper()
    patient = alert.get("patient_id", "?")
    prob = alert.get("risk_probability")
    rec = alert.get("recommendation", "Assess patient now.")
    title = f"Sepsis Alert [{risk}]"
    body = f"Patient {patient}"
    if prob is not None:
        body += f" — {prob:.0%} risk"
    body += f". {rec}"
    data = {
        "patient_id": patient,
        "risk_level": alert.get("risk_level", "unknown"),
        "alert_id": alert.get("id", ""),
    }
    return title, body, data


def _format_ws(alert: dict[str, Any]) -> dict[str, Any]:
    """Full JSON payload for WebSocket clients."""
    return {
        "type": "sepsis_alert",
        "patient_id": alert.get("patient_id"),
        "risk_level": alert.get("risk_level"),
        "risk_probability": alert.get("risk_probability"),
        "recommendation": alert.get("recommendation"),
        "alert_id": alert.get("id"),
        "timestamp": alert.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Maximum delivery records kept in memory.
_MAX_HISTORY = 500


class AlertDispatcher:
    """Fan-out alert delivery across all configured channels.

    The dispatcher auto-discovers which channels are available at init
    time by checking for configured providers / services.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._contacts: dict[str, Contact] = {}  # id -> Contact
        self._history: list[DeliveryRecord] = []

        # --- auto-discover channels ---
        self._sms_provider: Any = None
        self._push_service: Any = None

        try:
            from sepsis_vitals.alerts.sms import get_sms_provider

            self._sms_provider = get_sms_provider()
        except Exception as exc:  # pragma: no cover
            logger.debug("SMS provider not available: %s", exc)

        try:
            from sepsis_vitals.alerts.push import PushNotificationService

            self._push_service = PushNotificationService()
        except Exception as exc:
            logger.debug("Push service not available: %s", exc)

    # ------------------------------------------------------------------
    # Channel availability
    # ------------------------------------------------------------------

    @property
    def available_channels(self) -> list[str]:
        channels: list[str] = []
        if self._sms_provider is not None:
            channels.append(CHANNEL_SMS)
        if self._push_service is not None:
            channels.append(CHANNEL_PUSH)
        # WebSocket is always available (in-process).
        channels.append(CHANNEL_WEBSOCKET)
        return channels

    # ------------------------------------------------------------------
    # Contact management
    # ------------------------------------------------------------------

    def register_contact(
        self, user_id: str, channel: str, destination: str
    ) -> Contact:
        """Add a notification contact.  Returns the created ``Contact``."""
        contact = Contact(user_id, channel, destination)
        with self._lock:
            self._contacts[contact.id] = contact
        logger.info(
            "Contact registered id=%s user=%s channel=%s",
            contact.id,
            user_id,
            channel,
        )
        return contact

    def remove_contact(self, contact_id: str) -> bool:
        """Remove a contact by id.  Returns ``True`` if found."""
        with self._lock:
            removed = self._contacts.pop(contact_id, None)
        if removed:
            logger.info("Contact removed id=%s", contact_id)
        return removed is not None

    def list_contacts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._contacts.values()]

    # ------------------------------------------------------------------
    # Delivery history
    # ------------------------------------------------------------------

    def _record(self, rec: DeliveryRecord) -> None:
        with self._lock:
            self._history.append(rec)
            # Trim oldest entries if we exceed the cap.
            if len(self._history) > _MAX_HISTORY:
                self._history = self._history[-_MAX_HISTORY:]

    def get_history(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        with self._lock:
            # Most recent first.
            ordered = list(reversed(self._history))
        return [r.to_dict() for r in ordered[offset : offset + limit]]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch_alert(self, alert: dict[str, Any]) -> None:
        """Fan out *alert* to every registered contact.

        Delivery happens in a daemon thread so the caller is never
        blocked by slow network I/O (e.g. SMS gateway latency).
        """
        thread = threading.Thread(
            target=self._dispatch_sync,
            args=(alert,),
            daemon=True,
        )
        thread.start()

    def _dispatch_sync(self, alert: dict[str, Any]) -> None:
        """Synchronous dispatch executed inside a background thread."""
        alert_id = alert.get("id", uuid.uuid4().hex)

        with self._lock:
            contacts = list(self._contacts.values())

        # --- SMS ---
        sms_contacts = [c for c in contacts if c.channel == CHANNEL_SMS]
        if sms_contacts and self._sms_provider is None:
            logger.warning(
                "SMS alert for %s dropped — %d SMS contacts registered but no SMS provider "
                "configured (set TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM_NUMBER)",
                alert.get("patient_id", "unknown"),
                len(sms_contacts),
            )
            for contact in sms_contacts:
                self._record(
                    DeliveryRecord(
                        alert_id=alert_id,
                        channel=CHANNEL_SMS,
                        destination=contact.destination,
                        status="failed",
                        error="SMS provider not configured",
                    )
                )
        if sms_contacts and self._sms_provider is not None:
            sms_body = _format_sms(alert)
            for contact in sms_contacts:
                result = self._sms_provider.send(contact.destination, sms_body)
                status = result.get("status", "unknown")
                error = result.get("error")
                self._record(
                    DeliveryRecord(
                        alert_id=alert_id,
                        channel=CHANNEL_SMS,
                        destination=contact.destination,
                        status=status,
                        error=error,
                    )
                )

        # --- Web Push ---
        push_contacts = [c for c in contacts if c.channel == CHANNEL_PUSH]
        if push_contacts and self._push_service is None:
            logger.warning(
                "Push alert for %s dropped — push contacts registered but VAPID_PRIVATE_KEY "
                "not configured",
                alert.get("patient_id", "unknown"),
            )
        if self._push_service is not None:
            push_title, push_body, push_data = _format_push(alert)
            patient_id = alert.get("patient_id", "")
            risk_level = alert.get("risk_level", "unknown")
            summary = self._push_service.broadcast_alert(
                patient_id, risk_level, push_body
            )
            self._record(
                DeliveryRecord(
                    alert_id=alert_id,
                    channel=CHANNEL_PUSH,
                    destination="broadcast",
                    status="sent" if summary["sent"] > 0 else "no_subscribers",
                )
            )

        # --- WebSocket ---
        ws_payload = _format_ws(alert)
        try:
            from sepsis_vitals.realtime.websocket import manager

            # The ConnectionManager.broadcast() is async, so we need a loop.
            loop: asyncio.AbstractEventLoop | None = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(ws_payload), loop
                )
            else:
                asyncio.run(manager.broadcast(ws_payload))

            self._record(
                DeliveryRecord(
                    alert_id=alert_id,
                    channel=CHANNEL_WEBSOCKET,
                    destination="broadcast",
                    status="sent",
                )
            )
        except Exception as exc:
            logger.error("WebSocket broadcast failed: %s", exc)
            self._record(
                DeliveryRecord(
                    alert_id=alert_id,
                    channel=CHANNEL_WEBSOCKET,
                    destination="broadcast",
                    status="failed",
                    error=str(exc),
                )
            )

        logger.info(
            "Alert dispatched alert_id=%s channels=%d contacts=%d",
            alert_id,
            len(self.available_channels),
            len(contacts),
        )

    # ------------------------------------------------------------------
    # Test helper
    # ------------------------------------------------------------------

    def send_test_alert(
        self, channel: str, destination: str
    ) -> dict[str, Any]:
        """Send a single test notification to verify channel connectivity.

        Returns the raw provider response dict.
        """
        if channel == CHANNEL_SMS:
            if self._sms_provider is None:
                return {"status": "unavailable", "error": "No SMS provider configured."}
            result = self._sms_provider.send(
                destination,
                "Sepsis Vitals test alert. If you received this, SMS delivery is working.",
            )
            self._record(
                DeliveryRecord(
                    alert_id="test",
                    channel=CHANNEL_SMS,
                    destination=destination,
                    status=result.get("status", "unknown"),
                    error=result.get("error"),
                )
            )
            return result

        if channel == CHANNEL_PUSH:
            if self._push_service is None:
                return {"status": "unavailable", "error": "Push service not configured."}
            # For push, destination is not used — we broadcast to all.
            summary = self._push_service.broadcast_alert(
                patient_id="TEST",
                risk_level="test",
                message="Sepsis Vitals test push notification. Delivery confirmed.",
            )
            self._record(
                DeliveryRecord(
                    alert_id="test",
                    channel=CHANNEL_PUSH,
                    destination="broadcast",
                    status="sent" if summary["sent"] > 0 else "no_subscribers",
                )
            )
            return summary

        if channel == CHANNEL_WEBSOCKET:
            try:
                from sepsis_vitals.realtime.websocket import manager

                payload = {
                    "type": "test_alert",
                    "message": "Sepsis Vitals test WebSocket notification.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                loop: asyncio.AbstractEventLoop | None = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast(payload), loop
                    )
                else:
                    asyncio.run(manager.broadcast(payload))

                result_dict: dict[str, Any] = {
                    "status": "sent",
                    "connections": manager.active_connections,
                }
                self._record(
                    DeliveryRecord(
                        alert_id="test",
                        channel=CHANNEL_WEBSOCKET,
                        destination="broadcast",
                        status="sent",
                    )
                )
                return result_dict
            except Exception as exc:
                return {"status": "failed", "error": str(exc)}

        return {"status": "error", "error": f"Unknown channel {channel!r}."}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_dispatcher: AlertDispatcher | None = None
_init_lock = threading.Lock()


def get_dispatcher() -> AlertDispatcher:
    """Return (or create) the module-level dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        with _init_lock:
            if _dispatcher is None:
                _dispatcher = AlertDispatcher()
    return _dispatcher
