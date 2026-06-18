"""
sepsis_vitals.alerts.push — Web Push notification delivery.

Uses the Web Push protocol (RFC 8030) via *pywebpush* so that nurses
who have the PWA installed on their phones receive instant browser
notifications even when the app tab is in the background.

Subscriptions are held in memory for simplicity.  A production
deployment should persist them to the database.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PushNotificationService:
    """Manage Web Push subscriptions and deliver notifications.

    Parameters
    ----------
    vapid_private_key:
        Base-64-encoded VAPID private key (or path to PEM file).
        Defaults to ``VAPID_PRIVATE_KEY`` env var.
    vapid_email:
        Contact e-mail included in the VAPID claim.
        Defaults to ``VAPID_EMAIL`` env var.
    """

    def __init__(
        self,
        vapid_private_key: Optional[str] = None,
        vapid_email: Optional[str] = None,
    ) -> None:
        self.vapid_private_key = (
            vapid_private_key or os.environ["VAPID_PRIVATE_KEY"]
        )
        self.vapid_email = vapid_email or os.environ["VAPID_EMAIL"]
        self._vapid_claims = {
            "sub": f"mailto:{self.vapid_email}",
        }
        # Thread-safe subscription store.
        self._lock = threading.Lock()
        self._subscriptions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, subscription_info: dict[str, Any]) -> bool:
        """Store a push subscription.

        *subscription_info* must contain at minimum an ``endpoint`` key
        and a ``keys`` dict with ``p256dh`` and ``auth`` values (the
        standard PushSubscription JSON shape from the browser API).

        Returns ``True`` if the subscription was added, ``False`` if it
        was already registered (idempotent).
        """
        endpoint = subscription_info.get("endpoint")
        if not endpoint:
            raise ValueError("subscription_info must contain an 'endpoint' key")
        keys = subscription_info.get("keys", {})
        if "p256dh" not in keys or "auth" not in keys:
            raise ValueError(
                "subscription_info.keys must contain 'p256dh' and 'auth'"
            )

        with self._lock:
            # Deduplicate by endpoint.
            for existing in self._subscriptions:
                if existing.get("endpoint") == endpoint:
                    return False
            self._subscriptions.append(subscription_info)
        logger.info("Push subscription added endpoint=%s", endpoint[:60])
        return True

    def unsubscribe(self, endpoint: str) -> bool:
        """Remove a subscription by endpoint URL.

        Returns ``True`` if a matching subscription was removed.
        """
        with self._lock:
            before = len(self._subscriptions)
            self._subscriptions = [
                s for s in self._subscriptions if s.get("endpoint") != endpoint
            ]
            removed = len(self._subscriptions) < before
        if removed:
            logger.info("Push subscription removed endpoint=%s", endpoint[:60])
        return removed

    @property
    def subscription_count(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def send_notification(
        self,
        subscription: dict[str, Any],
        title: str,
        body: str,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send a single push notification.

        Returns a dict with ``status`` (``"sent"`` or ``"failed"``) and
        optional ``error`` detail.
        """
        from pywebpush import webpush, WebPushException  # type: ignore[import-untyped]

        payload = json.dumps(
            {
                "title": title,
                "body": body,
                "data": data or {},
            }
        )
        try:
            response = webpush(
                subscription_info=subscription,
                data=payload,
                vapid_private_key=self.vapid_private_key,
                vapid_claims=self._vapid_claims,
            )
            logger.info(
                "Push notification sent endpoint=%s status=%s",
                subscription.get("endpoint", "")[:60],
                response.status_code,
            )
            return {"status": "sent", "status_code": response.status_code}
        except WebPushException as exc:
            logger.error("Push notification failed: %s", exc)
            # If the subscription is expired / invalid, remove it.
            if hasattr(exc, "response") and exc.response is not None:
                status_code = exc.response.status_code
                if status_code in (404, 410):
                    self.unsubscribe(subscription.get("endpoint", ""))
                    return {
                        "status": "expired",
                        "error": "Subscription no longer valid; removed.",
                    }
            return {"status": "failed", "error": str(exc)}
        except Exception as exc:
            logger.error("Push notification unexpected error: %s", exc)
            return {"status": "failed", "error": str(exc)}

    def broadcast_alert(
        self,
        patient_id: str,
        risk_level: str,
        message: str,
    ) -> dict[str, Any]:
        """Send a sepsis alert to every registered push subscription.

        Returns a summary dict with counts of sent / failed deliveries.
        """
        with self._lock:
            targets = list(self._subscriptions)

        if not targets:
            logger.info("Push broadcast skipped — no subscriptions registered.")
            return {"sent": 0, "failed": 0, "total": 0}

        title = f"Sepsis Alert [{risk_level.upper()}]"
        data = {"patient_id": patient_id, "risk_level": risk_level}

        sent = 0
        failed = 0
        for sub in targets:
            result = self.send_notification(sub, title, message, data)
            if result["status"] == "sent":
                sent += 1
            else:
                failed += 1

        summary = {"sent": sent, "failed": failed, "total": len(targets)}
        logger.info("Push broadcast complete: %s", summary)
        return summary
