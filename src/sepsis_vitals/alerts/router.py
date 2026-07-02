"""
sepsis_vitals.alerts.router — FastAPI router for alert notification management.

All endpoints are mounted under the ``/alerts`` prefix.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from sepsis_vitals.alerts.dispatcher import (
    VALID_CHANNELS,
    get_dispatcher,
)
from sepsis_vitals.alerts.escalation import get_escalation_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


async def _require_auth(request: Request):
    """Require authentication for alerts endpoints."""
    from sepsis_vitals.api import verify_auth
    return await verify_auth(request)


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class ContactCreate(BaseModel):
    """Register a notification contact."""

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User or staff identifier.",
    )
    channel: str = Field(
        ...,
        description="Notification channel: sms, push, or websocket.",
    )
    destination: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description=(
            "Channel-specific address.  E.164 phone number for SMS, "
            "'browser' for push, 'ws' for websocket."
        ),
    )


class ContactResponse(BaseModel):
    id: str
    user_id: str
    channel: str
    destination: str
    created_at: str


class ContactListResponse(BaseModel):
    contacts: list[ContactResponse]
    count: int


class TestAlertRequest(BaseModel):
    channel: str = Field(
        ...,
        description="Channel to test: sms, push, or websocket.",
    )
    destination: str = Field(
        "",
        max_length=256,
        description="Destination for the test (e.g. phone number for SMS).",
    )


class TestAlertResponse(BaseModel):
    channel: str
    result: dict[str, Any]


class DeliveryHistoryItem(BaseModel):
    id: str
    alert_id: str
    channel: str
    destination: str
    status: str
    error: Optional[str] = None
    timestamp: str


class DeliveryHistoryResponse(BaseModel):
    history: list[DeliveryHistoryItem]
    count: int


class PushSubscriptionKeys(BaseModel):
    p256dh: str = Field(..., min_length=1)
    auth: str = Field(..., min_length=1)


class PushSubscriptionRequest(BaseModel):
    endpoint: str = Field(..., min_length=1, description="Push service endpoint URL.")
    keys: PushSubscriptionKeys


class PushSubscriptionResponse(BaseModel):
    registered: bool
    endpoint: str
    subscription_count: int


class ChannelsResponse(BaseModel):
    available: list[str]


class MessageResponse(BaseModel):
    detail: str


class SnoozeRequest(BaseModel):
    """Body for the snooze endpoint."""

    minutes: int = Field(15, ge=1, le=120, description="Minutes to snooze.")


class AckResponse(BaseModel):
    alert_id: str
    acknowledged_at: str
    time_to_ack_seconds: float
    status: str


class ResolveRequest(BaseModel):
    reason: Optional[str] = Field(None, max_length=500, description="Resolution reason.")


class ResolveResponse(BaseModel):
    alert_id: str
    resolved_at: str
    reason: Optional[str] = None
    status: str


class SnoozeResponse(BaseModel):
    alert_id: str
    snoozed_until: str
    snooze_minutes: int
    status: str


class AuditEntry(BaseModel):
    action: str
    user_id: Optional[str] = None
    detail: Optional[str] = None
    timestamp: str


class LifecycleResponse(BaseModel):
    alert_id: str
    audit_trail: list[AuditEntry]


class EscalationItem(BaseModel):
    alert_id: str
    patient_id: str
    risk_level: str
    tier: int
    tier_label: str
    elapsed_minutes: float


class EscalationCheckResponse(BaseModel):
    escalations: list[EscalationItem]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/channels",
    response_model=ChannelsResponse,
    summary="List available notification channels",
)
async def list_channels(user = Depends(_require_auth)) -> ChannelsResponse:
    """Return the notification channels that are currently configured."""
    dispatcher = get_dispatcher()
    return ChannelsResponse(available=dispatcher.available_channels)


@router.post(
    "/contacts",
    response_model=ContactResponse,
    status_code=201,
    summary="Register a notification contact",
)
async def create_contact(body: ContactCreate, user = Depends(_require_auth)) -> ContactResponse:
    """Register a new contact to receive sepsis alerts.

    Nurses in the field typically register their phone number once via
    this endpoint so that subsequent alerts are delivered by SMS.
    """
    if body.channel not in VALID_CHANNELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid channel {body.channel!r}. "
                f"Must be one of {sorted(VALID_CHANNELS)}."
            ),
        )

    dispatcher = get_dispatcher()
    contact = dispatcher.register_contact(
        user_id=body.user_id,
        channel=body.channel,
        destination=body.destination,
    )
    return ContactResponse(**contact.to_dict())


@router.get(
    "/contacts",
    response_model=ContactListResponse,
    summary="List registered contacts",
)
async def list_contacts(user = Depends(_require_auth)) -> ContactListResponse:
    """Return every notification contact currently registered."""
    dispatcher = get_dispatcher()
    contacts = dispatcher.list_contacts()
    return ContactListResponse(
        contacts=[ContactResponse(**c) for c in contacts],
        count=len(contacts),
    )


@router.delete(
    "/contacts/{contact_id}",
    response_model=MessageResponse,
    summary="Remove a notification contact",
)
async def delete_contact(contact_id: str, user = Depends(_require_auth)) -> MessageResponse:
    """Unregister a contact so it no longer receives alerts."""
    dispatcher = get_dispatcher()
    removed = dispatcher.remove_contact(contact_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Contact not found.")
    return MessageResponse(detail="Contact removed.")


@router.post(
    "/test",
    response_model=TestAlertResponse,
    summary="Send a test alert",
)
async def send_test_alert(body: TestAlertRequest, user = Depends(_require_auth)) -> TestAlertResponse:
    """Send a one-off test notification to verify channel connectivity.

    Use this after registering a new phone number to confirm SMS
    delivery is working before going live.
    """
    if body.channel not in VALID_CHANNELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid channel {body.channel!r}. "
                f"Must be one of {sorted(VALID_CHANNELS)}."
            ),
        )

    dispatcher = get_dispatcher()
    result = dispatcher.send_test_alert(body.channel, body.destination)
    return TestAlertResponse(channel=body.channel, result=result)


@router.get(
    "/history",
    response_model=DeliveryHistoryResponse,
    summary="Alert delivery history",
)
async def delivery_history(
    limit: int = Query(50, ge=1, le=200, description="Max records to return."),
    offset: int = Query(0, ge=0, description="Number of records to skip."),
    user = Depends(_require_auth),
) -> DeliveryHistoryResponse:
    """Return recent alert delivery attempts with their status.

    Useful for auditing whether SMS messages were actually delivered
    and diagnosing gateway failures.
    """
    dispatcher = get_dispatcher()
    history = dispatcher.get_history(limit=limit, offset=offset)
    return DeliveryHistoryResponse(
        history=[DeliveryHistoryItem(**h) for h in history],
        count=len(history),
    )


@router.post(
    "/subscribe/push",
    response_model=PushSubscriptionResponse,
    status_code=201,
    summary="Register a Web Push subscription",
)
async def subscribe_push(body: PushSubscriptionRequest, user = Depends(_require_auth)) -> PushSubscriptionResponse:
    """Register a browser push subscription.

    The client should call this with the ``PushSubscription`` object
    obtained from ``serviceWorkerRegistration.pushManager.subscribe()``.
    """
    dispatcher = get_dispatcher()
    push_service = dispatcher._push_service
    if push_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Web Push is not configured. "
                "Set VAPID_PRIVATE_KEY and VAPID_EMAIL environment variables."
            ),
        )

    subscription_info = {
        "endpoint": body.endpoint,
        "keys": {
            "p256dh": body.keys.p256dh,
            "auth": body.keys.auth,
        },
    }
    registered = push_service.subscribe(subscription_info)
    return PushSubscriptionResponse(
        registered=registered,
        endpoint=body.endpoint,
        subscription_count=push_service.subscription_count,
    )


# ---------------------------------------------------------------------------
# Alert lifecycle / escalation endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/ack/{alert_id}",
    response_model=AckResponse,
    summary="Acknowledge an alert",
)
async def acknowledge_alert(alert_id: str, user=Depends(_require_auth)) -> AckResponse:
    """Acknowledge an alert, stopping its escalation clock.

    The authenticated user is recorded as the acknowledger.
    """
    manager = get_escalation_manager()
    user_id = user.get("sub", "unknown") if isinstance(user, dict) else getattr(user, "id", "unknown")
    try:
        result = manager.acknowledge_alert(alert_id, user_id=user_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return AckResponse(**result)


@router.post(
    "/resolve/{alert_id}",
    response_model=ResolveResponse,
    summary="Resolve an alert",
)
async def resolve_alert(
    alert_id: str,
    body: ResolveRequest = None,
    user=Depends(_require_auth),
) -> ResolveResponse:
    """Mark an alert as resolved with an optional reason."""
    manager = get_escalation_manager()
    user_id = user.get("sub", "unknown") if isinstance(user, dict) else getattr(user, "id", "unknown")
    reason = body.reason if body else None
    try:
        result = manager.resolve_alert(alert_id, user_id=user_id, reason=reason)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return ResolveResponse(**result)


@router.post(
    "/snooze/{alert_id}",
    response_model=SnoozeResponse,
    summary="Snooze an alert",
)
async def snooze_alert(
    alert_id: str,
    body: SnoozeRequest = SnoozeRequest(),
    user=Depends(_require_auth),
) -> SnoozeResponse:
    """Delay escalation of an alert for the specified number of minutes."""
    manager = get_escalation_manager()
    user_id = user.get("sub", "unknown") if isinstance(user, dict) else getattr(user, "id", "unknown")
    try:
        result = manager.snooze_alert(
            alert_id, user_id=user_id, snooze_minutes=body.minutes
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return SnoozeResponse(**result)


@router.get(
    "/lifecycle/{alert_id}",
    response_model=LifecycleResponse,
    summary="Get alert audit trail",
)
async def get_alert_lifecycle(
    alert_id: str, user=Depends(_require_auth)
) -> LifecycleResponse:
    """Return the full lifecycle audit trail for an alert."""
    manager = get_escalation_manager()
    try:
        trail = manager.get_alert_lifecycle(alert_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return LifecycleResponse(
        alert_id=alert_id,
        audit_trail=[AuditEntry(**entry) for entry in trail],
    )


@router.post(
    "/check-escalations",
    response_model=EscalationCheckResponse,
    summary="Check and trigger alert escalations",
)
async def check_escalations(user=Depends(_require_auth)) -> EscalationCheckResponse:
    """Check all tracked alerts for escalation.

    Intended to be called periodically (e.g. by a cron job every minute).
    Returns any alerts that were escalated during this check.
    """
    manager = get_escalation_manager()
    escalations = manager.check_escalations()
    return EscalationCheckResponse(
        escalations=[EscalationItem(**e) for e in escalations],
        count=len(escalations),
    )
