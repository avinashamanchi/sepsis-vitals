"""
sepsis_vitals.realtime.websocket — WebSocket alert streaming.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class ConnectionManager:
    """Manages WebSocket connections for real-time alert broadcasting.

    Each connection can optionally be associated with an ``org_id``.
    When broadcasting, messages that include a ``patient_id`` are only
    sent to connections whose org owns that patient.  Connections with
    ``org_id=None`` (demo/dev mode) receive all messages.
    """

    def __init__(self):
        # List of (websocket, org_id) tuples
        self._connections: list[tuple[Any, str | None]] = []

    async def connect(self, websocket: Any, *, org_id: str | None = None) -> None:
        await websocket.accept()
        self._connections.append((websocket, org_id))

    def disconnect(self, websocket: Any) -> None:
        self._connections = [
            (ws, oid) for ws, oid in self._connections if ws is not websocket
        ]

    def _patient_org_id(self, patient_id: str) -> str | None:
        """Look up the site_id (org) for a patient. Returns None if not found."""
        try:
            from sepsis_vitals.db import SessionLocal, Patient
            db = SessionLocal()
            try:
                patient = db.query(Patient).filter(Patient.id == patient_id).first()
                return patient.site_id if patient else None
            finally:
                db.close()
        except Exception:
            return None

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        disconnected = []

        # Determine which org owns the patient in this message (if any)
        patient_id = message.get("patient_id")
        patient_org = self._patient_org_id(patient_id) if patient_id else None

        for ws, conn_org_id in self._connections:
            # Skip if this connection has an org and it doesn't match the patient's org
            if conn_org_id is not None and patient_org is not None and conn_org_id != patient_org:
                continue
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    @property
    def active_connections(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


def format_alert_message(
    alert_type: str,
    patient_id: str,
    risk_probability: float,
    risk_level: str,
    previous_risk_level: str | None = None,
    risk_delta: float = 0.0,
    deterioration_rate: float = 0.0,
    window_hours: float = 0.0,
) -> dict:
    """Format a typed alert message for WebSocket broadcast.

    Alert types:
    - patient_update: routine vitals/risk refresh
    - deterioration: sustained risk increase over 2-hour window
    - recovery: sustained risk decrease over 2-hour window
    - escalation: risk level crossed into high/critical
    - new_risk: first prediction for a patient
    """
    type_map = {
        "patient_update": "patient_update",
        "deterioration": "deterioration_alert",
        "recovery": "recovery_alert",
        "escalation": "escalation_alert",
        "new_risk": "new_risk_alert",
    }

    msg = {
        "type": type_map.get(alert_type, alert_type),
        "patient_id": patient_id,
        "risk_probability": risk_probability,
        "risk_level": risk_level,
    }

    if previous_risk_level is not None:
        msg["previous_risk_level"] = previous_risk_level

    if alert_type in ("deterioration", "recovery"):
        msg["risk_delta"] = risk_delta

    if alert_type == "deterioration":
        msg["deterioration_rate"] = deterioration_rate
        msg["window_hours"] = window_hours

    return msg


async def alert_producer(vitals_queue: asyncio.Queue) -> None:
    """Consume vitals from queue, score them, and broadcast alerts."""
    from sepsis_vitals.scores import compute_scores

    while True:
        vitals = await vitals_queue.get()
        result = compute_scores(vitals)
        if result.alert_flag:
            await manager.broadcast({
                "type": "alert",
                "risk_level": result.risk_level,
                "scores": result.as_dict(),
                "vitals": vitals,
            })
