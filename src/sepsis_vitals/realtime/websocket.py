"""
sepsis_vitals.realtime.websocket — WebSocket alert streaming.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class ConnectionManager:
    """Manages WebSocket connections for real-time alert broadcasting."""

    def __init__(self):
        self._connections: list[Any] = []

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: Any) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        disconnected = []
        for ws in self._connections:
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
