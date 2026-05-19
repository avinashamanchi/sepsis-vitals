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
