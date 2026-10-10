"""
sepsis_vitals.routes.realtime

Endpoints moved out of sepsis_vitals.api (behaviour unchanged). They
register on the shared ``app`` and reach shared state through ``core`` at
call time, so tests and callers that patch ``sepsis_vitals.api`` still work.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional
from fastapi import (
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sepsis_vitals.scores import compute_scores
from sepsis_vitals.realtime.websocket import manager as ws_manager

from sepsis_vitals import api as core

# ---------------------------------------------------------------------------
# WebSocket endpoint for real-time alerts
# ---------------------------------------------------------------------------

@core.app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Real-time sepsis alert stream via WebSocket.

    Clients receive JSON messages when any patient triggers a high/critical alert.
    Authentication uses the ``bearer.<JWT>`` WebSocket subprotocol so credentials
    never appear in URLs or access logs.
    """
    # Authenticate WebSocket handshake via JWT
    ws_org_id = None  # org_id for filtering broadcasts
    ws_expires_at: Optional[float] = None  # close when the access token expires
    selected_subprotocol = None
    if core._auth_enabled:
        offered = [
            value.strip()
            for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if value.strip()
        ]
        token_protocol = next(
            (value for value in offered if value.startswith("bearer.")),
            None,
        )
        token = token_protocol.removeprefix("bearer.") if token_protocol else None
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        try:
            from sepsis_vitals.auth.tokens import decode_token
            payload = decode_token(token)
            if payload.get("type") != "access":
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            ws_org_id = payload.get("org_id")
            ws_expires_at = float(payload["exp"])
            if ws_org_id is None and payload.get("role") != "system_admin":
                # Fail closed: an org-less connection would receive every site's alerts.
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            selected_subprotocol = "sepsis-vitals" if "sepsis-vitals" in offered else None
        except Exception:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await ws_manager.connect(
        websocket,
        org_id=ws_org_id,
        subprotocol=selected_subprotocol,
    )
    try:
        while True:
            # Keep connection alive, receive any client messages. The session
            # must not outlive its access token: the client reconnects with a
            # fresh token after refreshing.
            if ws_expires_at is not None:
                remaining = ws_expires_at - time.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                data = await asyncio.wait_for(websocket.receive_text(), timeout=remaining)
            else:
                data = await websocket.receive_text()
            # Client can send vitals for immediate scoring
            try:
                vitals = json.loads(data)
                scores = compute_scores(vitals)
                await websocket.send_json({
                    "type": "score_result",
                    "scores": scores.as_dict(),
                })
            except Exception:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except asyncio.TimeoutError:
        ws_manager.disconnect(websocket)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="token expired")


