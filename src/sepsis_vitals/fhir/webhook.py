"""
sepsis_vitals.fhir.webhook

FHIR R4 webhook handler and the standalone webhook HTTP server.
Split out of sepsis_vitals.fhir.listener, which re-exports every name.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional
from sepsis_vitals.security import log_ref  # noqa: E402
from sepsis_vitals.fhir.ingest_models import VitalsReading
from sepsis_vitals.fhir.ingest_queue import VitalsIngestionQueue
from sepsis_vitals.fhir.observation_parser import FHIRObservationParser

# Keep the original logger name so existing log routing is unchanged.
logger = logging.getLogger("sepsis_vitals.fhir.listener")

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
            log_ref(patient_id), vital,
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
        host: str = "127.0.0.1",
        port: int = 8090,
        path: str = "/fhir/webhook",
        queue: Optional[VitalsIngestionQueue] = None,
        webhook_secret: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.path = path
        self._handler = FHIRWebhookHandler(queue=queue)
        self._server: Optional[asyncio.AbstractServer] = None
        self._webhook_secret = webhook_secret or os.environ.get("SEPSIS_WEBHOOK_SECRET")

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

        # Verify webhook signature if secret is configured
        if self._webhook_secret:
            sig_header = headers.get("x-webhook-signature", "")
            if not sig_header:
                await self._send_response(
                    writer, 401, {"error": "Missing X-Webhook-Signature header"}
                )
                return
            try:
                from sepsis_vitals.security import verify_webhook_signature
                verify_webhook_signature(body_bytes, sig_header, self._webhook_secret)
            except Exception:
                logger.warning(
                    "FHIR webhook signature verification failed from %s:%s",
                    peer[0], peer[1],
                )
                await self._send_response(
                    writer, 403, {"error": "Invalid webhook signature"}
                )
                return
        else:
            env = os.environ.get("SEPSIS_ENV", "").lower()
            if env == "production":
                logger.error(
                    "SEPSIS_WEBHOOK_SECRET is not configured in production; "
                    "rejecting FHIR webhook request from %s:%s",
                    peer[0], peer[1],
                )
                await self._send_response(
                    writer,
                    503,
                    {"error": "Webhook secret is not configured"},
                )
                return
            else:
                logger.warning(
                    "SEPSIS_WEBHOOK_SECRET is not set; accepting "
                    "unauthenticated FHIR webhook data (non-production)"
                )

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
