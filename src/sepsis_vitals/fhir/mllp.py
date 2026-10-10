"""
sepsis_vitals.fhir.mllp

MLLP (HL7v2 over TCP) server with optional mTLS.
Split out of sepsis_vitals.fhir.listener, which re-exports every name.
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
from typing import Dict, Optional
from sepsis_vitals.security import log_ref  # noqa: E402
from sepsis_vitals.fhir.hl7 import HL7Parser
from sepsis_vitals.fhir.ingest_queue import VitalsIngestionQueue

# Keep the original logger name so existing log routing is unchanged.
logger = logging.getLogger("sepsis_vitals.fhir.listener")

# ---------------------------------------------------------------------------
# MLLP Server (HL7v2 over TCP)
# ---------------------------------------------------------------------------


class MLLPServer:
    """Minimal Lower Layer Protocol server for HL7v2 message reception.

    MLLP is the standard transport for HL7v2 messages over TCP.  Each
    message is wrapped in framing bytes:

    - Start block: ``\\x0b`` (vertical tab / VT)
    - End block: ``\\x1c\\x0d`` (file separator + carriage return)

    The server listens for connections, extracts HL7 messages from the
    MLLP framing, parses ORU^R01 observations, and feeds extracted vitals
    into the ``VitalsIngestionQueue``.

    Parameters
    ----------
    host : str
        Bind address. Defaults to loopback; explicitly configure a protected
        interface when receiving traffic from another host.
    port : int
        TCP port.  The HL7 MLLP default is 2575.
    queue : VitalsIngestionQueue | None
        Queue for extracted vitals.  If ``None``, a new queue is created.
    """

    START_BLOCK = b"\x0b"
    END_BLOCK = b"\x1c\x0d"

    # Maximum message size: 1 MB (HL7 messages rarely exceed a few KB,
    # but lab result batches can be larger).
    MAX_MESSAGE_SIZE = 1024 * 1024

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2575,
        queue: Optional[VitalsIngestionQueue] = None,
        tls_cert: Optional[str] = None,
        tls_key: Optional[str] = None,
        tls_ca: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.queue = queue or VitalsIngestionQueue()
        self._parser = HL7Parser()
        self._server: Optional[asyncio.AbstractServer] = None
        self._active_connections: int = 0
        # TLS configuration — read from params or environment
        self._tls_cert = tls_cert or os.environ.get("MLLP_TLS_CERT")
        self._tls_key = tls_key or os.environ.get("MLLP_TLS_KEY")
        self._tls_ca = tls_ca or os.environ.get("MLLP_TLS_CA")
        self._stats = {
            "connections": 0,
            "messages_received": 0,
            "messages_accepted": 0,
            "messages_rejected": 0,
            "parse_errors": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of server statistics."""
        return dict(self._stats)

    async def start(self) -> None:
        """Start the MLLP server.

        This coroutine starts the TCP server and serves connections until
        ``stop()`` is called or the task is cancelled.  When TLS certificate
        and key paths are configured (via constructor args or MLLP_TLS_CERT /
        MLLP_TLS_KEY environment variables), the server wraps connections in
        TLS to satisfy HIPAA transit encryption requirements.
        """
        ssl_ctx = self._create_ssl_context()

        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port, ssl=ssl_ctx,
        )

        proto = "MLLP+TLS" if ssl_ctx else "MLLP (plaintext)"
        addrs = [
            str(sock.getsockname()) for sock in self._server.sockets
        ]
        logger.info(
            "%s server started on %s (HL7v2 ingestion active)", proto, addrs
        )
        if not ssl_ctx and os.environ.get("SEPSIS_ENV") == "production":
            logger.warning(
                "MLLP running WITHOUT TLS in production. "
                "Transmitting PHI over unencrypted TCP violates HIPAA "
                "Security Rule §164.312(e)(1). Set MLLP_TLS_CERT and "
                "MLLP_TLS_KEY or route traffic through a TLS tunnel."
            )

        async with self._server:
            await self._server.serve_forever()

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Build an SSL context if TLS cert/key are configured.

        Returns ``None`` when TLS is not configured, causing the server
        to fall back to plaintext TCP.
        """
        if not self._tls_cert or not self._tls_key:
            return None

        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(self._tls_cert, self._tls_key)

            if self._tls_ca:
                # Mutual TLS: require client certificate
                ctx.load_verify_locations(self._tls_ca)
                ctx.verify_mode = ssl.CERT_REQUIRED
                logger.info(
                    "MLLP TLS: mutual authentication enabled (CA: %s)",
                    self._tls_ca,
                )
            else:
                ctx.verify_mode = ssl.CERT_NONE

            logger.info(
                "MLLP TLS enabled (cert: %s, min version: TLSv1.2)",
                self._tls_cert,
            )
            return ctx
        except Exception:
            logger.exception("Failed to create MLLP TLS context")
            raise

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming MLLP connection.

        A single TCP connection may carry multiple HL7 messages (persistent
        connection).  Each message is individually ACKed.
        """
        peer = writer.get_extra_info("peername", ("unknown", 0))
        self._active_connections += 1
        self._stats["connections"] += 1
        logger.info(
            "MLLP connection from %s:%s (active: %d)",
            peer[0], peer[1], self._active_connections,
        )

        try:
            buffer = b""
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        reader.read(4096), timeout=300.0
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "MLLP connection from %s:%s timed out", peer[0], peer[1]
                    )
                    break

                if not chunk:
                    break  # Connection closed

                buffer += chunk

                # Process all complete messages in the buffer
                while True:
                    start_idx = buffer.find(self.START_BLOCK)
                    if start_idx == -1:
                        # No start block found; discard leading garbage
                        buffer = b""
                        break

                    end_idx = buffer.find(self.END_BLOCK, start_idx)
                    if end_idx == -1:
                        # Incomplete message; wait for more data
                        if len(buffer) > self.MAX_MESSAGE_SIZE:
                            logger.error(
                                "MLLP buffer exceeded %d bytes from %s:%s; "
                                "resetting buffer",
                                self.MAX_MESSAGE_SIZE, peer[0], peer[1],
                            )
                            buffer = b""
                        break

                    # Extract the HL7 message between framing bytes
                    msg_bytes = buffer[start_idx + 1 : end_idx]
                    buffer = buffer[end_idx + len(self.END_BLOCK) :]

                    self._stats["messages_received"] += 1

                    # Process the message and send ACK
                    ack = await self._process_message(msg_bytes, peer)
                    if ack:
                        # Wrap ACK in MLLP framing
                        framed_ack = (
                            self.START_BLOCK
                            + ack.encode("utf-8")
                            + self.END_BLOCK
                        )
                        writer.write(framed_ack)
                        await writer.drain()

        except ConnectionResetError:
            logger.info(
                "MLLP connection reset by %s:%s", peer[0], peer[1]
            )
        except Exception:
            logger.exception(
                "Unexpected error handling MLLP connection from %s:%s",
                peer[0], peer[1],
            )
        finally:
            self._active_connections -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(
                "MLLP connection from %s:%s closed (active: %d)",
                peer[0], peer[1], self._active_connections,
            )

    async def _process_message(
        self,
        msg_bytes: bytes,
        peer: tuple,
    ) -> Optional[str]:
        """Parse an HL7 message and queue extracted vitals.

        Returns the ACK message string, or ``None`` if no ACK should be
        sent.
        """
        try:
            raw = msg_bytes.decode("utf-8", errors="replace")
        except Exception:
            logger.error(
                "Failed to decode MLLP message from %s:%s",
                peer[0], peer[1],
            )
            self._stats["parse_errors"] += 1
            return None

        try:
            msg = self._parser.parse(raw)
        except ValueError as exc:
            logger.warning(
                "Failed to parse HL7 message from %s:%s: %s",
                peer[0], peer[1], exc,
            )
            self._stats["parse_errors"] += 1
            return None

        # Only process ORU (observation result) messages
        msg_type_code = msg.message_type.split(
            HL7Parser.COMPONENT_SEP
        )[0] if msg.message_type else ""

        if msg_type_code != "ORU":
            logger.info(
                "Ignoring non-ORU message type %r from %s:%s",
                msg.message_type, peer[0], peer[1],
            )
            self._stats["messages_rejected"] += 1
            # Still ACK it so the sender doesn't retry
            return self._parser.build_ack(msg, ack_code="AA")

        # Extract vitals from OBX segments
        try:
            reading = self._parser.extract_vitals(msg)
        except Exception:
            logger.exception(
                "Error extracting vitals from ORU message (control_id=%s)",
                msg.message_control_id,
            )
            self._stats["parse_errors"] += 1
            return self._parser.build_ack(msg, ack_code="AE")

        if not reading.vitals:
            logger.info(
                "ORU message %s contained no recognised vital signs",
                msg.message_control_id,
            )
            self._stats["messages_accepted"] += 1
            return self._parser.build_ack(msg, ack_code="AA")

        # Queue for downstream processing
        await self.queue.put(reading)
        self._stats["messages_accepted"] += 1

        logger.info(
            "Accepted ORU message %s: patient=%s, vitals=%s",
            msg.message_control_id,
            log_ref(reading.patient_id),
            list(reading.vitals.keys()),
        )
        return self._parser.build_ack(msg, ack_code="AA")

    async def stop(self) -> None:
        """Stop the MLLP server gracefully.

        Closes the listening socket and waits for active connections to
        drain.
        """
        if self._server is not None:
            logger.info(
                "Stopping MLLP server (active connections: %d)",
                self._active_connections,
            )
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("MLLP server stopped")


