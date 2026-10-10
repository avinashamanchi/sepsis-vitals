"""
sepsis_vitals.fhir.ingest_queue

Bounded queue that fans VitalsReading objects out to handlers.
Split out of sepsis_vitals.fhir.listener, which re-exports every name.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List
from sepsis_vitals.security import log_ref  # noqa: E402
from sepsis_vitals.fhir.ingest_models import VitalsReading

# Keep the original logger name so existing log routing is unchanged.
logger = logging.getLogger("sepsis_vitals.fhir.listener")

# ---------------------------------------------------------------------------
# Vitals Ingestion Queue
# ---------------------------------------------------------------------------


class VitalsIngestionQueue:
    """Async queue for incoming vitals readings.

    Provides a single fan-out point: readings arrive from any ingestion
    source (MLLP, FHIR webhook, manual entry) and are dispatched to all
    registered handler callbacks.

    The queue is bounded to prevent unbounded memory growth if downstream
    processing stalls.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._queue: asyncio.Queue[VitalsReading] = asyncio.Queue(
            maxsize=max_size
        )
        self._handlers: List[Callable] = []
        self._running = False
        self._stats = {
            "received": 0,
            "processed": 0,
            "errors": 0,
            "dropped": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of queue processing statistics."""
        return dict(self._stats)

    @property
    def pending(self) -> int:
        """Number of readings waiting to be processed."""
        return self._queue.qsize()

    def register_handler(self, handler: Callable) -> None:
        """Register a callback for new vitals readings.

        Handlers are invoked in registration order for each reading.
        A handler may be a coroutine function (``async def``) or a plain
        callable.

        Parameters
        ----------
        handler : callable
            Receives a single ``VitalsReading`` argument.
        """
        self._handlers.append(handler)
        logger.info(
            "Registered vitals handler: %s (total: %d)",
            getattr(handler, "__name__", repr(handler)),
            len(self._handlers),
        )

    async def put(self, reading: VitalsReading) -> None:
        """Add a reading to the queue.

        If the queue is full, the reading is dropped and a warning is
        logged.  This prevents backpressure from blocking the ingestion
        server, which must remain responsive to send HL7 ACKs.
        """
        try:
            self._queue.put_nowait(reading)
            self._stats["received"] += 1
            logger.debug(
                "Queued vitals for patient %s (%d vitals, source=%s)",
                log_ref(reading.patient_id),
                len(reading.vitals),
                reading.source,
            )
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            logger.warning(
                "Vitals queue full (%d items) -- dropped reading for "
                "patient %s.  Consider increasing queue size or adding "
                "more processing capacity.",
                self._queue.maxsize,
                log_ref(reading.patient_id),
            )

    async def process_loop(self) -> None:
        """Background loop that processes queued readings.

        Runs indefinitely, pulling readings from the queue and dispatching
        them to all registered handlers.  Errors in individual handlers are
        logged but do not stop the loop or affect other handlers.
        """
        self._running = True
        logger.info(
            "Vitals ingestion queue started (max_size=%d, handlers=%d)",
            self._queue.maxsize,
            len(self._handlers),
        )

        try:
            while self._running:
                try:
                    reading = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Periodic check of _running flag
                    continue

                await self._dispatch(reading)
                self._queue.task_done()
        except asyncio.CancelledError:
            logger.info("Vitals ingestion queue shutting down")
            # Drain remaining items
            await self._drain()
            raise

    async def stop(self) -> None:
        """Signal the process loop to stop after draining the queue."""
        self._running = False
        logger.info("Vitals ingestion queue stop requested")

    async def _dispatch(self, reading: VitalsReading) -> None:
        """Dispatch a reading to all registered handlers."""
        for handler in self._handlers:
            try:
                result = handler(reading)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                self._stats["errors"] += 1
                logger.exception(
                    "Error in vitals handler %s for patient %s",
                    getattr(handler, "__name__", repr(handler)),
                    log_ref(reading.patient_id),
                )
            else:
                self._stats["processed"] += 1

    async def _drain(self) -> None:
        """Process any remaining items in the queue."""
        drained = 0
        while not self._queue.empty():
            try:
                reading = self._queue.get_nowait()
                await self._dispatch(reading)
                self._queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info("Drained %d remaining readings from queue", drained)


