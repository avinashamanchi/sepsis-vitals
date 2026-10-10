"""
sepsis_vitals.fhir.listener
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Background listener for automatic vital sign ingestion from hospital systems.

Supports:
- HL7v2 MLLP: Listens on a TCP socket for ORU^R01 observation messages
- FHIR R4 webhook: Accepts POST requests with FHIR Observation bundles

Eliminates the "manual entry problem" -- vitals flow directly from bedside
monitors and EHR systems without nurse double-documentation.

Both ingestion paths normalise observations into ``VitalsReading`` objects and
feed them through a single ``VitalsIngestionQueue``, keeping downstream
processing uniform regardless of wire format.

Usage::

    queue = VitalsIngestionQueue()
    queue.register_handler(my_callback)

    mllp = MLLPServer(port=2575, queue=queue)
    webhook = FHIRWebhookServer(port=8090, queue=queue)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(queue.process_loop())
        tg.create_task(mllp.start())
        tg.create_task(webhook.start())
"""

# This module is a compatibility facade: the implementation is split by
# responsibility into the modules below, and every public name is re-exported.

from __future__ import annotations

import logging

from sepsis_vitals.fhir.ingest_models import (  # noqa: F401
    DISPLAY_NAME_MAP,
    HL7Message,
    LOINC_VITAL_MAP,
    VitalsReading,
)
from sepsis_vitals.fhir.hl7 import (  # noqa: F401
    HL7Parser,
)
from sepsis_vitals.fhir.observation_parser import (  # noqa: F401
    FHIRObservationParser,
)
from sepsis_vitals.fhir.ingest_queue import (  # noqa: F401
    VitalsIngestionQueue,
)
from sepsis_vitals.fhir.mllp import (  # noqa: F401
    MLLPServer,
)
from sepsis_vitals.fhir.webhook import (  # noqa: F401
    FHIRWebhookHandler,
    FHIRWebhookServer,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DISPLAY_NAME_MAP",
    "HL7Message",
    "LOINC_VITAL_MAP",
    "VitalsReading",
    "HL7Parser",
    "FHIRObservationParser",
    "VitalsIngestionQueue",
    "MLLPServer",
    "FHIRWebhookHandler",
    "FHIRWebhookServer",
]
