"""
sepsis_vitals.logging_config — make application logs reach stdout.

uvicorn configures only its own loggers. Without this, records from
``sepsis_vitals.*`` loggers below WARNING (including the ``HIPAA_AUDIT``
structured audit lines) reach Python's last-resort handler, which drops
them, so the container's log stream had no audit trail.

The handler is attached to the ``sepsis_vitals`` logger, not the root, and
only when the process has configured no logging of its own (no root
handlers). A host that sets up logging, or pytest's capture, keeps control.
"""

from __future__ import annotations

import logging
import os
import sys

_HANDLER_NAME = "sepsis_vitals.stdout"


def configure_logging() -> bool:
    """Attach a stdout handler to ``sepsis_vitals`` once. Returns True if added.

    The level comes from ``LOG_LEVEL`` (default INFO).
    """
    app_logger = logging.getLogger("sepsis_vitals")
    if logging.getLogger().handlers or any(h.get_name() == _HANDLER_NAME for h in app_logger.handlers):
        return False
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    app_logger.addHandler(handler)
    level = logging.getLevelName(os.getenv("LOG_LEVEL", "INFO").upper())
    app_logger.setLevel(level if isinstance(level, int) else logging.INFO)
    return True
