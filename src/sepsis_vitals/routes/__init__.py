"""HTTP and WebSocket endpoint modules split out of sepsis_vitals.api.

Each module registers its endpoints on ``sepsis_vitals.api.app`` at import
time, and ``sepsis_vitals.api`` imports them at the end of its own module.
Importing ``sepsis_vitals.api`` here first means a direct import of a route
module (``import sepsis_vitals.routes.status``) completes the app module
before the route module runs, instead of meeting it half-initialised.
"""

import sepsis_vitals.api  # noqa: F401
