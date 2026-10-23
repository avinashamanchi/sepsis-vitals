"""HTTP and WebSocket endpoint modules split out of sepsis_vitals.api.

Each module declares its endpoints on its own ``router`` and takes shared
dependencies from :mod:`sepsis_vitals.dependencies` and models from
:mod:`sepsis_vitals.schemas`. None of them imports ``sepsis_vitals.api`` at
import time; ``sepsis_vitals.api`` includes their routers.
"""
