#!/usr/bin/env python3
"""Same as `python -m sepsis_vitals.study` (see sepsis_vitals/study/__main__.py)."""

import sys

from sepsis_vitals.study.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
