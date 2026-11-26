#!/usr/bin/env python3
"""Launch the DirGuide lighting experiment window.

Usage:
    python3 run_ui.py [--port 8765] [--host 127.0.0.1]
Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from dirguide.server import main

if __name__ == "__main__":
    raise SystemExit(main())
