"""Pytest configuration ensuring project root is importable.

Adds the repository root to sys.path explicitly to avoid ModuleNotFoundError
for `solution` when running under certain invocation contexts / Python 3.13.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
