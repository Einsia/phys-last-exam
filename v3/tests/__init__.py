"""Test package bootstrap for running discovery from the repository root."""
from pathlib import Path
import sys

V3_ROOT = str(Path(__file__).resolve().parents[1])
if V3_ROOT not in sys.path:
    sys.path.insert(0, V3_ROOT)
