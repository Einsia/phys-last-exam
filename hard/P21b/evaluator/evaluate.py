#!/usr/bin/env python3
"""P21b: fresh measurement with the unified G1-G9 V2 output contract."""
from pathlib import Path
import sys
ROOT = (Path(__file__).resolve().parent / "_shared")
sys.path.insert(0, str(ROOT))
from unified_evaluators.runtime import main
if __name__ == '__main__':
    raise SystemExit(main('P21b', Path(__file__).resolve().parents[1]))
