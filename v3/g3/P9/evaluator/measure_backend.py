#!/usr/bin/env python3
"""V2 scoring entry: replay saved measurements, not fresh video extraction."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scoring_v2"))
from package_cli import main
if __name__ == "__main__":
    raise SystemExit(main(default_group='G3', default_task='P9'))
