#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from evaluator_common import run
if __name__ == "__main__": sys.exit(run("P8c"))
