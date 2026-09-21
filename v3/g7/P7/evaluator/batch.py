#!/usr/bin/env python3
from pathlib import Path
import os,sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
os.environ["EVALUATOR_TASK_DIR"] = str(Path(__file__).resolve().parents[1])
from evaluator_common import batch
if __name__ == "__main__": sys.exit(batch("P7"))
