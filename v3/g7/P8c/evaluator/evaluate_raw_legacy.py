from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluator.task_entrypoint import run_task_cli
if __name__ == '__main__':
    raise SystemExit(run_task_cli('P8c'))
