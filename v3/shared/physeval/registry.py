"""Task lookup. One module per benchmark task id."""
from __future__ import annotations

import importlib
from typing import Callable

from .context import Context
from .schema import Result
from .video import Clip

# Only the tasks this delivery covers: Group 1, Group 4, Group 6.
TASK_MODULES = {
    "P1": "p1", "P2": "p2", "P5": "p5", "P8a": "p8a", "P8b": "p8b",
    "P14": "p14", "P16": "p16", "P18": "p18", "P20": "p20", "P21": "p21",
    "P45": "p45", "P48": "p48",
}


def get_evaluator(task_id: str) -> Callable[[Clip, Context], Result]:
    if task_id not in TASK_MODULES:
        raise KeyError(f"no evaluator for task {task_id!r}; "
                       f"known: {sorted(TASK_MODULES)}")
    mod = importlib.import_module(f".tasks.{TASK_MODULES[task_id]}", __package__)
    return mod.evaluate
