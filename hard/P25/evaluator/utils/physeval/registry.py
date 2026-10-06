"""Task lookup. One module per benchmark task id."""
from __future__ import annotations

import importlib
from typing import Callable

from .context import Context
from .schema import Result
from .video import Clip

# Only the tasks this delivery covers: Group 1, Group 4, Group 6.
TASK_MODULES = {
    "P2": "p2", "P4": "p4", "P5": "p5", "P15": "p15", "P12": "p12",
    "P19": "p19", "P16": "p16", "P23": "p23", "P22": "p22", "P25": "p25",
    "P37": "p37", "P40": "p40",
}


def get_evaluator(task_id: str) -> Callable[[Clip, Context], Result]:
    if task_id not in TASK_MODULES:
        raise KeyError(f"no evaluator for task {task_id!r}; "
                       f"known: {sorted(TASK_MODULES)}")
    mod = importlib.import_module(f".tasks.{TASK_MODULES[task_id]}", __package__)
    return mod.evaluate
