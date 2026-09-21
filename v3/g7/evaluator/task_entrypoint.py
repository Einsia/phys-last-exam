"""Fixed-task command-line adapter used by the five task entrypoints.

The shared implementation is intentionally kept in :mod:`evaluator.evaluate`
for unit-test reuse, but each delivery entrypoint binds one task ID.  Passing a
different ``--task_id`` is rejected instead of silently routing a video to a
neighbouring task's physics rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluator.evaluate import evaluate


def run_task_cli(task_id: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Deterministic evaluator for {task_id}")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample_id", default="0",
                        help="sample index or stable current-batch sample ID")
    parser.add_argument("--debug-dir", default=None)
    # Kept for compatibility with the old task-local shell wrappers, but it
    # can only name this entrypoint's task.
    parser.add_argument("--task_id", default=task_id, choices=[task_id])
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.output)
    debug_dir = Path(args.debug_dir) if args.debug_dir else output.parent / "debug" / str(args.sample_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(task_id, args.video, args.sample_id, debug_dir)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.get("extract_success") else 2
