"""Validate public ``sample_xx.json`` result envelopes.

Usage from a task directory or the Group 7 root::

    python evaluator/validate_schema.py eval_results/json

The command exits non-zero on the first structural/path error and prints a
short per-file diagnostic.  It has no third-party dependencies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .result_schema import TASK_METRIC_SPECS, validate_sample_result
except ImportError:  # direct script execution
    from result_schema import TASK_METRIC_SPECS, validate_sample_result


def _task_for(path: Path, root: Path, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    # A task directory is conventionally the parent of eval_results/json.
    parts = path.resolve().parts
    for item in reversed(parts):
        if item in TASK_METRIC_SPECS:
            return item
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="one JSON file or a directory containing sample_*.json")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--allow-absolute-paths", action="store_true")
    args = parser.parse_args(argv)

    target = args.path.resolve()
    files = [target] if target.is_file() else sorted(target.glob("sample_*.json"))
    if not files:
        print(f"no sample_*.json files found under {target}", file=sys.stderr)
        return 2

    failures = 0
    for path in files:
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"{path}: invalid JSON: {exc}", file=sys.stderr)
            failures += 1
            continue
        errors = validate_sample_result(
            value,
            task_id=_task_for(path, target, args.task_id),
            require_relative_paths=not args.allow_absolute_paths,
        )
        if errors:
            failures += 1
            for error in errors:
                print(f"{path}: {error}", file=sys.stderr)
    if failures:
        print(f"schema validation failed: {failures}/{len(files)} file(s)", file=sys.stderr)
        return 1
    print(f"schema validation passed: {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
