"""Validate every canonical G7 sample JSON and report aggregate counts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.contract import validate_public_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing task/eval_results/json/sample_*.json")
    args = parser.parse_args(argv)
    files = sorted(args.results.glob("P*/eval_results/json/sample_*.json"))
    if not files:
        files = sorted(args.results.glob("*/eval_results/json/sample_*.json"))
    if not files:
        print(f"no sample JSON files found under {args.results}", file=sys.stderr)
        return 2
    failures = 0
    for path in files:
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"{path}: invalid JSON: {exc}", file=sys.stderr)
            failures += 1
            continue
        errors = validate_public_result(sample)
        if errors:
            failures += 1
            for error in errors:
                print(f"{path}: {error}", file=sys.stderr)
    if failures:
        print(f"contract validation failed: {failures}/{len(files)} files", file=sys.stderr)
        return 1
    print(f"contract validation passed: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
