#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python "$ROOT/evaluator/run_batch.py" --data-root "$ROOT" --output-root "$ROOT/eval_results_batch" --force
