#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${1:-$TASK_ROOT/eval_results_recomputed}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite evaluation output: $OUTPUT_DIR" >&2
  exit 2
fi
exec "$PYTHON_BIN" "$TASK_ROOT/evaluator/batch_evaluate.py" --videos "$TASK_ROOT/minimax_h3/videos" --output-root "$OUTPUT_DIR" "$@"
