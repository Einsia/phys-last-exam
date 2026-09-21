#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evaluation_root="$(cd -- "$task_dir/../.." && pwd)"
evaluation_python="${EVALUATOR_PYTHON:-$evaluation_root/.venv/bin/python}"
model="minimax_h3"
if [[ $# -gt 0 && "$1" != -* ]]; then model="$1"; shift; fi
exec "$evaluation_python" "$evaluation_root/scripts/run_all_eval.py" --groups g1 --tasks P5 --model "$model" --log-dir "$task_dir/eval_results/run_logs" "$@"
