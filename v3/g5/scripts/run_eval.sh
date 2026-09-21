#!/usr/bin/env bash
set -euo pipefail
group_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
evaluation_root="$(cd -- "$group_dir/.." && pwd)"
evaluation_python="${EVALUATOR_PYTHON:-$evaluation_root/.venv/bin/python}"
model="minimax_h3"
if [[ $# -gt 0 && "$1" != -* ]]; then model="$1"; shift; fi
exec "$evaluation_python" "$evaluation_root/scripts/run_all_eval.py" --groups g5 --model "$model" --log-dir "$group_dir/eval_run_logs" "$@"
