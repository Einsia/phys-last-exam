#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PLE_PYTHON:-python3.12}" "$root/scripts/run.py" evaluate "$@"
