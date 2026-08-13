#!/usr/bin/env bash
# Create envs/track (torch + SAM2 + CoTracker) and fetch the weights into cache/.
#
# Separate from create_env.sh on purpose: torch lives here and nowhere else on the
# measurement path. envs/physbench stays numpy/opencv-only, the worker is called as a
# subprocess, and the two dependency sets never have to agree -- the same argument as
# the VDM in physbench/vdm.py.
#
# Skip this entirely if you only want the classic trackers: set
#   tracking: {backends: [color, bgsub]}
# in the task config and nothing here is needed.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$PROJ/envs/track"

command -v uv >/dev/null || {
  echo "uv not found; install it or adapt this to python -m venv + pip" >&2
  exit 1
}

uv venv --python 3.12 "$ENV_DIR"
uv pip install --python "$ENV_DIR/bin/python" -r "$PROJ/envs/track-requirements.txt"

"$ENV_DIR/bin/python" - <<'PY'
import torch, transformers
from transformers import Sam2VideoModel, Sam2VideoProcessor  # noqa: F401
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"({torch.cuda.device_count()} devices)")
print(f"transformers {transformers.__version__}  Sam2Video* present")
PY

bash "$PROJ/scripts/fetch_tracker_weights.sh"

echo
echo "envs/track ready. Smoke-test the pair end to end with:"
echo "  envs/physbench/bin/python scripts/tracker_precision.py"
