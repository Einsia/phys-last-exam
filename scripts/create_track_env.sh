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

uv venv --python 3.12 --allow-existing "$ENV_DIR"
uv pip install --python "$ENV_DIR/bin/python" -r "$PROJ/envs/track-requirements.txt"

# SAM3 is distributed as a source checkout plus an access-controlled checkpoint.
# Keep it optional so the existing SAM2 + CoTracker setup remains reproducible; when
# cache/sam3 (or SAM3_SRC) is present, install that checkout into the same worker env.
SAM3_SRC="${SAM3_SRC:-$PROJ/cache/sam3}"
if [[ -f "$SAM3_SRC/pyproject.toml" ]]; then
  uv pip install --python "$ENV_DIR/bin/python" -e "$SAM3_SRC"
  echo "SAM3 source installed from $SAM3_SRC"
else
  echo "SAM3 source not found at $SAM3_SRC (SAM3 backend will be unavailable)"
fi

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
