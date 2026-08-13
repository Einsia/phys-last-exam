#!/usr/bin/env bash
# Fetch the tracker weights and vendor the CoTracker source, once, into cache/.
#
# Evaluation must not need the network. A benchmark that silently re-downloads a
# checkpoint can also silently get a *different* checkpoint, and then the noise floor
# in the README no longer describes the tracker that produced the numbers. So both
# models are pinned to a local path here and the worker runs with HF_HUB_OFFLINE=1.
#
# CoTracker ships as a git repo rather than a pip package, and torch.hub.load would
# reach GitHub at call time. Cloning it once and loading with source="local" keeps the
# measurement path offline and the revision recorded.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$PROJ/cache"
COTRACKER_REV="${COTRACKER_REV:-main}"

mkdir -p "$CACHE/torch/hub/checkpoints" "$CACHE/hf"

# ---- CoTracker3 source + offline checkpoint -------------------------------------
if [ ! -d "$CACHE/cotracker/.git" ]; then
  git clone --depth 1 --branch "$COTRACKER_REV" \
    https://github.com/facebookresearch/co-tracker "$CACHE/cotracker"
fi
git -C "$CACHE/cotracker" rev-parse HEAD > "$CACHE/cotracker.rev"
echo "cotracker at $(cat "$CACHE/cotracker.rev")"

CKPT="$CACHE/torch/hub/checkpoints/scaled_offline.pth"
if [ ! -s "$CKPT" ]; then
  curl -fL --retry 3 -o "$CKPT.part" \
    https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
  mv "$CKPT.part" "$CKPT"
fi
echo "cotracker3 offline ckpt: $(du -h "$CKPT" | cut -f1)"

# ---- SAM2.1 (transformers-native repo, so no upstream sam2 package needed) ------
# hiera-small: the ball is a large high-contrast blob, so the extra capacity of
# base-plus/large buys nothing measurable and costs ~3x the propagation time.
SAM2_REPO="${SAM2_REPO:-facebook/sam2.1-hiera-small}"
HF_HOME="$CACHE/hf" "$PROJ/envs/track/bin/python" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    "$SAM2_REPO",
    allow_patterns=["config.json", "*preprocessor_config.json", "processor_config.json",
                    "model.safetensors"],
)
print("sam2 at", p)
PY

echo "done. cache/ is now self-contained; the worker runs with HF_HUB_OFFLINE=1."
