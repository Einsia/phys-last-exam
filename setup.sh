#!/usr/bin/env bash
# Install the recommended Linux/CUDA environment and all evaluation models.
set -euo pipefail

case "${1:-}" in
    -h|--help)
        echo "Usage: bash setup.sh [--skip-models]"
        echo "Run inside a Python 3.12 Conda environment or virtual environment."
        echo "Installs dependencies and downloads models into ./models/."
        echo "--skip-models installs dependencies only."
        exit 0 ;;
    ""|--skip-models) ;;
    *) echo "Unknown option: $1. Use --help." >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: bash setup.sh [--skip-models]" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v git >/dev/null || { echo "Install Git first." >&2; exit 1; }
python - <<'PY'
import os
import sys
if sys.version_info[:2] != (3, 12):
    sys.exit("Use Python 3.12: conda create -n phys-last-exam python=3.12 -y && conda activate phys-last-exam")
if sys.prefix == sys.base_prefix and not os.environ.get("CONDA_PREFIX"):
    sys.exit("Activate your phys-last-exam Conda environment or a Python 3.12 virtual environment first.")
if sys.platform != "linux":
    sys.exit("This installer targets Linux with an NVIDIA GPU.")
PY

echo "[1/3] Installing Python packages..."
python -m pip install --upgrade pip setuptools wheel
# Official CUDA 12.8 pair: https://pytorch.org/get-started/previous-versions/
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r requirements.txt transformers==5.14.1 huggingface-hub==1.27.0
python -m pip check
python - <<'PY'
import torch
import cv2, av, scipy, matplotlib, yaml, sam2, cotracker
from transformers import AutoModelForImageTextToText, AutoProcessor, Sam2VideoModel, Sam2VideoProcessor
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check that nvidia-smi works and update the NVIDIA driver for CUDA 12.8.")
print("CUDA ready:", torch.cuda.get_device_name(0))
PY

if [[ "${1:-}" == "--skip-models" ]]; then
    echo "Dependencies installed. Model downloads were skipped."
    exit 0
fi

echo "[2/3] Downloading evaluation models..."
model_dir="$(python -c 'import os; from pathlib import Path; print(Path(os.environ.get("FINAL_MODELS_DIR", "models")).expanduser().resolve())')"
mkdir -p -- "$model_dir"
# Pin upstream snapshots so rerunning setup does not silently change the models.
hf download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --local-dir "$model_dir/Qwen3.6-27B"
hf download IDEA-Research/grounding-dino-tiny --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 --local-dir "$model_dir/grounding-dino-tiny" --exclude '*.bin'
hf download facebook/sam2.1-hiera-small --revision ee5bba1d82bb8749febdf90f45e84b687142ba03 --local-dir "$model_dir/sam2.1-hiera-small" --include '*.pt'
hf download facebook/sam2.1-hiera-small --revision ee5bba1d82bb8749febdf90f45e84b687142ba03 --local-dir "$model_dir/sam2.1-hiera-small-transformers" --include '*.json' --include '*.safetensors'
hf download facebook/sam2.1-hiera-large --revision 665f8e2ad61cf5f53d65644ff27c8ee525124610 --local-dir "$model_dir/sam2.1-hiera-large-transformers" --include '*.json' --include '*.safetensors'
hf download facebook/cotracker3 scaled_offline.pth --revision bf55ea50d4390e1820a267f131cd6587240fb2c5 --local-dir "$model_dir/cotracker3"

echo "[3/3] Preparing the CoTracker source used by the task evaluators..."
cotracker_source="$model_dir/cotracker3/source"
cotracker_revision=82e02e8029753ad4ef13cf06be7f4fc5facdda4d
if [[ ! -e "$cotracker_source" ]]; then
    git clone https://github.com/facebookresearch/co-tracker.git "$cotracker_source"
    git -C "$cotracker_source" checkout --detach "$cotracker_revision"
elif [[ "$(git -C "$cotracker_source" rev-parse HEAD)" != "$cotracker_revision" ]]; then
    echo "CoTracker source already exists at a different revision: $cotracker_source" >&2
    echo "Move that directory aside, then rerun setup." >&2
    exit 1
fi
echo "Setup complete. Put your generated videos in videos/ and follow the README to evaluate them."
