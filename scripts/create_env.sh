#!/usr/bin/env bash
# Create the physbench env: tracking + metrics only, no torch.
# The VDM lives in its own env (minimax-h3/envs/mh3) and is called as a subprocess,
# so these two dependency sets never have to agree.
set -euo pipefail

PROJ="/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/physical-bench"
ENV_DIR="$PROJ/envs/physbench"

mkdir -p "$PROJ/envs"

conda create -y -p "$ENV_DIR" -c conda-forge --override-channels \
  python=3.12 \
  numpy=2.2.6 \
  scipy=1.15.2 \
  opencv=4.11.0 \
  av=14.4.0 \
  imageio=2.37.0 \
  imageio-ffmpeg=0.6.0 \
  pillow=11.1.0 \
  matplotlib=3.10.1 \
  pyyaml=6.0.2 \
  pandas=2.2.3 \
  ffmpeg=7.1.1

echo "physbench created at $ENV_DIR"
"$ENV_DIR/bin/python" -c "import numpy, scipy, cv2, av, PIL, matplotlib, yaml, pandas; print('imports ok')"
