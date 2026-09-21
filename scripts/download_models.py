#!/usr/bin/env python3
"""Download the optional V3 model assets into the paths used by the evaluator.

The repository intentionally does not commit model weights. Use one or more of
--qwen and --p34 on a machine with network access, then run the evaluator with
local-only model loading.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
QWEN_DIR = ROOT / "v3/.models/Qwen3-VL-8B-Instruct"
DINO_DIR = ROOT / "v3/g8/P34/models/grounding-dino-tiny"
SAM_DIR = ROOT / "v3/g8/P34/models/sam2.1-hiera-small"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"


def download_qwen() -> None:
    from huggingface_hub import snapshot_download
    QWEN_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        local_dir=str(QWEN_DIR),
    )
    print(f"Qwen model ready: {QWEN_DIR}")


def download_p34() -> None:
    from huggingface_hub import snapshot_download
    DINO_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="IDEA-Research/grounding-dino-tiny",
        local_dir=str(DINO_DIR),
    )
    SAM_DIR.mkdir(parents=True, exist_ok=True)
    target = SAM_DIR / "sam2.1_hiera_small.pt"
    if not target.exists():
        print(f"Downloading {SAM_URL}")
        urlretrieve(SAM_URL, target)
    print(f"P34 Grounding DINO ready: {DINO_DIR}")
    print(f"P34 SAM 2 checkpoint ready: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", action="store_true", help="download Qwen3-VL-8B-Instruct")
    parser.add_argument("--p34", action="store_true", help="download Grounding DINO Tiny and SAM 2.1 Small")
    args = parser.parse_args()
    if not args.qwen and not args.p34:
        parser.error("choose at least one of --qwen or --p34")
    if args.qwen:
        download_qwen()
    if args.p34:
        download_p34()


if __name__ == "__main__":
    main()
