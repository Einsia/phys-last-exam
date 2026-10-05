#!/usr/bin/env python
"""Run HunyuanVideo-1.5 I2V through the official repository entry point."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _bool(value: str) -> str:
    return str(value).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", default="720p", choices=("480p", "720p"))
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", default="false")
    parser.add_argument("--rewrite", default="false")
    parser.add_argument("--offloading", default="true")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--overlap-group-offloading", default="true")
    parser.add_argument("--group-offloading", default=None)
    parser.add_argument("--group_offloading", dest="group_offloading", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--distributed", action="store_true",
                        help="Launch the official sampler with torchrun")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    args = parser.parse_args()

    missing_modules = [name for name in ("loguru", "torch", "diffusers", "transformers", "einops", "imageio")
                       if importlib.util.find_spec(name) is None]
    if missing_modules:
        parser.error("HunyuanVideo-1.5 runtime is missing dependencies: " + ", ".join(missing_modules))

    if not args.repo.is_dir():
        parser.error(f"HunyuanVideo source directory not found: {args.repo}")
    if not args.model_path.is_dir():
        parser.error(f"HunyuanVideo checkpoint directory not found: {args.model_path}")
    if not args.image.is_file():
        parser.error(f"first-frame image not found: {args.image}")
    required = [args.model_path / "vae", args.model_path / "scheduler",
                args.model_path / "text_encoder" / "llm",
                args.model_path / "text_encoder" / "byt5-small",
                args.model_path / "text_encoder" / "Glyph-SDXL-v2",
                args.model_path / "vision_encoder" / "siglip",
                args.model_path / "transformer" / f"{args.resolution}_i2v"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error(
            "HunyuanVideo-1.5 checkpoint is missing: " + ", ".join(missing)
            + ". The official snapshot keeps these components separate; see "
            f"{args.repo / 'checkpoints-download.md'} for Qwen2.5-VL-7B-Instruct, google/byt5-small, "
            "AI-ModelScope/Glyph-SDXL-v2, and FLUX.1-Redux-dev."
        )
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be 4k+1 and at least 5")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    child = [sys.executable, str(args.repo / "generate.py"),
          "--prompt", args.prompt, "--resolution", args.resolution,
          "--model_path", str(args.model_path), "--aspect_ratio", args.aspect_ratio,
          "--video_length", str(args.num_frames), "--num_inference_steps", str(args.steps),
          "--seed", str(args.seed), "--image_path", str(args.image),
          "--output_path", str(args.output), "--sr", _bool(args.sr),
          "--rewrite", _bool(args.rewrite), "--offloading", _bool(args.offloading),
          "--dtype", args.dtype, "--overlap_group_offloading", _bool(args.overlap_group_offloading)]
    if args.distributed:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.nproc_per_node}", str(args.repo / "generate.py"), *child[2:]]
    else:
        cmd = child
    if args.group_offloading is not None:
        cmd += ["--group_offloading", _bool(args.group_offloading)]
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    subprocess.run(cmd, cwd=args.repo, env=env, check=True)
    if int(os.environ.get("RANK", "0")) == 0 and not args.output.is_file():
        raise RuntimeError(f"HunyuanVideo-1.5 did not write output: {args.output}")
    print(f"Saved video to {args.output} ({args.output.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
