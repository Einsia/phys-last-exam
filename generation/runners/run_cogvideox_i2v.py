
#!/usr/bin/env python
"""Run CogVideoX1.5-5B-I2V from a local Diffusers snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default=None,
                        help="Accelerate device map (for example balanced) across visible GPUs")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Keep the complete pipeline on CUDA; never use CPU offload")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        parser.error(f"model checkpoint directory not found: {args.model_path}")
    if not args.image.is_file():
        parser.error(f"first-frame image not found: {args.image}")
    required = [args.model_path / "model_index.json", args.model_path / "transformer",
                args.model_path / "text_encoder", args.model_path / "tokenizer",
                args.model_path / "vae", args.model_path / "scheduler"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("CogVideoX1.5 checkpoint is incomplete; missing: " + ", ".join(missing))
    if args.num_frames < 17 or (args.num_frames - 1) % 16:
        parser.error("--num-frames must be 16k+1 and at least 17 for CogVideoX1.5")

    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(str(args.model_path), **load_kwargs)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    if args.gpu_only:
        pipe.to("cuda")
    elif hasattr(pipe, "enable_model_cpu_offload") and args.device_map is None:
        pipe.enable_model_cpu_offload()
    elif args.device_map is None:
        pipe.to("cuda")
    if args.gpu_only and args.device_map is not None:
        raise RuntimeError("--gpu-only cannot be combined with --device-map")
    if args.device_map:
        placement = getattr(pipe, "hf_device_map", {})
        host = [name for name, device in placement.items()
                if str(device).lower() in {"cpu", "disk"} or str(device).lower().startswith(("cpu", "disk"))]
        if host:
            raise RuntimeError("GPU-only CogVideoX placement failed; modules on CPU/disk: " + ", ".join(host))
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        image=load_image(str(args.image)),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        use_dynamic_cfg=True,
        generator=generator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(args.output), fps=args.fps, macro_block_size=1)
    if not args.output.is_file():
        raise RuntimeError(f"CogVideoX did not write output: {args.output}")
    print(f"Saved video to {args.output} ({args.output.stat().st_size / 1024**2:.1f} MB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
