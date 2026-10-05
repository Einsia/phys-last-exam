#!/usr/bin/env python
"""Run VBVR-Wan2.2 from its local Diffusers image-to-video snapshot."""

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
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default=None,
                        help="Accelerate device map (for example balanced) across visible GPUs")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        parser.error(f"model checkpoint directory not found: {args.model_path}")
    if not args.image.is_file():
        parser.error(f"first-frame image not found: {args.image}")
    required = [args.model_path / "model_index.json", args.model_path / "transformer",
                args.model_path / "transformer_2", args.model_path / "text_encoder",
                args.model_path / "tokenizer", args.model_path / "vae"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("VBVR-Wan2.2 checkpoint is incomplete; missing: " + ", ".join(missing))
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be 4k+1 and at least 5")

    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    vae = AutoencoderKLWan.from_pretrained(
        str(args.model_path), subfolder="vae", torch_dtype=torch.float32
    )
    load_kwargs = {"vae": vae, "torch_dtype": torch.bfloat16}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    pipe = WanImageToVideoPipeline.from_pretrained(str(args.model_path), **load_kwargs)
    if hasattr(pipe, "scheduler"):
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=5.0
        )
    if hasattr(pipe, "enable_model_cpu_offload"):
        if args.device_map is None:
            pipe.enable_model_cpu_offload()
    else:
        if args.device_map is None:
            pipe.to("cuda")
    if args.device_map:
        placement = getattr(pipe, "hf_device_map", {})
        gpu_devices = [torch.device(d) for d in placement.values()
                       if str(d).startswith("cuda")]
        vae_device = gpu_devices[0] if gpu_devices else torch.device("cuda:0")
        pipe.vae = pipe.vae.to(vae_device)
    image = load_image(str(args.image)).convert("RGB").resize((args.width, args.height))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt="",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(args.output), fps=args.fps, macro_block_size=1)
    if not args.output.is_file():
        raise RuntimeError(f"VBVR-Wan2.2 did not write output: {args.output}")
    print(f"Saved video to {args.output} ({args.output.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
