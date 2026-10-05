#!/usr/bin/env python
"""Run Cosmos3-Super-Image2Video from a local Diffusers checkpoint.

This small process boundary keeps Cosmos3's torch/diffusers dependencies out of
the benchmark driver.  The command intentionally mirrors the public Diffusers
example while accepting the benchmark's stable image/prompt/output contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path


def _prompt_payload(prompt: str, *, height: int, width: int, fps: int,
                    num_frames: int) -> str:
    """Cosmos3 handles structured prompts best; preserve JSON supplied by callers."""
    try:
        parsed = json.loads(prompt)
    except (TypeError, json.JSONDecodeError):
        parsed = {
            "temporal_caption": prompt.strip(),
            "duration": f"{num_frames / fps:g}s",
            "fps": float(fps),
            "resolution": {"H": height, "W": width},
            "aspect_ratio": f"{width // math.gcd(width, height)},{height // math.gcd(width, height)}",
        }
    if not isinstance(parsed, dict):
        raise ValueError("Cosmos3 prompt JSON must be an object")
    parsed.setdefault("duration", f"{num_frames / fps:g}s")
    parsed.setdefault("fps", float(fps))
    parsed.setdefault("resolution", {"H": height, "W": width})
    return json.dumps(parsed, ensure_ascii=False)


def _parse_gpu_memory_gib(value: str | None) -> int | None:
    """Parse the per-GPU budget used by the explicit layer sharder."""
    if value is None:
        return None
    try:
        gib = int(value)
    except ValueError as exc:
        raise ValueError(f"--gpu-memory-gib must be an integer, got {value!r}") from exc
    if gib <= 0:
        raise ValueError(f"--gpu-memory-gib must be positive, got {gib}")
    return gib


def _parse_max_memory(value: str | None, device_count: int) -> dict[int, str] | None:
    """Build an Accelerate max-memory map for visible logical CUDA devices."""
    gib = _parse_gpu_memory_gib(value)
    if gib is None:
        return None
    if device_count < 1:
        raise RuntimeError("CUDA is required for transformer-parallel Cosmos3 inference")
    return {index: f"{gib}GiB" for index in range(device_count)}


def _module_nbytes(module, *, dtype_bytes: int = 2, fp32: bool = False) -> int:
    """Estimate loaded parameter/buffer bytes for a meta-initialized module."""
    element_bytes = 4 if fp32 else dtype_bytes
    return sum(parameter.numel() * element_bytes for parameter in module.parameters()) + sum(
        buffer.numel() * element_bytes for buffer in module.buffers()
    )


def _layer_shard_map(
    model_path: Path,
    device_count: int,
    transformer_cls,
    *,
    gpu_memory_gib: int | None = None,
) -> dict[str, int]:
    """Map decoder layers contiguously, balancing estimated BF16 bytes.

    The root module is deliberately omitted. Mapping it would make Accelerate
    attach a root hook that eagerly moves every child to GPU 0. Shared
    projections stay on logical GPU 0; only the sequential decoder layers are
    split across the visible devices.
    """
    config = transformer_cls.load_config(str(model_path / "transformer"))
    num_layers = int(config.get("num_hidden_layers", config.get("num_layers", 36)))
    if num_layers < 1:
        raise RuntimeError(f"Cosmos3 transformer config has invalid layer count: {num_layers}")
    # Build the list of direct children from a meta-only instance. Do not map the
    # top-level ``""`` module: Accelerate would attach a root hook that eagerly
    # moves every child to GPU 0, defeating the layer map.
    from accelerate import init_empty_weights

    with init_empty_weights():
        empty_transformer = transformer_cls.from_config(config)
    mapping: dict[str, int] = {
        name: 0 for name, _ in empty_transformer.named_children() if name != "layers"
    }
    shared_bytes = sum(
        _module_nbytes(module, fp32=(name == "time_embedder"))
        for name, module in empty_transformer.named_children()
        if name != "layers"
    )
    layer_bytes = [
        _module_nbytes(empty_transformer.layers[layer_index])
        for layer_index in range(num_layers)
    ]
    del empty_transformer

    if device_count > num_layers:
        raise RuntimeError(
            f"Cosmos3 has {num_layers} decoder layers but {device_count} GPUs were requested; "
            "use no more than one GPU per decoder layer."
        )

    # Find contiguous boundaries that minimize the largest static allocation.
    # This is a small O(G * L^2) dynamic program (G<=8 and L=64 here), and is
    # more balanced than a greedy threshold when the first device owns shared
    # projections in addition to its decoder layers.
    prefix = [0]
    for layer_size in layer_bytes:
        prefix.append(prefix[-1] + layer_size)
    costs: list[list[int | None]] = [[None] * (num_layers + 1) for _ in range(device_count + 1)]
    boundaries: list[list[int | None]] = [[None] * (num_layers + 1) for _ in range(device_count + 1)]
    for end in range(1, num_layers + 1):
        costs[1][end] = shared_bytes + prefix[end]
    for groups in range(2, device_count + 1):
        for end in range(groups, num_layers + 1):
            best_cost: int | None = None
            best_start: int | None = None
            for start in range(groups - 1, end):
                previous = costs[groups - 1][start]
                if previous is None:
                    continue
                candidate = max(previous, prefix[end] - prefix[start])
                if best_cost is None or candidate < best_cost:
                    best_cost, best_start = candidate, start
            costs[groups][end] = best_cost
            boundaries[groups][end] = best_start

    ends = [num_layers]
    end = num_layers
    for groups in range(device_count, 1, -1):
        start = boundaries[groups][end]
        if start is None:
            raise RuntimeError("Cosmos3 layer sharder could not construct non-empty GPU partitions")
        ends.append(start)
        end = start
    ends.append(0)
    ends.reverse()
    layer_to_device: list[int] = []
    for device_index, (start, end) in enumerate(zip(ends, ends[1:])):
        layer_to_device.extend([device_index] * (end - start))
    assigned_bytes = [shared_bytes] + [0] * (device_count - 1)
    for layer_index, device_index in enumerate(layer_to_device):
        mapping[f"layers.{layer_index}"] = device_index
        assigned_bytes[device_index] += layer_bytes[layer_index]

    if gpu_memory_gib is not None:
        budget_bytes = gpu_memory_gib * 1024**3
        over_budget = {
            index: f"{size / 1024**3:.2f}GiB"
            for index, size in enumerate(assigned_bytes)
            if size > budget_bytes
        }
        if over_budget:
            raise RuntimeError(
                "Cosmos3 explicit layer map exceeds the configured static GPU budget "
                f"({gpu_memory_gib}GiB per visible GPU): {over_budget}. "
                "Use more GPUs or increase --gpu-memory-gib."
            )
    return mapping


def _configure_attention_backend(transformer, requested: str, torch) -> str:
    """Select FlashAttention when available, with a native SDPA fallback."""
    requested = requested.lower()
    if requested not in {"auto", "native", "flash"}:
        raise ValueError(f"--attention-backend must be auto, native, or flash; got {requested!r}")
    selected = requested
    if selected == "auto":
        selected = "flash" if importlib.util.find_spec("flash_attn") else "native"
    try:
        if selected == "flash":
            # Import success alone does not guarantee that the wheel matches the
            # node's CUDA driver/SM. A tiny real kernel call makes auto mode
            # fail over before the expensive model starts denoising.
            from flash_attn import flash_attn_func

            with torch.inference_mode():
                query = torch.zeros((1, 8, 4, 64), device="cuda", dtype=torch.bfloat16)
                key = torch.zeros((1, 8, 2, 64), device="cuda", dtype=torch.bfloat16)
                flash_attn_func(query, key, key, dropout_p=0.0, causal=False)
                torch.cuda.synchronize()
        transformer.set_attention_backend(selected)
    except Exception as exc:
        if requested == "auto" and selected == "flash":
            print(f"Cosmos3 FlashAttention unavailable ({exc}); falling back to native SDPA", flush=True)
            transformer.set_attention_backend("native")
            selected = "native"
        else:
            raise RuntimeError(f"Unable to configure Cosmos3 attention backend {selected!r}: {exc}") from exc
    return selected


def _print_cuda_memory(torch, device_count: int, *, label: str) -> None:
    """Print per-device allocator stats so CPU offload is easy to diagnose."""
    stats = []
    for index in range(device_count):
        stats.append(
            f"cuda:{index} allocated={torch.cuda.memory_allocated(index) / 1024**3:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved(index) / 1024**3:.2f}GiB "
            f"peak={torch.cuda.max_memory_allocated(index) / 1024**3:.2f}GiB"
        )
    print(f"Cosmos3 {label}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}; "
          + "; ".join(stats), flush=True)


def _load_pipeline(args, torch, Cosmos3OmniPipeline, Cosmos3OmniTransformer):
    """Load the pipeline, optionally sharding Cosmos3's Transformer by decoder layer.

    Diffusers pipeline ``device_map=balanced`` only places complete components. The
    64B Cosmos3 Transformer is larger than one H100, so that mode silently puts it
    on CPU. ModelMixin's loader can instead split the Transformer's decoder layers
    across all visible CUDA devices; the rest of the pipeline stays on logical GPU 0.
    """
    dtype = torch.bfloat16
    if args.device_map not in {"transformer-balanced", "transformer-auto"}:
        pipe = Cosmos3OmniPipeline.from_pretrained(
            str(args.model_path),
            torch_dtype=dtype,
            device_map=args.device_map,
            enable_safety_checker=not args.no_safety_checker,
        )
        return pipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --device-map transformer-balanced")
    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError(
            f"--device-map {args.device_map} needs at least 2 visible GPUs; found {device_count}"
        )

    # Diffusers' pipeline-level map cannot split this component. An automatic
    # model-level map also moves shared projections to later devices, while the
    # transformer forward expects all packed index tensors on one device. Keep
    # those shared modules together and shard only the decoder layer list.
    gpu_memory_gib = _parse_gpu_memory_gib(args.gpu_memory_gib)
    transformer_map = _layer_shard_map(
        args.model_path,
        device_count,
        Cosmos3OmniTransformer,
        gpu_memory_gib=gpu_memory_gib,
    )
    max_memory = _parse_max_memory(args.gpu_memory_gib, device_count)
    transformer = Cosmos3OmniTransformer.from_pretrained(
        str(args.model_path),
        subfolder="transformer",
        torch_dtype=dtype,
        device_map=transformer_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    placement = getattr(transformer, "hf_device_map", {})
    cpu_modules = {
        name: str(device)
        for name, device in placement.items()
        if str(device).lower() in {"cpu", "disk"} or str(device).startswith("disk")
    }
    if cpu_modules:
        raise RuntimeError(
            "Cosmos3 transformer parallel map spilled modules to CPU/disk: "
            f"{cpu_modules}. Increase --gpu-memory-gib or free the selected GPUs."
        )
    devices = sorted({str(device) for device in placement.values()})
    expected_devices = {str(index) for index in range(device_count)}
    if set(devices) != expected_devices:
        raise RuntimeError(
            f"Cosmos3 transformer map did not use every visible GPU: expected {sorted(expected_devices)}, "
            f"got {devices}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    counts = {device: sum(str(value) == device for value in placement.values()) for device in devices}
    print(
        f"Cosmos3 transformer device map: {len(placement)} entries across {devices}; "
        f"module counts={counts}; logical-to-physical={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    selected_backend = _configure_attention_backend(transformer, args.attention_backend, torch)
    print(f"Cosmos3 attention backend: {selected_backend}", flush=True)

    # Passing the already-dispatched transformer prevents pipeline-level
    # component balancing from moving it back to CPU. VAE and all caller tensors
    # use logical cuda:0, which is physical GPU 4 when the launcher exposes 4-7.
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(args.model_path),
        transformer=transformer,
        torch_dtype=dtype,
        device_map=None,
        enable_safety_checker=not args.no_safety_checker,
    )
    pipe.vae.to("cuda:0")
    _print_cuda_memory(torch, device_count, label="after load")
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--device-map",
        default="transformer-balanced",
        help="pipeline map (balanced/cuda) or layer-shard the Transformer (transformer-balanced/transformer-auto)",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        default=os.environ.get("COSMOS3_GPU_MEMORY_GIB", "74"),
        help="per-visible-GPU static parameter budget for Transformer sharding; leave headroom for activations",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "native", "flash"),
        default=os.environ.get("COSMOS3_ATTENTION_BACKEND", "auto"),
        help="attention kernel: auto prefers flash_attn and falls back to PyTorch SDPA",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=189)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--add-resolution-template", action="store_true")
    parser.add_argument("--add-duration-template", action="store_true")
    parser.add_argument("--no-safety-checker", action="store_true")
    args = parser.parse_args()

    if not args.model_path.is_dir():
        parser.error(f"model checkpoint directory not found: {args.model_path}")
    if not args.image.is_file():
        parser.error(f"first-frame image not found: {args.image}")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be 4k+1 and at least 5")

    import torch
    from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler
    from diffusers.utils import export_to_video, load_image

    load_started = time.perf_counter()
    pipe = _load_pipeline(args, torch, Cosmos3OmniPipeline, Cosmos3OmniTransformer)
    print(f"Cosmos3 load completed in {time.perf_counter() - load_started:.1f}s", flush=True)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    prompt = _prompt_payload(
        args.prompt, height=args.height, width=args.width,
        fps=args.fps, num_frames=args.num_frames,
    )
    inference_started = time.perf_counter()
    result = pipe(
        prompt=prompt,
        image=load_image(str(args.image)),
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=float(args.fps),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        add_resolution_template=args.add_resolution_template,
        add_duration_template=args.add_duration_template,
        generator=generator,
    )
    inference_seconds = time.perf_counter() - inference_started
    print(
        f"Cosmos3 inference completed in {inference_seconds:.1f}s "
        f"({inference_seconds / args.steps:.2f}s/step, {args.steps} steps)",
        flush=True,
    )
    _print_cuda_memory(torch, torch.cuda.device_count(), label="after inference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.video, str(args.output), fps=args.fps, quality=7, macro_block_size=1)
    if not args.output.is_file():
        raise RuntimeError(f"Cosmos3 did not write output: {args.output}")
    print(f"Saved video to {args.output} ({args.output.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
