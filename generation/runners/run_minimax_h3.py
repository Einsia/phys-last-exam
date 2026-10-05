#!/usr/bin/env python
"""MiniMax-H3 image -> video+audio (i2va), four-card full bf16.

This is the fl2va workflow driven from a single first frame, split across four cards so
every component stays resident. The two-card recipe in t2va.py/fl2va.py puts the
transformer and both VAEs on one card (72.1 GB of weights), which leaves ~5 GB for
activations after the 3 GB reserve margin, so auto offload evicts the VAEs during denoise
and reloads them to decode. One component per card removes that:

  cuda:0  text_encoder  63 GB   Qwen3-VL conditioner (+ the cheap keyframe resize)
  cuda:1  transformer   62 GB   ~18 GB left for activations, 3x the two-card recipe
  cuda:2  vae          9.8 GB   keyframe encode *and* video decode, never evicted
  cuda:3  audio_vae    578 MB   stereo decode

Each ComponentsManager binds exactly one execution device, so the split is one manager per
card. Hooks only evict each other when they share an execution device, so nothing here ever
offloads to host RAM.

The state threads through the five calls: resize+conditioning -> keyframe encode -> denoise
-> video decode / audio decode. Omit `output=` to get the state back, pass it to get a dict.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_minimax_h3.py \\
    --model /path/to/MiniMax-H3 --image f.jpg --prompt "..." --out video.mp4
"""

import argparse
import os
import time
from pathlib import Path

import torch

from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

PROJ = str(Path.cwd())

# Logical index -> what lives there. Logical, so CUDA_VISIBLE_DEVICES picks the physical cards.
CARD_TEXT, CARD_DIT, CARD_VAE, CARD_AUDIO = 0, 1, 2, 3


def preload_flash3_from_cache():
    """Make `_flash_3_hub` usable under HF_HUB_OFFLINE=1.

    `kernels.get_kernel` resolves the backend's `version=1` through the Hub, and its offline
    branch asks `snapshot_download` for the *whole* repo before narrowing to this system's
    build variant. Only one of the 57 variants is ever cached, so the completeness check
    raises `IncompleteSnapshotError` and diffusers falls back to the default backend --
    quietly, since `set_attention_backend` only logs. Resolving the variant off the cache
    directory and pre-filling the registry skips `get_kernel` altogether, because
    `_maybe_download_kernel_for_backend` returns early once the three fns are set.

    Returns True if the kernel was injected, False to leave the normal (online) path alone.
    """
    try:
        import huggingface_hub.constants as hfc
        from kernels.utils import _import_from_path, get_variants_local, resolve_variant

        from diffusers.models.attention_dispatch import (
            _HUB_KERNELS_REGISTRY,
            AttentionBackendName,
            _resolve_kernel_attr,
        )

        config = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
        if config.kernel_fn is not None:
            return True

        repo_dir = Path(hfc.HF_HUB_CACHE) / ("kernels--" + config.repo_id.replace("/", "--"))
        snapshots = sorted(
            (repo_dir / "snapshots").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        variant, _ = resolve_variant(get_variants_local(snapshots[0] / "build"), None)
        if variant is None:
            return False

        module = _import_from_path(snapshots[0] / "build" / variant.variant_str)
        config.kernel_fn = _resolve_kernel_attr(module, config.function_attr)
        config.wrapped_forward_fn = _resolve_kernel_attr(module, config.wrapped_forward_attr)
        config.wrapped_backward_fn = _resolve_kernel_attr(module, config.wrapped_backward_attr)
        print(f"flash3: loaded {variant.variant_str} from local cache")
        return True
    except Exception as e:
        print(f"flash3: local cache preload skipped ({type(e).__name__}: {e})")
        return False


def parse_args():
    p = argparse.ArgumentParser(description="MiniMax-H3 image->video+audio on four cards")
    p.add_argument("--image", required=True, help="first frame (path or URL)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--last-image", default=None, help="optional final frame")
    p.add_argument("--out", default=os.path.join(PROJ, "outputs", "i2va.mp4"))
    p.add_argument("--num-frames", type=int, default=124,
                   help="snaps up to next 17*n+5; 124..345 legal (5.17..14.38 s at 24 fps)")
    p.add_argument("--height", type=int, default=None, help="multiple of 32; default follows the image")
    p.add_argument("--width", type=int, default=None, help="multiple of 32; default follows the image")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", required=True)
    p.add_argument("--flash3", action="store_true", help="_flash_3_hub backend, ~3x faster on Hopper")
    args = p.parse_args()
    if (args.height is None) != (args.width is None):
        p.error("--height and --width must be passed together, or neither")
    for name, value in (("--height", args.height), ("--width", args.width)):
        if value is not None and value % 32:
            p.error(f"{name} must be a multiple of 32, got {value}")
    return args


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n = torch.cuda.device_count()
    if n < 4:
        raise SystemExit(
            f"this recipe needs 4 visible GPUs, found {n}. "
            f"Set CUDA_VISIBLE_DEVICES=0,1,2,3 (currently {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})."
        )
    print(f"visible GPUs: {n}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    for i in range(4):
        print(f"  cuda:{i} {torch.cuda.get_device_properties(i).name}")

    t0 = time.time()
    workflow = ModularPipeline.from_pretrained(args.model).blocks.get_workflow("fl2va")

    # Pop in dependency order. before_encode produces `keyframes`, which both the text encoder
    # and the keyframe VAE encoder read, so it has to travel with the first pipeline rather than
    # stay behind with denoise.
    resize_block = workflow.sub_blocks.pop("before_encode")
    text_block = workflow.sub_blocks.pop("text_encoder")
    vae_encode_block = workflow.sub_blocks.pop("vae_encoder")
    video_decode_block = workflow.sub_blocks.pop("decode.video")
    audio_decode_block = workflow.sub_blocks.pop("decode.audio")

    def stage(blocks, device, label):
        """One sub-pipeline pinned to one card. Returns (pipeline, manager)."""
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=f"cuda:{device}")
        built = SequentialPipelineBlocks.from_blocks_dict(blocks) if isinstance(blocks, dict) else blocks
        pipe = built.init_pipeline(args.model, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
        print(f"  {label} -> cuda:{device}")
        return pipe, manager

    print("loading components ...")
    conditioner, _ = stage(
        {"before_encode": resize_block, "text_encoder": text_block}, CARD_TEXT, "resize + conditioner"
    )
    # vae_encoder and decode.video share the video VAE, so they share one manager and one card;
    # the weights load once and serve both ends of the run.
    vae_manager = ComponentsManager()
    vae_manager.enable_auto_cpu_offload(device=f"cuda:{CARD_VAE}")
    keyframe_encoder = vae_encode_block.init_pipeline(args.model, components_manager=vae_manager)
    keyframe_encoder.load_components(dtype=torch.bfloat16)
    video_decoder = video_decode_block.init_pipeline(args.model, components_manager=vae_manager)
    # Hand over the VAE the encoder already loaded rather than calling load_components again --
    # a second load would put a duplicate 9.8 GB copy on the card and warn about the load_id.
    # video_processor has no weights, so it still comes from the spec.
    video_decoder.load_components(names=["video_processor"], dtype=torch.bfloat16)
    video_decoder.update_components(vae=keyframe_encoder.vae)
    print(f"  keyframe encode + video decode -> cuda:{CARD_VAE}")

    denoiser, _ = stage(workflow, CARD_DIT, "transformer")
    audio_decoder, _ = stage(audio_decode_block, CARD_AUDIO, "audio decode")

    if args.flash3:
        preload_flash3_from_cache()
        try:
            denoiser.transformer.set_attention_backend("_flash_3_hub")
            print("attention backend: _flash_3_hub")
        except Exception as e:
            print(f"flash3 unavailable ({type(e).__name__}: {e}); default backend")
    print(f"load took {time.time() - t0:.1f} s")

    print(f"\nprompt: {args.prompt}")
    print(f"image : {args.image}")

    t1 = time.time()

    cond_call = {"prompt": args.prompt, "image": load_image(args.image)}
    if args.last_image:
        cond_call["last_image"] = load_image(args.last_image)
    if args.height:
        cond_call["height"], cond_call["width"] = args.height, args.width

    state = conditioner(**cond_call)
    state = keyframe_encoder(state=state)

    denoise_call = dict(
        state=state,
        num_frames=args.num_frames,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.steps is not None:
        denoise_call["num_inference_steps"] = args.steps
    state = denoiser(**denoise_call)

    # The decode blocks scale the latents *before* calling into the VAE, so the offload hook -- which
    # only moves a module's own forward args -- never sees them. Denoise leaves both tensors on the
    # transformer's card, so relocate them to whichever card is about to decode them.
    def move(key, device):
        value = state.get(key)
        if torch.is_tensor(value):
            state.set(key, value.to(f"cuda:{device}"))

    move("latents", CARD_VAE)
    video = video_decoder(state=state, output=["videos"])
    move("audio_latents", CARD_AUDIO)
    audio = audio_decoder(state=state, output=["audio", "sampling_rate"])
    gen_s = time.time() - t1

    encode_video(
        video["videos"][0],
        fps=24,
        output_path=args.out,
        audio=audio["audio"][0],
        audio_sample_rate=audio["sampling_rate"],
    )

    print(f"\ngenerate took {gen_s:.1f} s")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024**2:.1f} MB)")
    print(f"audio sample rate: {audio['sampling_rate']} Hz")
    for i in range(4):
        print(f"peak VRAM cuda:{i}: {torch.cuda.max_memory_allocated(i) / 1024**3:.1f} GiB")


if __name__ == "__main__":
    main()
