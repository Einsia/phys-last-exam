#!/usr/bin/env python3
"""Joint two-ball SAM2 + CoTracker worker for P3.

This process intentionally contains no language or vision-language model.  SAM2 is
prompted only with the two fixed frame-zero boxes/centres, and CoTracker receives
fixed point queries.  The output is coordinates, masks-derived radii and visibility;
all physics and scoring live in ``evaluate.py``.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


def _empty(n: int) -> dict[str, np.ndarray]:
    return {
        "xy": np.full((n, 2), np.nan, np.float32),
        "radius": np.full(n, np.nan, np.float32),
        "score": np.full(n, np.nan, np.float32),
    }


def _soft_centroid(prob: np.ndarray, threshold: float = 0.5):
    from scipy import ndimage

    hard = np.asarray(prob > threshold, bool)
    if not hard.any():
        return None
    labels, count = ndimage.label(hard)
    if count > 1:
        sizes = ndimage.sum(hard, labels, index=np.arange(1, count + 1))
        hard = labels == int(np.argmax(sizes) + 1)
    weight = np.asarray(prob, np.float32) * hard
    total = float(weight.sum())
    if total <= 1e-6:
        return None
    ys, xs = np.nonzero(hard)
    w = weight[ys, xs]
    cx = float(np.sum(xs * w) / total)
    cy = float(np.sum(ys * w) / total)
    return cx, cy, float(np.sqrt(total / np.pi)), float(np.mean(w))


def run_sam2(frames: np.ndarray, seeds: list[dict], model_path: str,
             device: str, dtype_name: str) -> dict[str, dict[str, np.ndarray]]:
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    n, h, w = frames.shape[:3]
    result = {s["slot"]: _empty(n) for s in seeds}
    dtype = getattr(torch, dtype_name)
    processor = Sam2VideoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(
        model_path, dtype=dtype, local_files_only=True
    ).to(device).eval()
    # Separate propagation sessions preserve frame-zero identity when two visually
    # identical balls later approach or overlap.  The model weights stay loaded once;
    # only the video session is reset for the second fixed seed.
    for seed in seeds:
        session = processor.init_video_session(
            video=frames, inference_device=device, dtype=dtype
        )
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=1,
            input_points=[[[[float(seed["cx"]), float(seed["cy"])]]]],
            input_labels=[[[1]]],
            original_size=(h, w),
        )
        with torch.inference_mode():
            for output in model.propagate_in_video_iterator(
                inference_session=session, start_frame_idx=0
            ):
                frame_idx = int(output.frame_idx)
                if getattr(output, "object_score_logits", None) is not None:
                    present = float(output.object_score_logits.detach().float().flatten()[0]) > 0.0
                    if not present:
                        continue
                masks = processor.post_process_masks(
                    [output.pred_masks], original_sizes=[[h, w]], binarize=False
                )[0]
                while masks.ndim > 2:
                    masks = masks[0]
                prob = torch.sigmoid(masks.float()).cpu().numpy()
                hit = _soft_centroid(prob)
                if hit is None:
                    continue
                cx, cy, radius, confidence = hit
                slot = seed["slot"]
                result[slot]["xy"][frame_idx] = (cx, cy)
                result[slot]["radius"][frame_idx] = radius
                result[slot]["score"][frame_idx] = confidence
        del session

    del model
    torch.cuda.empty_cache()
    return result


def _ball_queries(seed: dict, ring_points: int = 8) -> np.ndarray:
    cx, cy, radius = float(seed["cx"]), float(seed["cy"]), float(seed["radius"])
    angles = np.arange(ring_points) * 2.0 * np.pi / ring_points
    points = [(cx, cy)]
    points.extend((cx + 0.55 * radius * np.cos(a), cy + 0.55 * radius * np.sin(a))
                  for a in angles)
    return np.asarray(points, np.float32)


def _background_queries(shape: tuple[int, int], seeds: list[dict], step: int = 96):
    h, w = shape
    keep = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if all(np.hypot(x - s["cx"], y - s["cy"]) > 4.0 * s["radius"]
                   for s in seeds):
                keep.append((x, y))
    return np.asarray(keep, np.float32)


def run_cotracker(frames: np.ndarray, seeds: list[dict], checkpoint: str,
                  source: str, device: str):
    import torch

    sys.path.insert(0, source)
    from cotracker.predictor import CoTrackerPredictor

    n, h, w = frames.shape[:3]
    predictor = CoTrackerPredictor(
        checkpoint=checkpoint, offline=True, window_len=60, v2=False
    ).to(device).eval()

    groups = [_ball_queries(seed) for seed in seeds]
    background = _background_queries((h, w), seeds)
    points = np.concatenate(groups + [background], axis=0)
    queries = np.zeros((1, len(points), 3), np.float32)
    queries[0, :, 1:] = points
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    with torch.inference_mode():
        tracks, visible = predictor(video, queries=torch.from_numpy(queries).to(device))
    tracks = tracks[0].float().cpu().numpy()
    visible = visible[0].cpu().numpy().astype(bool)

    output: dict[str, dict[str, np.ndarray]] = {}
    cursor = 0
    for seed, initial in zip(seeds, groups):
        count = len(initial)
        tr = tracks[:, cursor:cursor + count]
        vis = visible[:, cursor:cursor + count]
        cursor += count
        offsets = initial - initial[0]
        slot_out = _empty(n)
        spread0 = float(np.median(np.linalg.norm(initial[1:] - initial[0], axis=1)))
        for frame_idx in range(n):
            good = vis[frame_idx]
            if int(good.sum()) < max(3, count // 2):
                continue
            estimates = tr[frame_idx, good] - offsets[good]
            centre = np.median(estimates, axis=0)
            slot_out["xy"][frame_idx] = centre
            if int(good[1:].sum()) >= 3:
                spread = float(np.median(np.linalg.norm(
                    tr[frame_idx, 1:][good[1:]] - centre, axis=1
                )))
                slot_out["radius"][frame_idx] = float(seed["radius"]) * spread / spread0
            slot_out["score"][frame_idx] = float(good.mean())
        output[seed["slot"]] = slot_out

    bg_tracks = tracks[:, cursor:]
    bg_visible = visible[:, cursor:]
    shift = np.zeros((n, 2), np.float32)
    for frame_idx in range(n):
        good = bg_visible[frame_idx] & bg_visible[0]
        if int(good.sum()) >= 4:
            shift[frame_idx] = np.median(
                bg_tracks[frame_idx, good] - bg_tracks[0, good], axis=0
            )
        elif frame_idx:
            shift[frame_idx] = shift[frame_idx - 1]

    del video, predictor
    torch.cuda.empty_cache()
    return output, shift, int(len(background))


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    frames = np.load(job["frames"])
    seeds = list(job["seeds"])
    n = len(frames)
    device = job.get("device", "cuda")
    arrays: dict[str, np.ndarray] = {}
    meta: dict = {"errors": {}, "backends": []}

    try:
        sam = run_sam2(
            frames, seeds, job["sam2_path"], device, job.get("dtype", "bfloat16")
        )
        for slot, values in sam.items():
            for key, value in values.items():
                arrays[f"sam2_{slot}_{key}"] = value
        meta["backends"].append("sam2")
    except Exception as exc:  # retain CoTracker evidence if SAM2 fails
        meta["errors"]["sam2"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        for seed in seeds:
            for key, value in _empty(n).items():
                arrays[f"sam2_{seed['slot']}_{key}"] = value

    try:
        cot, shift, bg_count = run_cotracker(
            frames, seeds, job["cotracker_checkpoint"], job["cotracker_source"], device
        )
        for slot, values in cot.items():
            for key, value in values.items():
                arrays[f"cotracker_{slot}_{key}"] = value
        arrays["camera_shift"] = shift
        meta["background_points"] = bg_count
        meta["backends"].append("cotracker")
    except Exception as exc:
        meta["errors"]["cotracker"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        arrays["camera_shift"] = np.zeros((n, 2), np.float32)
        for seed in seeds:
            for key, value in _empty(n).items():
                arrays[f"cotracker_{seed['slot']}_{key}"] = value

    arrays["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(job["output"], **arrays)
    return 0 if meta["backends"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
