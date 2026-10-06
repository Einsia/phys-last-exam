#!/usr/bin/env python3
"""Coordinate-only CoTracker worker for the P14 pendulum evaluator.

This module intentionally uses no LLM or VLM.  All queries are fixed frame-zero
points supplied by the versioned task configuration.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


def _empty(n):
    return {"xy": np.full((n, 2), np.nan, np.float32),
            "radius": np.full(n, np.nan, np.float32),
            "score": np.zeros(n, np.float32)}


def _mask_centroid(prob):
    from scipy import ndimage
    hard = np.asarray(prob > .5, bool)
    if not hard.any(): return None
    labels, count = ndimage.label(hard)
    if count > 1:
        sizes = ndimage.sum(hard, labels, index=np.arange(1, count+1))
        hard = labels == int(np.argmax(sizes)+1)
    ys, xs = np.nonzero(hard); weight = np.asarray(prob, np.float32)[ys, xs]
    total = float(weight.sum())
    if total <= 1e-6: return None
    return (float(np.sum(xs*weight)/total), float(np.sum(ys*weight)/total),
            float(np.sqrt(len(xs)/np.pi)), float(np.mean(weight)))


def run_sam2(frames, seeds, model_path, device, dtype_name):
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    n, h, w = frames.shape[:3]; dtype = getattr(torch, dtype_name)
    processor = Sam2VideoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(model_path, dtype=dtype,
                                            local_files_only=True).to(device).eval()
    result = {slot: _empty(n) for slot in ("short", "long")}
    # Separate sessions prevent identity exchange if the two identical bobs approach.
    for slot in ("short", "long"):
        x, y = map(float, seeds[slot]["bob"])
        session = processor.init_video_session(video=frames, inference_device=device,
                                               dtype=dtype)
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=1,
            input_points=[[[[x, y]]]], input_labels=[[[1]]], original_size=(h, w))
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(inference_session=session,
                                                          start_frame_idx=0):
                i = int(out.frame_idx)
                if getattr(out, "object_score_logits", None) is not None and float(out.object_score_logits.detach().float().flatten()[0]) <= 0:
                    continue
                masks = processor.post_process_masks([out.pred_masks],
                                                       original_sizes=[[h,w]],
                                                       binarize=False)[0]
                while masks.ndim > 2: masks = masks[0]
                hit = _mask_centroid(torch.sigmoid(masks.float()).cpu().numpy())
                if hit is not None:
                    result[slot]["xy"][i] = hit[:2]
                    result[slot]["radius"][i] = hit[2]
                    result[slot]["score"][i] = hit[3]
        del session
    del model
    torch.cuda.empty_cache()
    return result


def ring(center, radius, n=12, scale=.58):
    cx, cy = map(float, center)
    a = np.arange(n, dtype=np.float32) * (2 * np.pi / n)
    pts = [(cx, cy)]
    pts += [(cx + scale * radius * np.cos(x), cy + scale * radius * np.sin(x)) for x in a]
    return np.asarray(pts, np.float32)


def fixture(center):
    cx, cy = map(float, center)
    offsets = [(0, 0), (-6, 0), (6, 0), (0, -6), (0, 6),
               (-4, -4), (4, -4), (-4, 4), (4, 4)]
    return np.asarray([(cx + x, cy + y) for x, y in offsets], np.float32)


def background(shape, forbidden, step=105):
    h, w = shape
    pts = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if all(np.hypot(x - q[0], y - q[1]) > q[2] for q in forbidden):
                pts.append((x, y))
    return np.asarray(pts, np.float32)


def robust_group(tracks, visible, initial, radius=None):
    n = len(tracks)
    xy = np.full((n, 2), np.nan, np.float32)
    score = np.zeros(n, np.float32)
    rad = np.full(n, np.nan, np.float32)
    offsets = initial - initial[0]
    spread0 = float(np.median(np.linalg.norm(initial[1:] - initial[0], axis=1))) if len(initial) > 1 else 1.
    for i in range(n):
        good = visible[i]
        if int(good.sum()) < max(4, len(initial) // 3):
            continue
        estimates = tracks[i, good] - offsets[good]
        med = np.median(estimates, axis=0)
        distances = np.linalg.norm(estimates - med, axis=1)
        mad = np.median(distances) + 1e-3
        keep = distances <= max(3., 3.5 * mad)
        if int(keep.sum()) >= 3:
            med = np.median(estimates[keep], axis=0)
        xy[i] = med
        score[i] = float(good.mean())
        if radius is not None and int(good[1:].sum()) >= 4:
            spread = float(np.median(np.linalg.norm(tracks[i, 1:][good[1:]] - med, axis=1)))
            rad[i] = float(radius) * spread / max(spread0, 1e-4)
    return xy, score, rad


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    frames = np.load(job["frames"])
    n, h, w = frames.shape[:3]
    device = job.get("device", "cuda:7")
    arrays = {}
    meta = {"backend": "SAM2+CoTracker", "errors": {}}
    try:
        sam = run_sam2(frames, job["seeds"], job["sam2_path"], device,
                       job.get("sam2_dtype", "bfloat16"))
        for slot, vals in sam.items():
            for key, value in vals.items(): arrays[f"sam2_{slot}_bob_{key}"] = value
    except Exception as exc:
        meta["errors"]["sam2"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
        for slot in ("short", "long"):
            for key, value in _empty(n).items(): arrays[f"sam2_{slot}_bob_{key}"] = value
    try:
        import torch
        sys.path.insert(0, job["cotracker_source"])
        from cotracker.predictor import CoTrackerPredictor

        groups = []
        names = []
        forbidden = []
        for slot in ("short", "long"):
            s = job["seeds"][slot]
            groups.append(ring(s["bob"], float(s["radius"])))
            names.append((slot, "bob", float(s["radius"])))
            groups.append(fixture(s["pivot"]))
            names.append((slot, "pivot", None))
            forbidden.append((float(s["bob"][0]), float(s["bob"][1]), 5 * float(s["radius"])))
            forbidden.append((float(s["pivot"][0]), float(s["pivot"][1]), 55.))
        bg = background((h, w), forbidden)
        points = np.concatenate(groups + [bg], axis=0)
        queries = np.zeros((1, len(points), 3), np.float32)
        queries[0, :, 1:] = points

        predictor = CoTrackerPredictor(checkpoint=job["cotracker_checkpoint"],
                                       offline=True, window_len=60, v2=False).to(device).eval()
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
        with torch.inference_mode():
            tracks, visible = predictor(video, queries=torch.from_numpy(queries).to(device))
        tracks = tracks[0].float().cpu().numpy()
        visible = visible[0].cpu().numpy().astype(bool)
        cursor = 0
        for initial, (slot, kind, radius) in zip(groups, names):
            k = len(initial)
            xy, score, rad = robust_group(tracks[:, cursor:cursor+k], visible[:, cursor:cursor+k], initial, radius)
            cursor += k
            arrays[f"{slot}_{kind}_xy"] = xy
            arrays[f"{slot}_{kind}_score"] = score
            if radius is not None:
                arrays[f"{slot}_{kind}_radius"] = rad

        bgtr, bgvis = tracks[:, cursor:], visible[:, cursor:]
        shift = np.zeros((n, 2), np.float32)
        coverage = np.zeros(n, np.float32)
        for i in range(n):
            good = bgvis[i] & bgvis[0]
            coverage[i] = float(good.mean()) if len(good) else 0.
            if int(good.sum()) >= 5:
                delta = bgtr[i, good] - bgtr[0, good]
                med = np.median(delta, axis=0)
                d = np.linalg.norm(delta - med, axis=1)
                keep = d <= max(2., 3 * (np.median(d) + 1e-3))
                shift[i] = np.median(delta[keep], axis=0)
            elif i:
                shift[i] = shift[i-1]
        arrays["camera_shift"] = shift
        arrays["camera_bg_coverage"] = coverage
        meta.update({"background_points": int(len(bg)), "query_points": int(len(points))})
        del predictor, video
        torch.cuda.empty_cache()
    except Exception as exc:
        meta["errors"]["cotracker"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
        for slot in ("short", "long"):
            for kind in ("bob", "pivot"):
                arrays[f"{slot}_{kind}_xy"] = np.full((n, 2), np.nan, np.float32)
                arrays[f"{slot}_{kind}_score"] = np.zeros(n, np.float32)
            arrays[f"{slot}_bob_radius"] = np.full(n, np.nan, np.float32)
        arrays["camera_shift"] = np.zeros((n, 2), np.float32)
        arrays["camera_bg_coverage"] = np.zeros(n, np.float32)
    arrays["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(job["output"], **arrays)
    return 0 if len(meta["errors"]) < 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
