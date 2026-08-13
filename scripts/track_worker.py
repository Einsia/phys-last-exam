#!/usr/bin/env python
"""Neural tracking backends, run as a subprocess in envs/track.

Same argument as physbench/vdm.py: torch lives in its own env and is called, not
imported, so the measurement env (numpy/opencv, no torch) and the tracker env never
have to agree on a dependency set.

Two backends, both seeded with nothing but the ball centre and radius from the first
frame's spec:

  sam2       SAM2.1 video propagation from a single positive click. Gives a mask per
             frame, so position AND apparent size come from the model rather than
             from a threshold on colour distance.
  cotracker  CoTracker3 offline. Tracks a ring of points on the ball disc; the ball
             centre is their median. Also tracks a background grid, which yields
             camera translation as a by-product (see --bg-grid).
  sam3       SAM3 video predictor. A text prompt plus the exact frame-0 point selects
             the ball; each propagated mask becomes a centroid/radius/presence track.

They fail differently -- CoTracker drifts under motion blur while keeping its point
identity, while a mask tracker can flip to a different object or drop the mask entirely
-- so their disagreement is a real QC signal, which is what the proposal's
"SAM2/CoTracker disagreement" gate asks for.

Contract: read a job JSON on argv[1], write an npz to the path it names. Frames are
passed as a .npy the parent already decoded, so both processes see identical pixels
and no second decoder is introduced into the measurement path.

Usage (normally called via physbench/tracking.py, not by hand):
  envs/track/bin/python scripts/track_worker.py job.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]


def _soft_centroid(prob: np.ndarray, thr: float = 0.5) -> tuple[float, float, float, float] | None:
    """Centroid, equivalent-disc radius and mean confidence of the largest blob.

    The centroid is weighted by the mask probability rather than taken from the
    binarised mask, which keeps the sub-pixel information in the antialiased rim: a
    hard centroid quantises to ~1/sqrt(area) px and that is the same order as the
    residual floor the tolerances are calibrated against.

    Largest-connected-component only. SAM2 occasionally emits a few stray pixels
    elsewhere in the frame, and including them would drag the centre off the ball --
    unlike a threshold, "biggest blob" needs no tuning.
    """
    from scipy import ndimage

    hard = prob > thr
    if not hard.any():
        return None
    lab, n = ndimage.label(hard)
    if n < 1:
        return None
    if n > 1:
        sizes = ndimage.sum(hard, lab, index=np.arange(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        hard = lab == keep
    w = prob * hard
    tot = float(w.sum())
    if tot <= 1e-6:
        return None
    ys, xs = np.nonzero(hard)
    wv = w[ys, xs]
    cx = float((xs * wv).sum() / tot)
    cy = float((ys * wv).sum() / tot)
    # Soft area, so a half-covered rim pixel counts as half -- matches how the
    # apparent radius is defined for the classic backends, keeping radius_cv
    # comparable across backends.
    return cx, cy, float(np.sqrt(tot / np.pi)), float(wv.mean())


def run_sam2(frames: np.ndarray, seed: dict, device: str, model_path: str,
             dtype: str = "bfloat16") -> dict:
    """SAM2.1 video propagation from one positive click at the seeded ball centre.

    Presence comes from the model's own object_score_logits rather than from a mask
    area threshold. When the ball leaves the frame SAM2 reports the object as absent,
    which is exactly the observation we want to record as "not found" -- and it is the
    model's judgement, not a tuned cutoff of ours.
    """
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    n = len(frames)
    xy = np.full((n, 2), np.nan)
    rad = np.full(n, np.nan)
    sc = np.full(n, np.nan)

    processor = Sam2VideoProcessor.from_pretrained(model_path)
    model = Sam2VideoModel.from_pretrained(
        model_path, dtype=getattr(torch, dtype)).to(device).eval()

    session = processor.init_video_session(
        video=frames, inference_device=device, dtype=getattr(torch, dtype))
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=1,
        input_points=[[[[float(seed["cx"]), float(seed["cy"])]]]],
        input_labels=[[[1]]],
        original_size=(frames.shape[1], frames.shape[2]),
    )

    with torch.inference_mode():
        # start_frame_idx is explicit: the session only knows where the inputs are
        # after a forward pass has run on that frame, and we have not run one yet.
        for out in model.propagate_in_video_iterator(inference_session=session,
                                                     start_frame_idx=0):
            i = int(out.frame_idx)
            masks = processor.post_process_masks(
                [out.pred_masks], original_sizes=[[frames.shape[1], frames.shape[2]]],
                binarize=False)[0]
            present = True
            if getattr(out, "object_score_logits", None) is not None:
                present = bool(float(out.object_score_logits.flatten()[0]) > 0.0)
            if not present:
                continue
            prob = torch.sigmoid(masks[0].float()).cpu().numpy()
            if prob.ndim == 3:
                prob = prob[0]
            hit = _soft_centroid(prob)
            if hit is None:
                continue
            cx, cy, r, conf = hit
            xy[i] = (cx, cy)
            rad[i] = r
            sc[i] = conf

    del model
    torch.cuda.empty_cache()
    return {"xy": xy, "radius": rad, "score": sc}


def _write_sam3_frames(frames: np.ndarray, out_dir: Path) -> str:
    """Write the already-decoded clip in SAM3's JPEG-folder format.

    SAM3's video predictor accepts either an MP4 or a directory containing numbered
    JPEGs. The parent process deliberately hands us a numpy array so both trackers see
    identical decoded pixels; writing the frames here keeps that contract while using
    the predictor's supported input path.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(
            out_dir / f"{i:05d}.jpg", quality=100, subsampling=0
        )
    return str(out_dir)


def _as_numpy(value):
    """Move a torch/numpy SAM3 output to a detached numpy array."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sam3_mask_for_target(outputs: dict, target_id: int | None,
                          previous: np.ndarray) -> tuple[np.ndarray | None, int | None,
                                                          float]:
    """Select the prompted object from one SAM3 frame output.

    Text prompts can return more than one instance. The point prompt identifies the
    ball on frame 0; after that, the persistent SAM3 object id is preferred. The
    nearest-centroid fallback is only for older/alternate SAM3 output adapters that do
    not return object ids.
    """
    masks = _as_numpy(outputs.get("out_binary_masks"))
    if masks is None:
        return None, target_id, float("nan")
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] == 0:
        return None, target_id, float("nan")
    masks = masks.astype(bool)

    ids = _as_numpy(outputs.get("out_obj_ids"))
    ids = np.asarray(ids).reshape(-1) if ids is not None else None
    probs = _as_numpy(outputs.get("out_probs"))
    probs = np.asarray(probs).reshape(-1) if probs is not None else None

    candidate = np.flatnonzero(np.asarray([m.any() for m in masks]))
    if len(candidate) == 0:
        return None, target_id, float("nan")
    if target_id is not None and ids is not None:
        same = candidate[ids[candidate].astype(int) == int(target_id)]
        if len(same):
            candidate = same
        else:
            # Do not silently jump to another object after the prompted one vanishes.
            return None, target_id, float("nan")

    def centroid(i: int) -> np.ndarray:
        yy, xx = np.nonzero(masks[i])
        return np.array([float(xx.mean()), float(yy.mean())])

    if len(candidate) == 1:
        picked = int(candidate[0])
    else:
        ref = np.asarray(previous, dtype=float)
        if not np.isfinite(ref).all():
            picked = int(candidate[np.argmax([masks[i].sum() for i in candidate])])
        else:
            picked = int(min(candidate, key=lambda i: np.linalg.norm(centroid(int(i)) - ref)))

    picked_id = int(ids[picked]) if ids is not None and picked < len(ids) else target_id
    score = float(probs[picked]) if probs is not None and picked < len(probs) else 1.0
    return masks[picked], picked_id, score


def run_sam3(frames: np.ndarray, seed: dict, device: str, source_dir: str,
             checkpoint_path: str, prompt: str = "ball",
             frame_dir: str | None = None) -> dict:
    """SAM3 video propagation from a text prompt plus an exact positive click.

    The SAM3 predictor consumes a JPEG folder and yields per-frame masklets. We use the
    object id selected on frame 0 to prevent a text prompt such as ``ball`` from
    switching identities if another instance appears later. The returned interface is
    deliberately identical to SAM2/CoTracker: ``xy``, equivalent-disc ``radius`` and
    ``score`` arrays with NaN marking absent frames.
    """
    if source_dir:
        source = Path(source_dir)
        if source.is_dir():
            sys.path.insert(0, str(source))
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {checkpoint_path!r}. "
            "Download an approved checkpoint and set tracking.sam3_checkpoint."
        )

    import torch
    from sam3.model_builder import build_sam3_video_predictor

    n, h, w = frames.shape[:3]
    frame_path = Path(frame_dir or "sam3-frames")
    resource_path = _write_sam3_frames(frames, frame_path)
    xy = np.full((n, 2), np.nan)
    rad = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    predictor = None
    session_id = None
    target_id = None
    previous = np.array([float(seed["cx"]), float(seed["cy"])])

    try:
        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
        )
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": resource_path,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
        })
        session_id = response["session_id"]
        # SAM3 expects relative point coordinates by default. Combining a point with
        # the text prompt makes the target deterministic while retaining open-vocab
        # prompting when the scene contains several balls.
        prompt_response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "obj_id": 1,
            "text": prompt or None,
            "points": [[float(seed["cx"]) / w, float(seed["cy"]) / h]],
            "point_labels": [1],
            "rel_coordinates": True,
            "output_prob_thresh": 0.5,
        })
        pending = {0: prompt_response.get("outputs", {})}
        for item in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": n,
            "output_prob_thresh": 0.5,
        }):
            pending[int(item["frame_index"])] = item.get("outputs", {})

        for i in sorted(pending):
            if i < 0 or i >= n:
                continue
            mask, target_id, score = _sam3_mask_for_target(
                pending[i], target_id, previous
            )
            if mask is None:
                continue
            yy, xx = np.nonzero(mask)
            if len(xx) == 0:
                continue
            xy[i] = (float(xx.mean()), float(yy.mean()))
            rad[i] = float(np.sqrt(len(xx) / np.pi))
            sc[i] = score
            previous = xy[i]
    finally:
        if predictor is not None and session_id is not None:
            try:
                predictor.handle_request({
                    "type": "close_session", "session_id": session_id
                })
            except Exception:
                pass
        if predictor is not None:
            try:
                predictor.shutdown()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"xy": xy, "radius": rad, "score": sc}


def _ball_queries(seed: dict, n_ring: int = 8, frac: float = 0.55) -> np.ndarray:
    """Centre plus a ring of points inside the ball disc, as (x, y) at frame 0.

    A ring rather than the centre alone, because a single point can be lost to blur
    while the ball is plainly still there; and inside the disc at 0.55r rather than on
    the rim, so the queries stay on the ball rather than on the boundary where the
    background leaks in.
    """
    cx, cy, r = float(seed["cx"]), float(seed["cy"]), float(seed["radius"])
    ang = np.arange(n_ring) * (2 * np.pi / n_ring)
    pts = [(cx, cy)] + [(cx + frac * r * np.cos(a), cy + frac * r * np.sin(a)) for a in ang]
    return np.asarray(pts, np.float32)


def _bg_queries(shape: tuple[int, int], seed: dict, step: int = 96) -> np.ndarray:
    """Grid of background points, ball neighbourhood excluded.

    Only the ball's frame-0 neighbourhood can be excluded here, and the ball then flies
    across the rest of the grid. That is handled downstream rather than geometrically:
    a point the ball passes over is reported occluded by CoTracker's visibility head, and
    the camera-shift median takes only points visible in both this frame and frame 0. So
    the ball removes itself from its own exclusion zone, without us having to predict
    where it will go.
    """
    h, w = shape
    cx, cy, r = float(seed["cx"]), float(seed["cy"]), float(seed["radius"])
    keep = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if np.hypot(x - cx, y - cy) > 4.0 * r:
                keep.append((x, y))
    return np.asarray(keep, np.float32)


def run_cotracker(frames: np.ndarray, seed: dict, device: str, ckpt: str,
                  src: str, bg_grid: bool = True) -> dict:
    """CoTracker3 offline on a ring of ball points, plus a background grid.

    The ball centre is estimated as the median over visible ring points of
    (point position - its frame-0 offset from the centre). Subtracting the offset
    first matters: a plain median over surviving points is pulled toward whichever
    side of the disc stayed visible, and that bias moves with the blur direction --
    i.e. it would correlate with speed, which is exactly what the kinematics are
    measuring. Offset-corrected, partial visibility costs precision but not accuracy.

    Runs at the model's 384x512 internal resolution, so its positional precision is
    coarser than SAM2's full-resolution mask centroid. That is why SAM2 leads and this
    is the cross-check (see physbench/tracking.py).
    """
    import torch

    sys.path.insert(0, src)
    from cotracker.predictor import CoTrackerPredictor

    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    # window_len=60 is the offline CoTracker3 configuration the scaled_offline
    # checkpoint was trained with; the state dict will not load under any other.
    pred = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60,
                             v2=False).to(device).eval()

    ball = _ball_queries(seed)
    bg = _bg_queries((h, w), seed) if bg_grid else np.zeros((0, 2), np.float32)
    pts = np.concatenate([ball, bg], axis=0)
    q = np.zeros((1, len(pts), 3), np.float32)
    q[0, :, 1:] = pts
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)

    with torch.inference_mode():
        tracks, vis = pred(video, queries=torch.from_numpy(q).to(device))
    tracks = tracks[0].cpu().numpy()          # (T, N, 2)
    vis = vis[0].cpu().numpy().astype(bool)   # (T, N)
    del video, pred
    torch.cuda.empty_cache()

    nb = len(ball)
    bt, bv = tracks[:, :nb, :], vis[:, :nb]
    offset = ball - ball[0]                   # frame-0 offsets from the ball centre

    xy = np.full((n, 2), np.nan)
    rad = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    r_seed = float(seed["radius"])
    spread0 = float(np.median(np.linalg.norm(ball[1:] - ball[0], axis=1))) or 1.0

    for i in range(n):
        m = bv[i]
        # Majority of the ring, so the median has a stable support. This is the
        # model's own visibility head, not a threshold of ours.
        if m.sum() < max(3, nb // 2):
            continue
        est = bt[i][m] - offset[m]
        xy[i] = np.median(est, axis=0)
        if m[1:].sum() >= 3:
            spread = float(np.median(np.linalg.norm(bt[i][1:][m[1:]] - xy[i], axis=1)))
            rad[i] = r_seed * spread / spread0
        sc[i] = float(m.mean())

    out = {"xy": xy, "radius": rad, "score": sc}

    if bg_grid and tracks.shape[1] > nb:
        gt, gv = tracks[:, nb:, :], vis[:, nb:]
        shift = np.zeros((n, 2))
        for i in range(n):
            m = gv[i] & gv[0]
            if m.sum() >= 4:
                shift[i] = np.median(gt[i][m] - gt[0][m], axis=0)
            elif i:
                shift[i] = shift[i - 1]
        out["camera_shift"] = shift
        out["bg_points"] = float(tracks.shape[1] - nb)
    return out


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    # Read into a writable array: torch.from_numpy on a read-only mmap warns and the
    # video processor writes into its input.
    frames = np.load(job["frames"])
    seed = job["seed"]
    device = job.get("device") or "cuda"
    out: dict[str, np.ndarray] = {}
    meta: dict = {"backends": []}

    for backend in job["backends"]:
        if backend == "sam2":
            r = run_sam2(frames, seed, device, job["sam2_path"],
                         dtype=job.get("dtype", "bfloat16"))
        elif backend == "sam3":
            r = run_sam3(
                frames,
                seed,
                device,
                job.get("sam3_src", ""),
                job.get("sam3_checkpoint", ""),
                prompt=job.get("sam3_prompt", "ball"),
                frame_dir=job.get("sam3_frames"),
            )
        elif backend == "cotracker":
            r = run_cotracker(frames, seed, device,
                              job["cotracker_ckpt"], job["cotracker_src"],
                              bg_grid=bool(job.get("bg_grid", True)))
        else:
            raise ValueError(f"unknown backend {backend}")
        for k, v in r.items():
            if k in ("bg_points",):
                meta[f"{backend}_{k}"] = v
                continue
            out[f"{backend}_{k}"] = np.asarray(v, np.float64)
        meta["backends"].append(backend)

    Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez(job["out"], **out)
    Path(str(job["out"]) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[track_worker] wrote {job['out']} ({', '.join(meta['backends'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
