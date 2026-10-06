#!/usr/bin/env python3
"""Self-contained P6 static-friction evaluator."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TASK_ID = "P6"
TASK_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vdmbench-matplotlib")


def finite(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite(v) for v in value]
    if isinstance(value, np.ndarray): return finite(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)): return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def residual_physics_score(error: float, scale: float = 1.0) -> float:
    """V2 residual normalization q(e; a) = 1 / (1 + |e| / a)."""
    error, scale = float(error), float(scale)
    if not math.isfinite(error) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("residual error must be finite and scale must be finite and positive")
    return 1.0 / (1.0 + abs(error) / scale)


def recognized_score(physics_score: float) -> float:
    return 0.15 + 0.85 * float(np.clip(physics_score, 0.0, 1.0))


def relative_path(value: str | Path | None) -> str | None:
    if value is None: return None
    path = Path(value).expanduser()
    try: return path.resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError: return str(path)


def metadata(video: str, image_path: str | None, seed: int | None) -> tuple[str, str | None, int | None]:
    stem = Path(video).stem; match = re.search(r"_seed(\d+)$", stem, flags=re.IGNORECASE)
    actual_seed = seed if seed is not None else (int(match.group(1)) if match else None)
    base = re.sub(r"_seed\d+$", "", stem, flags=re.IGNORECASE)
    if image_path is None:
        simulation = base.startswith(("sim_", "simulation_"))
        candidates = ([TASK_ROOT / "sim_first_frame.png", TASK_ROOT / "first_frames" / "simulation" / f"{base}.png"]
                      if simulation else [TASK_ROOT / "first_frame.png", TASK_ROOT / "first_frames" / "gpt" / f"{base}.png"])
        image_path = str(next((candidate for candidate in candidates if candidate.is_file()), "")) or None
        if image_path is None:
            group = "simulation" if simulation else "gpt"
            images = sorted((TASK_ROOT / "first_frames" / group).glob("*.png"))
            image_path = str(images[0]) if len(images) == 1 else None
    return relative_path(video) or str(video), relative_path(image_path), actual_seed


def infer_model(video: str, supplied: str) -> str:
    if supplied and supplied != "unknown": return supplied
    for part in Path(video).parts:
        if "minimax" in part.lower() and "h3" in part.lower(): return part
    return supplied or "unknown"


def result_payload(video: str, image_path: str | None, seed: int | None, model: str,
                   success: bool, measurement: dict[str, Any] | None, m1: float | None,
                   m2: float | None, reason: str | None, debug: dict[str, Any], sample_id: str,
                   warnings: list[str] | None = None) -> dict[str, Any]:
    video_value, image_value, seed_value = metadata(video, image_path, seed)
    return finite({"task_id": TASK_ID, "video_path": video_value, "image_path": image_value,
                   "seed": seed_value, "model": infer_model(video, model),
                   "metrics": {"M1": {"extract_success": success, "metric": m1 if success else None},
                               "M2": {"extract_success": success, "metric": m2 if success else None},
                               "M3": None},
                   "verbose": {"sample_id": sample_id, "failure_reason": reason,
                               "quality_warnings": warnings or [],
                               "measurements": measurement or {}, "debug": debug}})


def read_video(path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0); frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames: break
    cap.release()
    if not frames: raise RuntimeError(f"video contains no frames: {path}")
    return np.stack(frames), fps


def hough_lines(frame: np.ndarray, min_length: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY); edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_length, maxLineGap=8)
    if raw is None: return []
    out = []
    for x1, y1, x2, y2 in raw[:, 0]:
        a, b = np.array([float(x1), float(y1)]), np.array([float(x2), float(y2)])
        out.append((a, b, float(np.linalg.norm(b - a))))
    return sorted(out, key=lambda item: item[2], reverse=True)


def load_rois() -> tuple[dict[str, tuple[float, float, float, float]], str]:
    path = TASK_ROOT / "evaluator" / "roi.json"
    if not path.is_file(): return {}, "auto"
    try:
        data = json.loads(path.read_text()); raw = data.get("rois", data); result = {}
        for key, value in raw.items():
            if key in {"task_id", "frame_size", "source_video", "source_frame"}: continue
            if isinstance(value, (list, tuple)) and len(value) == 4:
                box = tuple(float(v) for v in value)
                if all(0 <= v <= 1 for v in box) and box[2] > 0 and box[3] > 0 and box[0] + box[2] <= 1 and box[1] + box[3] <= 1:
                    result[key] = box
        return result, "annotated"
    except (OSError, ValueError, TypeError): return {}, "auto_invalid"


def roi_pixels(box: tuple[float, float, float, float] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if box is None: return None
    x, y, w, h = box; x0, y0 = max(0, int(x * width)), max(0, int(y * height)); x1, y1 = min(width, int((x + w) * width)), min(height, int((y + h) * height))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def board_candidates(frame: np.ndarray, box: tuple[int, int, int, int]) -> list[dict[str, float]]:
    x0, y0, x1, y1 = box; h, w = frame.shape[:2]; result = []
    for a, b, length in hough_lines(frame, max(35, int(.14 * w))):
        if not (x0 <= a[0] <= x1 and x0 <= b[0] <= x1 and y0 <= a[1] <= y1 and y0 <= b[1] <= y1): continue
        dx, dy = b - a
        if abs(dx) < 1: continue
        slope = float(dy / dx)
        # The experiment starts with a horizontal board.  Rejecting slopes
        # below 0.04 discarded the entire pre-tilt baseline and delayed block
        # tracking by roughly a quarter of the clip.
        if abs(slope) > .75: continue
        intercept = float(a[1] - slope * a[0]); center = slope * (x0 + x1) / 2 + intercept
        result.append({"slope": slope, "intercept": intercept, "length": float(length), "center_y": float(center)})
    return result


def fit_board(frame: np.ndarray, box: tuple[int, int, int, int], previous: dict[str, float] | None = None) -> dict[str, Any] | None:
    x0, y0, x1, y1 = box; h, w = frame.shape[:2]; candidates = board_candidates(frame, box)
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            if abs(first["slope"] - second["slope"]) > .08: continue
            separation = abs(first["intercept"] - second["intercept"]) * math.sqrt(1 + first["slope"] ** 2)
            # The physical board is thin (roughly 15--40 px here).  A much
            # wider interval admitted the two horizontal table edges as a
            # false board pair whenever the real board was near horizontal.
            if not .012 * h <= separation <= .09 * h: continue
            top, bottom = sorted((first, second), key=lambda c: c["center_y"])
            score = top["length"] + bottom["length"]
            if previous:
                score -= 900 * abs(top["slope"] - previous["top_slope"])
                score -= .35 * abs(top["intercept"] - previous["top_intercept"])
            pairs.append((score, top, bottom, separation))
    if not pairs:
        return None
    _, top, bottom, separation = max(pairs, key=lambda item: item[0])
    return {"top_slope": top["slope"], "top_intercept": top["intercept"],
            "bottom_slope": bottom["slope"], "bottom_intercept": bottom["intercept"],
            "thickness_px": float(separation), "angle_deg": float(np.degrees(np.arctan(abs(top["slope"])))),
            "residual_px": float(abs(top["slope"] - bottom["slope"]) * (x1 - x0)), "paired": True}


def block_observations(frame: np.ndarray, model: dict[str, Any], boxes: dict[str, tuple[int, int, int, int]]) -> list[dict[str, Any]]:
    h, w = frame.shape[:2]; hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV); saturation = hsv[:, :, 1]; value = hsv[:, :, 2]
    # Wood blocks are the saturated foreground above the board surface. The
    # search is two-dimensional and retains a bbox/centroid, unlike a column-only projection.
    x0, _, x1, _ = boxes["board"]; top = model["top_slope"] * np.arange(w) + model["top_intercept"]
    wood = ((saturation >= max(18, int(np.percentile(saturation, 45)))) & (value > 70)).astype(np.uint8)
    # Detect resolved foreground components above the measured support line.
    # Comparing their height with a percentile of other blocks incorrectly
    # removes both objects when the board itself is not chromatic.
    yy,xx=np.indices((h,w));distance=(top[None,:]-yy)/math.sqrt(1+model['top_slope']**2)
    foreground=(wood.astype(bool)&(distance>2)&(distance<.25*h)&(xx>=x0)&(xx<x1)).astype('uint8')
    foreground=cv2.morphologyEx(foreground,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,cent=cv2.connectedComponentsWithStats(foreground,8);resolved=[]
    for j in range(1,n):
        bx,by,bw,bh,area=stats[j]
        if area<100 or min(bw,bh)<10 or area>.08*h*w or area/(bw*bh)<.3:continue
        ys,xs=np.where(labels==j);gaps=distance[ys,xs]
        if np.percentile(gaps,5)>18:continue
        resolved.append({'x':float(cent[j,0]),'y':float(cent[j,1]),'width':float(bw),'height':float(bh),'area':float(area),'bbox':[int(bx),int(by),int(bw),int(bh)],'top_y':float(by),'surface_y':float(top[int(cent[j,0])]),'rise':float(np.median(gaps))})
    if len(resolved)>=2:return sorted(resolved,key=lambda item:item['x'])
    top_y = np.full(w, np.nan)
    for x in range(x0, x1):
        sy = int(round(top[x])); lo, hi = max(0, sy - int(.24 * h)), min(h, sy - max(5, int(.012 * h)))
        if hi > lo:
            ys = np.flatnonzero(wood[lo:hi, x])
            if len(ys): top_y[x] = lo + ys.min()
    rise = top - top_y; indices = np.arange(w); usable = rise[np.isfinite(rise) & (indices >= x0) & (indices < x1)]
    if len(usable) < 20: return []
    baseline = float(np.percentile(usable, 25)); raised = np.zeros(w, np.uint8)
    raised[(np.isfinite(rise)) & (rise > baseline + max(12.0, .025 * h))] = 1; raised[:x0] = 0; raised[x1:] = 0
    raised = cv2.morphologyEx(raised[None, :], cv2.MORPH_CLOSE, np.ones((1, max(5, w // 250)), np.uint8))[0]
    transitions = np.diff(np.r_[0, raised.astype(np.int8), 0]); starts = np.flatnonzero(transitions == 1); ends = np.flatnonzero(transitions == -1); out = []
    intervals = list(zip(starts.tolist(), ends.tolist()))
    major = [(start, end) for start, end in intervals if end - start >= .035 * w]
    if len(major) == 1 and boxes.get("block_left") and boxes.get("block_right"):
        start, end = major[0]
        center_left = .5 * (boxes["block_left"][0] + boxes["block_left"][2])
        center_right = .5 * (boxes["block_right"][0] + boxes["block_right"][2])
        split = int(round((center_left + center_right) / 2))
        if start + .035 * w < split < end - .035 * w: intervals = [(start, split), (split, end)]
    for start, end in intervals:
        if end - start < .035 * w: continue
        # block_left/block_right are first-frame identity hints, not permanent
        # motion limits.  Both blocks travel well outside those boxes near the
        # end of the clip, so candidates are retained across the full board ROI.
        columns = np.arange(start, end); valid_top = top_y[start:end][np.isfinite(top_y[start:end])]
        if len(valid_top) < max(4, int(.5 * (end - start))): continue
        y_top = float(np.min(valid_top)); y_surface = float(np.median(top[start:end])); y_bottom = min(float(h - 1), y_surface + max(8.0, .03 * h)); height_box = max(8.0, y_bottom - y_top)
        out.append({"x": float(np.mean(columns)), "y": float((y_top + y_bottom) / 2), "width": float(end - start), "height": float(height_box), "area": float((end - start) * height_box), "bbox": [int(start), int(y_top), int(end - start), int(height_box)], "top_y": y_top, "surface_y": y_surface, "rise": float(np.nanmedian(rise[start:end]))})
    return sorted(out, key=lambda item: item["x"])


def board_line(frame: np.ndarray) -> tuple[float, float, float] | None:
    """Return the upper board edge as slope, intercept, and angle in degrees."""
    height, width = frame.shape[:2]
    candidates = []
    for a, b, length in hough_lines(frame, max(40, width // 5)):
        dx, dy = b - a
        if abs(dx) < 1:
            continue
        slope = float(dy / dx)
        if not 0.04 <= abs(slope) <= 0.75:
            continue
        intercept = float(a[1] - slope * a[0])
        center_y = slope * width * 0.5 + intercept
        if 0.2 * height <= center_y <= 0.85 * height:
            candidates.append((slope, intercept, length, center_y))
    if not candidates:
        return None
    longest = max(item[2] for item in candidates)
    parallel = [item for item in candidates if item[2] >= 0.45 * longest]
    # The upper of the long parallel edges is the contact surface.
    slope, intercept, _, _ = min(parallel, key=lambda item: item[3])
    angle = float(np.degrees(np.arctan(abs(slope))))
    return slope, intercept, angle


def block_candidates(frame: np.ndarray, line: tuple[float, float, float]) -> list[dict[str, float]]:
    """Find raised wood-colored regions above the fitted board surface."""
    height, width = frame.shape[:2]
    slope, intercept, _ = line
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    wood = saturation > max(18, int(np.percentile(saturation, 60)))
    present = wood.any(axis=0)
    top = np.argmax(wood, axis=0).astype(float)
    top[~present] = np.nan
    top[:int(0.02 * width)] = np.nan
    top[int(0.98 * width):] = np.nan
    surface = slope * np.arange(width) + intercept
    rise = surface - top
    finite_rise = rise[np.isfinite(rise)]
    board_thickness = float(np.median(finite_rise)) if len(finite_rise) else 0.0
    raised = (np.isfinite(rise) &
              (rise > board_thickness + max(20.0, 0.03 * height))).astype(np.uint8)
    raised = cv2.morphologyEx(raised[None, :], cv2.MORPH_CLOSE,
                              np.ones((1, max(3, width // 400)), np.uint8))[0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raised[None, :], 8)
    regions = []
    for label in range(1, count):
        x, _, region_width, _, _ = stats[label]
        if region_width < 0.035 * width:
            continue
        xs = np.arange(x, x + region_width)
        local_rise = rise[x:x + region_width]
        finite_rise = local_rise[np.isfinite(local_rise)]
        if not len(finite_rise):
            continue
        regions.append({"x": float(np.mean(xs)), "width": float(region_width),
                        "rise": float(np.median(finite_rise)),
                        "score": float(region_width * np.median(finite_rise))})
    return sorted(regions, key=lambda item: item["score"], reverse=True)[:2]


def fill_track(values: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(values)
    if valid.mean() < 0.75:
        return None
    indices = np.arange(len(values))
    filled = np.interp(indices, indices[valid], values[valid])
    if len(filled) >= 5:
        filled = np.convolve(np.pad(filled, (2, 2), mode="edge"), np.ones(5) / 5.0,
                             mode="valid")
    return filled


def displacement_onset(track: np.ndarray, width: int) -> int | None:
    baseline_count = max(3, min(8, len(track) // 8))
    baseline = float(np.median(track[:baseline_count]))
    noise = float(np.median(np.abs(track[:baseline_count] - baseline))) * 1.4826
    threshold = max(0.01 * width, 5.0 * noise, 3.0)
    moved = np.abs(track - baseline) > threshold
    for index in range(baseline_count, len(track) - 2):
        if bool(np.all(moved[index:index + 3])):
            return index
    return None


def extract(frames: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    height, width = frames.shape[1:3]; rois, roi_source = load_rois()
    rois = {"board": (.02, .35, .96, .50), "block_left": (.20, .35, .28, .35), "block_right": (.38, .38, .28, .35), **rois}
    board_box = roi_pixels(rois.get("board"), width, height)
    boxes = {"board": board_box, "block_left": roi_pixels(rois.get("block_left"), width, height), "block_right": roi_pixels(rois.get("block_right"), width, height)}
    if any(value is None for value in boxes.values()): return None, {"roi_source": roi_source, "roi": rois}, "P6_roi_missing"
    models = []; observations = []; angles = np.full(len(frames), np.nan); previous = None
    for frame in frames:
        model = fit_board(frame, board_box, previous)
        if model is None and previous is not None:
            model = {**previous, "paired": False, "residual_px": float("nan")}
        models.append(model)
        if model is not None:
            angles[len(observations)] = model["angle_deg"]; previous = model
            observations.append(block_observations(frame, model, boxes))
        else:
            observations.append([])
    # Identity is initialized from the first frame containing two objects and
    # subsequently associated by nearest board-coordinate position.
    tracks = {"left": np.full((len(frames), 2), np.nan), "right": np.full((len(frames), 2), np.nan)}
    bboxes = {"left": [None] * len(frames), "right": [None] * len(frames)}; previous_centers = None
    for index, candidates in enumerate(observations):
        if len(candidates) < 2: continue
        if previous_centers is None:
            selected = sorted(candidates, key=lambda item: item["x"])[:2]
        else:
            selected = []
            remaining = candidates.copy()
            for side in ("left", "right"):
                if not remaining: break
                target = previous_centers[side]; item = min(remaining, key=lambda c: (c["x"] - target[0]) ** 2 + (c["y"] - target[1]) ** 2); selected.append(item); remaining.remove(item)
            if len(selected) < 2: continue
        selected = sorted(selected, key=lambda item: item["x"])
        previous_centers = {"left": (selected[0]["x"], selected[0]["y"]), "right": (selected[1]["x"], selected[1]["y"])}
        for side, item in zip(("left", "right"), selected):
            tracks[side][index] = [item["x"], item["y"]]; bboxes[side][index] = item["bbox"]
    valid_pair = np.isfinite(tracks["left"][:, 0]) & np.isfinite(tracks["right"][:, 0])
    gray = np.mean(frames.astype(float), axis=3); energy = np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2))
    debug: dict[str, Any] = {"roi_source": roi_source, "roi": rois, "board_models": models, "board_angles_deg": angles,
                             "block_observations": observations, "left_track": tracks["left"], "right_track": tracks["right"],
                             "left_bboxes": bboxes["left"], "right_bboxes": bboxes["right"], "motion_energy": energy,
                             "pair_coverage": float(valid_pair.mean())}
    valid_angles = np.isfinite(angles)
    if valid_angles.mean() < .65: return None, debug, "P6_board_candidate_missing"
    if valid_pair.mean() < .55: return None, debug, "P6_block_tracks_incomplete"
    # Rotate coordinates with the measured board. A fixed image-axis
    # projection confounds board rotation with the onset of sliding.
    slopes=np.array([.5*(m['top_slope']+m['bottom_slope']) if m else np.nan for m in models])
    intercepts=np.array([.5*(m['top_intercept']+m['bottom_intercept']) if m else np.nan for m in models])
    good=np.isfinite(slopes)&np.isfinite(intercepts)
    if np.ptp(slopes[good])<.05:
        # No meaningful rotation: use the observed board centre as a reference.
        pivot=np.array([0.,float(np.median(intercepts[good]))])
    else:
        A=np.c_[-slopes[good],np.ones(good.sum())];pivot=np.linalg.lstsq(A,intercepts[good],rcond=None)[0]
        residual=np.abs(A@pivot-intercepts[good]);inliers=residual<max(3.,3*np.median(residual))
        if inliers.sum()>=5:pivot=np.linalg.lstsq(A[inliers],intercepts[good][inliers],rcond=None)[0]
    directions=np.c_[np.ones(len(slopes)),slopes];directions/=np.linalg.norm(directions,axis=1)[:,None]
    projections={side:np.sum((tracks[side]-pivot)*directions,axis=1) for side in ('left','right')}
    debug.update({'observed_board_pivot_xy':pivot,'coordinate_definition':'instantaneous board axis relative to the intersection of observed board centerlines'})
    left = fill_track(projections["left"]); right = fill_track(projections["right"])
    if left is None or right is None: return None, debug, "P6_block_tracks_incomplete"
    left_onset = displacement_onset(left, width); right_onset = displacement_onset(right, width)
    if left_onset is None and right_onset is None:
        measurement={'both_not_started_sliding':True,'observed_outcome':'both_stationary_on_board',
            'block_track_coverage':float(valid_pair.mean()),'onset_sync_error':None,'critical_angle_difference_deg':None,
            'board_start_angle_deg':float(angles[valid_angles][0]),'board_end_angle_deg':float(angles[valid_angles][-1]),
            'board_angle_monotonicity':1.,'experiment_frame_count':len(frames)}
        debug.update(left_displacement_board_px=left,right_displacement_board_px=right,onset_frames={'left':None,'right':None})
        return measurement,debug,None
    if left_onset is None or right_onset is None: return None, debug, "P6_block_onset_missing"
    angle_track = np.interp(np.arange(len(angles)), np.flatnonzero(valid_angles), angles[valid_angles])
    monotonicity = float(np.mean(np.diff(angle_track) >= -0.25)) if len(angle_track) > 1 else 1.0
    measurement = {"left_onset_frame": int(left_onset), "right_onset_frame": int(right_onset),
                   "left_onset_angle_deg": float(angle_track[left_onset]), "right_onset_angle_deg": float(angle_track[right_onset]),
                   "board_start_angle_deg": float(angle_track[0]), "board_end_angle_deg": float(angle_track[-1]),
                   "board_angle_monotonicity": monotonicity, "board_edge_pair_coverage": float(sum(bool(m and m.get("paired")) for m in models) / len(models)),
                   "pre_onset_stable_frames": int(min(left_onset, right_onset)), "experiment_frame_count": int(len(frames)),
                   "onset_sync_error": abs(left_onset - right_onset) / max(len(frames), 1),
                   "critical_angle_difference_deg": abs(float(angle_track[left_onset]) - float(angle_track[right_onset])),
                   "block_track_coverage": float(valid_pair.mean())}
    debug.update({"left_displacement_board_px": left, "right_displacement_board_px": right, "angle_track": angle_track,
                  "onset_frames": {"left": int(left_onset), "right": int(right_onset)},
                  "stable_indices": list(range(max(0, len(frames) - max(5, len(frames) // 8)), len(frames)))})
    return measurement, debug, None


def diagnostic_measurements(debug: dict[str, Any], frame_count: int,
                            reason: str | None) -> dict[str, Any]:
    """Summarize measured detector coverage even when scoring cannot finish."""
    angles = np.asarray(debug.get("board_angles_deg", []), dtype=float)
    left = np.asarray(debug.get("left_track", []), dtype=float)
    right = np.asarray(debug.get("right_track", []), dtype=float)
    observations = debug.get("block_observations", [])
    left_valid = np.isfinite(left[:, 0]) if left.ndim == 2 and left.shape[1] else np.zeros(frame_count, bool)
    right_valid = np.isfinite(right[:, 0]) if right.ndim == 2 and right.shape[1] else np.zeros(frame_count, bool)
    candidate_counts = [len(items) for items in observations]
    return {
        "failure_reason": reason,
        "experiment_frame_count": int(frame_count),
        "board_detection_coverage": float(np.isfinite(angles).mean()) if len(angles) else 0.0,
        "left_block_track_coverage": float(left_valid.mean()) if len(left_valid) else 0.0,
        "right_block_track_coverage": float(right_valid.mean()) if len(right_valid) else 0.0,
        "paired_block_track_coverage": float(debug.get("pair_coverage", 0.0)),
        "frames_with_two_block_candidates": int(sum(count >= 2 for count in candidate_counts)),
        "maximum_block_candidate_count": int(max(candidate_counts, default=0)),
    }


def debug_artifacts(frames: np.ndarray, debug: dict[str, Any], root: Path, sample_id: str, fps: float) -> dict[str, str | None]:
    folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True); raw = folder / "measurements.json"; raw.write_text(json.dumps(finite(debug), indent=2))
    plot = folder / "plot.png"; import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    raw_angles = np.asarray(debug.get("board_angles_deg", []), dtype=float)
    angle_track = np.asarray(debug.get("angle_track", []), dtype=float)
    if len(raw_angles): axes[0].plot(raw_angles, ".-", markersize=2, alpha=.65, label="measured board angle")
    if len(angle_track): axes[0].plot(angle_track, linewidth=1.5, label="interpolated board angle")
    left_track = np.asarray(debug.get("left_track", []), dtype=float)
    right_track = np.asarray(debug.get("right_track", []), dtype=float)
    if left_track.ndim == 2 and left_track.shape[1]: axes[1].plot(left_track[:, 0], ".-", markersize=2, label="left block x (raw)")
    if right_track.ndim == 2 and right_track.shape[1]: axes[1].plot(right_track[:, 0], ".-", markersize=2, label="right block x (raw)")
    left_displacement = np.asarray(debug.get("left_displacement_board_px", []), dtype=float)
    right_displacement = np.asarray(debug.get("right_displacement_board_px", []), dtype=float)
    if len(left_displacement): axes[1].plot(left_displacement, linewidth=1.5, label="left board coordinate")
    if len(right_displacement): axes[1].plot(right_displacement, linewidth=1.5, label="right board coordinate")
    candidate_counts = np.asarray([len(items) for items in debug.get("block_observations", [])], dtype=float)
    if len(candidate_counts): axes[2].step(np.arange(len(candidate_counts)), candidate_counts, where="mid", label="block candidates")
    motion = np.asarray(debug.get("motion_energy", []), dtype=float)
    if len(motion):
        scale = max(float(np.nanpercentile(motion, 95)), 1e-9)
        axes[2].plot(np.arange(1, len(motion) + 1), motion / scale, alpha=.65, label="motion energy (normalized)")
    onset = debug.get("onset_frames", {})
    if onset.get("left") is not None:
        axes[1].axvline(int(onset["left"]), color="tab:blue", linestyle="--", label="left onset")
    if onset.get("right") is not None:
        axes[1].axvline(int(onset["right"]), color="tab:orange", linestyle="--", label="right onset")
    axes[0].set_ylabel("angle (deg)"); axes[1].set_ylabel("position (px)")
    axes[2].set_ylabel("count / normalized energy"); axes[2].set_xlabel("frame")
    for axis in axes:
        if axis.lines or axis.patches: axis.legend(loc="best")
        axis.grid(alpha=.2)
    reason = debug.get("failure_reason")
    title = f"{sample_id}: board and block tracking"
    if reason: title += f"\nfailed: {reason}"
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(plot, dpi=120); plt.close(fig)
    overlay = folder / "overlay.mp4"; writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps or 24.0, (frames.shape[2], frames.shape[1]))
    if not writer.isOpened(): return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": None}
    models = debug.get("board_models", []); regions = debug.get("block_observations", []); rois = debug.get("roi", {})
    for index, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        board = roi_pixels(tuple(rois["board"]), bgr.shape[1], bgr.shape[0]) if "board" in rois else None
        if board: cv2.rectangle(bgr, (board[0], board[1]), (board[2], board[3]), (255, 180, 0), 1)
        item = models[index] if index < len(models) else None
        if item:
            for slope_key, intercept_key, color in (("top_slope", "top_intercept", (0, 220, 220)), ("bottom_slope", "bottom_intercept", (220, 0, 220))):
                y0 = int(item[intercept_key]); y1 = int(item[slope_key] * (frames.shape[2] - 1) + item[intercept_key]); cv2.line(bgr, (0, y0), (frames.shape[2] - 1, y1), color, 2)
        if index < len(regions):
            for region in regions[index]:
                x, y, w, h = region["bbox"]; cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 230), 2); cv2.circle(bgr, (int(region["x"]), int(region["y"])), 4, (0, 255, 255), -1)
        for side, color in (("left", (0, 180, 255)), ("right", (255, 0, 180))):
            track = debug.get(side + "_track", [])
            if index < len(track) and np.all(np.isfinite(track[index])): cv2.circle(bgr, tuple(np.asarray(track[index], int)), 5, color, 2)
        onset = debug.get("onset_frames", {})
        if index == onset.get("left"):
            cv2.putText(bgr, "left onset", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 180, 255), 2)
        if index == onset.get("right"):
            cv2.putText(bgr, "right onset", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 180), 2)
        writer.write(bgr)
    writer.release(); return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": str(overlay)}


def evaluate_video(video: str, output: Path, sample_id: str, max_frames: int | None,
                   image_path: str | None, seed: int | None, model: str) -> int:
    try:
        frames, fps = read_video(video, max_frames); measurement, data, reason = extract(frames); data["failure_reason"] = reason; root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"; debug = debug_artifacts(frames, data, root, sample_id, fps)
        if measurement is None:
            diagnostics = diagnostic_measurements(data, len(frames), reason)
            payload = result_payload(video, image_path, seed, model, False, diagnostics, None, None, reason, debug, sample_id)
        else:
            violations = []
            if measurement["board_end_angle_deg"] - measurement["board_start_angle_deg"] < 5.0:
                violations.append("P6_board_tilt_insufficient")
            if measurement["board_angle_monotonicity"] < 0.9:
                violations.append("P6_nonmonotonic_board_tilt")
            q1 = .7 if measurement.get("both_not_started_sliding") else residual_physics_score(measurement["onset_sync_error"])
            q2 = .7 if measurement.get("both_not_started_sliding") else residual_physics_score(measurement["critical_angle_difference_deg"], 90.0)
            m1, m2 = recognized_score(q1), recognized_score(q2)
            measurement["score_normalization"] = {
                "score_range": [0.0, 1.0], "higher_is_better": True,
                "M1": {"raw_measurement": "onset_sync_error",
                       "scale": 1.0, "physics_score": q1, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(onset_sync_error))", "score": m1},
                "M2": {"raw_measurement": "critical_angle_difference_deg",
                       "scale": 90.0, "physics_score": q2, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(critical_angle_difference_deg) / 90)", "score": m2},
            }
            if measurement.get('both_not_started_sliding'):
                for key in ('M1','M2'):
                    measurement['score_normalization'][key].update(raw_measurement='observed_outcome',formula='opinion_v2: both stationary => physics score 0.7')
            payload = result_payload(video, image_path, seed, model, True, measurement,
                                     m1, m2, None, debug, sample_id, violations)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"; folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True); error_path = folder / "error.txt"; error_path.write_text(error + "\n"); payload = result_payload(video, image_path, seed, model, False, None, None, None, error, {"directory": str(folder), "error": str(error_path)}, sample_id); print(error, file=sys.stderr); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2)); return 2
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(finite(payload), indent=2, allow_nan=False)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-contained P6 evaluator"); parser.add_argument("--video"); parser.add_argument("--task_id", default=TASK_ID); parser.add_argument("--output"); parser.add_argument("--videos"); parser.add_argument("--outdir"); parser.add_argument("--model", default="unknown"); parser.add_argument("--image-path", "--image_path", dest="image_path"); parser.add_argument("--seed", type=int); parser.add_argument("--sample-id"); parser.add_argument("--max-frames", type=int); args = parser.parse_args()
    if args.task_id.upper() != TASK_ID: parser.error(f"this evaluator only supports {TASK_ID}")
    if args.video:
        if not args.output: parser.error("--video requires --output")
        return evaluate_video(args.video, Path(args.output), args.sample_id or Path(args.video).stem, args.max_frames, args.image_path, args.seed, args.model)
    if not args.videos: parser.error("provide --video or --videos")
    source = Path(args.videos); files = [source] if source.is_file() else sorted(source.glob("*.mp4"))
    if not files: print(f"no MP4 videos found under {source}", file=sys.stderr); return 1
    outdir = Path(args.outdir or "eval_results"); json_dir = outdir / "json"; json_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, 1):
        rc = evaluate_video(str(path), json_dir / f"{path.stem}.json", path.stem, args.max_frames, args.image_path, args.seed, args.model); print(f"[{index}/{len(files)}] {path.stem} {'OK' if rc == 0 else 'ERROR'}")
    with (outdir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "video_path", "image_path", "seed", "model", "M1_extract_success", "M1", "M2_extract_success", "M2", "M3", "failure_reason"]); writer.writeheader()
        for path in sorted(json_dir.glob("*.json")):
            payload = json.loads(path.read_text()); writer.writerow({"task_id": payload["task_id"], "video_path": payload["video_path"], "image_path": payload["image_path"], "seed": payload["seed"], "model": payload["model"], "M1_extract_success": payload["metrics"]["M1"]["extract_success"], "M1": payload["metrics"]["M1"]["metric"], "M2_extract_success": payload["metrics"]["M2"]["extract_success"], "M2": payload["metrics"]["M2"]["metric"], "M3": None, "failure_reason": payload["verbose"]["failure_reason"]})
    return 0


if __name__ == "__main__": raise SystemExit(main())
