#!/usr/bin/env python3
"""Self-contained P21 communicating-vessels evaluator."""
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

TASK_ID = "P21"
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
        candidates = ([TASK_ROOT / "sim_first_frame.png", TASK_ROOT / "first_frames" / "simulation" / f"{base}.png", TASK_ROOT / "first_frame.png"]
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
                   m2: float | None, reason: str | None, debug: dict[str, Any],
                   sample_id: str) -> dict[str, Any]:
    video_value, image_value, seed_value = metadata(video, image_path, seed)
    return finite({"task_id": TASK_ID, "video_path": video_value, "image_path": image_value,
                   "seed": seed_value, "model": infer_model(video, model),
                   "metrics": {"M1": {"extract_success": success, "metric": m1 if success else None},
                               "M2": {"extract_success": success, "metric": m2 if success else None},
                               "M3": None},
                   "verbose": {"sample_id": sample_id, "failure_reason": reason,
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


def load_rois() -> tuple[dict[str, tuple[float, float, float, float]], str]:
    path = TASK_ROOT / "evaluator" / "roi.json"
    if not path.is_file():
        return {}, "auto"
    try:
        data = json.loads(path.read_text())
        raw = data.get("rois", data)
        result = {}
        for key, value in raw.items():
            if key in {"task_id", "frame_size", "source_video", "source_frame"}:
                continue
            if isinstance(value, (list, tuple)) and len(value) == 4:
                box = tuple(float(v) for v in value)
                if (all(0 <= v <= 1 for v in box) and box[2] > 0 and box[3] > 0
                        and box[0] + box[2] <= 1 and box[1] + box[3] <= 1):
                    result[key] = box
        return result, "annotated"
    except (OSError, ValueError, TypeError):
        return {}, "auto_invalid"


def roi_pixels(box: tuple[float, float, float, float] | None,
               width: int, height: int) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x, y, w, h = box
    x0, y0 = max(0, int(round(x * width))), max(0, int(round(y * height)))
    x1, y1 = min(width, int(round((x + w) * width))), min(height, int(round((y + h) * height)))
    return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))


def liquid_mask(frame: np.ndarray) -> np.ndarray:
    """Blue liquid is separated from the neutral background in HSV space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue, saturation, value = [hsv[:, :, i] for i in range(3)]
    mask = ((hue >= 78) & (hue <= 125) & (saturation >= 28) & (value >= 45)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def auto_arm_rois(frames: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    """Find the two vertical columns containing the blue liquid on the first frame."""
    mask = liquid_mask(frames[0])
    profile = mask.mean(axis=0)
    active = profile > 0.08
    # Cast before differencing.  ``np.diff`` on bools returns a bool XOR, so
    # comparing it with ``-1`` silently discarded every falling edge and made
    # the detector fall back to an unnecessarily wide right-arm ROI.
    transitions = np.diff(np.r_[0, active.astype(np.int8), 0])
    groups = []
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    for start, end in zip(starts, ends):
        width = int(end - start)
        mean_occupancy = float(profile[start:end].mean())
        if width >= max(8, frames.shape[2] // 100) and mean_occupancy >= 0.14:
            groups.append((int(start), int(end)))
    if len(groups) < 2:
        width = frames.shape[2]
        # Narrow fallback boxes are important for pale renders where the blue
        # mask is too weak to identify both columns.  In particular, the right
        # tube is much narrower than the left one.
        groups = [(int(.35 * width), int(.46 * width)),
                  (int(.58 * width), int(.63 * width))]
    groups = sorted(groups, key=lambda item: item[0])
    height, width = frames.shape[1:3]
    result = {}
    for label, (x0, x1) in zip(("left_arm", "right_arm"), groups[:2]):
        pad = max(3, int(.12 * (x1 - x0)))
        result[label] = (max(0, (x0 - pad) / width), .06,
                         min(width, x1 + pad) / width - max(0, (x0 - pad) / width), .72)
    return result


def arm_level(frame: np.ndarray, box: tuple[int, int, int, int], previous: float | None = None) -> tuple[float, float, list[float]]:
    x0, y0, x1, y1 = box
    mask = liquid_mask(frame)
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return float("nan"), 0.0, []
    # Ignore tube walls and use the row occupancy of the liquid body. The first
    # sustained occupied row is the surface; a local gradient fallback handles
    # transparent or pale liquid where occupancy is weak.
    if crop.shape[1] > 8:
        crop = crop[:, int(.12 * crop.shape[1]):int(.88 * crop.shape[1])]
    profile = cv2.GaussianBlur(crop.astype(np.float32), (1, 9), 0).mean(axis=1)
    candidates = np.flatnonzero(profile >= .42)
    if previous is not None:
        candidates = candidates[np.abs(candidates + y0 - previous) <= max(28, .10 * (y1 - y0))]
    if len(candidates):
        # Prefer the upper edge of the largest contiguous liquid run.
        level = int(candidates[0]) + y0
        score = float(profile[level - y0])
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        row = np.mean(np.abs(np.diff(gray, axis=0)), axis=1)
        row = cv2.GaussianBlur(row[:, None], (1, 9), 0).ravel()
        lo, hi = y0 + 2, y1 - 2
        if previous is not None:
            lo = max(lo, int(previous - .12 * (y1 - y0)))
            hi = min(hi, int(previous + .12 * (y1 - y0)))
        if hi <= lo:
            return float("nan"), 0.0, []
        level = lo + int(np.argmax(row[lo:hi]))
        score = float(row[level] / max(np.percentile(row[lo:hi], 90), 1e-6))
    return float(level), min(1.0, score), (candidates + y0).astype(float).tolist()


def surface_y(mask: np.ndarray, x0: int, x1: int) -> float:
    ys = []
    for x in range(max(0, x0), min(mask.shape[1], x1)):
        hit = np.flatnonzero(mask[:, x])
        if len(hit): ys.append(hit.min())
    return float(np.median(ys)) if ys else float("nan")


def stable_tail(values: np.ndarray, min_frames: int = 5) -> tuple[int, int] | None:
    valid = np.isfinite(values)
    if valid.sum() < min_frames: return None
    last = int(np.flatnonzero(valid)[-1])
    start = max(0, last - min_frames + 1)
    for candidate in range(start, -1, -1):
        seg = values[candidate:last + 1]
        if np.isfinite(seg).all() and float(np.median(np.abs(seg - np.median(seg)))) <= max(float(np.std(seg)) * 3.0, 1e-6):
            return candidate, last + 1
    return start, last + 1


def horizontal_lines(frame: np.ndarray, min_length: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                          minLineLength=min_length, maxLineGap=8)
    out = []
    if raw is not None:
        for x1, y1, x2, y2 in raw[:, 0]:
            a, b = np.array([x1, y1], float), np.array([x2, y2], float)
            d = b - a
            if abs(d[1]) < 0.15 * max(np.linalg.norm(d), 1.0):
                out.append((a, b, float(np.linalg.norm(d))))
    return sorted(out, key=lambda item: item[2], reverse=True)


def extract(frames: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    n, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
    rois, roi_source = load_rois()
    if "left_arm" not in rois or "right_arm" not in rois:
        rois = {**auto_arm_rois(frames), **rois}
    arm_boxes = {key: roi_pixels(rois.get(key), width, height) for key in ("left_arm", "right_arm")}
    if not arm_boxes["left_arm"] or not arm_boxes["right_arm"]:
        return None, {"roi_source": roi_source, "roi": rois}, "P21_arm_roi_missing"
    levels = {"left": np.full(n, np.nan), "right": np.full(n, np.nan)}
    confidence = {"left": np.zeros(n), "right": np.zeros(n)}
    candidates = {"left": [], "right": []}
    for index, frame in enumerate(frames):
        for side, key in (("left", "left_arm"), ("right", "right_arm")):
            previous = levels[side][index - 1] if index else None
            previous_value = float(previous) if previous is not None and np.isfinite(previous) else None
            value, score, rows = arm_level(frame, arm_boxes[key], previous_value)
            levels[side][index] = value
            confidence[side][index] = score
            candidates[side].append(rows)
    ys_l, ys_r = levels["left"], levels["right"]
    debug = {"roi_source": roi_source, "roi": rois, "arm_boxes_px": arm_boxes,
             "left_levels": ys_l, "right_levels": ys_r,
             "left_confidence": confidence["left"], "right_confidence": confidence["right"],
             "left_candidates": candidates["left"], "right_candidates": candidates["right"],
             "surface_coverage": float(np.isfinite(ys_l).mean() * np.isfinite(ys_r).mean())}
    if np.isfinite(ys_l).mean() < 0.75 or np.isfinite(ys_r).mean() < 0.75:
        return None, debug, "P21_surface_visibility_incomplete"
    # Reject isolated jumps before selecting the stable tail. A jump larger
    # than 12% of the arm is not a physically plausible frame-to-frame motion.
    for values, key in ((ys_l, "left"), (ys_r, "right")):
        jump = np.abs(np.diff(values))
        bad = np.r_[False, jump > .12 * (arm_boxes[key + "_arm"][3] - arm_boxes[key + "_arm"][1])]
        values[bad] = np.nan
    debug["left_levels_filtered"], debug["right_levels_filtered"] = ys_l.copy(), ys_r.copy()
    tail_l, tail_r = stable_tail(ys_l), stable_tail(ys_r)
    if tail_l is None or tail_r is None: return None, debug, "P21_stable_tail_missing"
    start, stop = max(tail_l[0], tail_r[0]), min(tail_l[1], tail_r[1])
    if stop - start < 5: return None, debug, "P21_stable_tail_overlap_missing"
    sl, sr = ys_l[start:stop], ys_r[start:stop]
    slope_l = float(np.polyfit(np.arange(len(sl)), sl, 1)[0] / height)
    slope_r = float(np.polyfit(np.arange(len(sr)), sr, 1)[0] / height)
    reference_height = float(max(height, (max(arm_boxes["left_arm"][3], arm_boxes["right_arm"][3])
                                          - min(arm_boxes["left_arm"][1], arm_boxes["right_arm"][1]))))
    measurement = {"left_level_px": float(np.median(sl)), "right_level_px": float(np.median(sr)),
                   "reference_height_px": reference_height, "left_velocity_norm": slope_l,
                   "right_velocity_norm": slope_r, "stable_frame_count": stop - start,
                   "level_difference_signed_norm": float(np.median(sl) - np.median(sr)) / reference_height,
                   "level_difference_norm": float(np.median(sl) - np.median(sr)) / reference_height,
                   "terminal_surface_speed": max(abs(slope_l), abs(slope_r))}
    return measurement, debug, None


def debug_artifacts(frames: np.ndarray, debug: dict[str, Any], root: Path,
                    sample_id: str, fps: float) -> dict[str, str | None]:
    folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "measurements.json"; raw.write_text(json.dumps(finite(debug), indent=2))
    plot = folder / "plot.png"
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    left = np.asarray(debug.get("left_levels", []), dtype=float)
    right = np.asarray(debug.get("right_levels", []), dtype=float)
    axes[0].plot(left, label="left surface")
    axes[0].plot(right, label="right surface")
    axes[0].set_ylabel("level y (px)"); axes[0].legend(loc="best")
    if len(left): axes[1].plot(np.r_[np.nan, np.diff(left)], label="left velocity")
    if len(right): axes[1].plot(np.r_[np.nan, np.diff(right)], label="right velocity")
    axes[1].set_title(f"{sample_id}: liquid surface tracking"); axes[1].set_xlabel("frame"); axes[1].set_ylabel("dy/frame"); axes[1].legend(loc="best")
    fig.tight_layout(); fig.savefig(plot, dpi=120); plt.close(fig)
    overlay = folder / "overlay.mp4"
    writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps or 24.0,
                             (frames.shape[2], frames.shape[1]))
    if writer.isOpened():
        left, right = debug.get("left_levels", []), debug.get("right_levels", [])
        boxes = debug.get("arm_boxes_px", {})
        for index, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for side, ys, color in (("left_arm", left, (0, 220, 0)), ("right_arm", right, (0, 0, 220))):
                box = boxes.get(side)
                if box:
                    x0, y0, x1, y1 = [int(v) for v in box]
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 1)
                    if index < len(ys) and np.isfinite(ys[index]):
                        cv2.line(bgr, (x0, int(ys[index])), (x1, int(ys[index])), color, 2)
                        cv2.putText(bgr, f"{side[:1].upper()} {ys[index]:.1f}px", (x0, max(18, int(ys[index]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            writer.write(bgr)
        writer.release()
        overlay_value: str | None = str(overlay)
    else:
        overlay_value = None
    return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": overlay_value}


def evaluate_video(video: str, output: Path, sample_id: str, max_frames: int | None,
                   image_path: str | None, seed: int | None, model: str) -> int:
    try:
        frames, fps = read_video(video, max_frames)
        measurement, debug_data, reason = extract(frames)
        root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"
        debug = debug_artifacts(frames, debug_data, root, sample_id, fps)
        if measurement is None:
            payload = result_payload(video, image_path, seed, model, False, None,
                                     None, None, reason, debug, sample_id)
        else:
            q1 = residual_physics_score(measurement["level_difference_signed_norm"])
            q2 = residual_physics_score(measurement["terminal_surface_speed"])
            m1, m2 = recognized_score(q1), recognized_score(q2)
            measurement["score_normalization"] = {
                "score_range": [0.0, 1.0], "higher_is_better": True,
                "M1": {"raw_measurement": "level_difference_signed_norm",
                       "scale": 1.0, "physics_score": q1, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(level_difference_signed_norm))", "score": m1},
                "M2": {"raw_measurement": "terminal_surface_speed",
                       "scale": 1.0, "physics_score": q2, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(terminal_surface_speed))", "score": m2},
            }
            payload = result_payload(video, image_path, seed, model, True, measurement,
                                     m1, m2, None, debug, sample_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"
        folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True); error_path = folder / "error.txt"; error_path.write_text(error + "\n")
        payload = result_payload(video, image_path, seed, model, False, None, None, None,
                                 error, {"directory": str(folder), "error": str(error_path)}, sample_id)
        print(error, file=sys.stderr); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2)); return 2
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(finite(payload), indent=2, allow_nan=False)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-contained P21 evaluator")
    parser.add_argument("--video"); parser.add_argument("--task_id", default=TASK_ID); parser.add_argument("--output")
    parser.add_argument("--videos"); parser.add_argument("--outdir"); parser.add_argument("--model", default="unknown"); parser.add_argument("--image-path", "--image_path", dest="image_path"); parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-id"); parser.add_argument("--max-frames", type=int); args = parser.parse_args()
    if args.task_id.upper() != TASK_ID: parser.error(f"this evaluator only supports {TASK_ID}")
    if args.video:
        if not args.output: parser.error("--video requires --output")
        return evaluate_video(args.video, Path(args.output), args.sample_id or Path(args.video).stem, args.max_frames, args.image_path, args.seed, args.model)
    if not args.videos: parser.error("provide --video or --videos")
    source = Path(args.videos); files = [source] if source.is_file() else sorted(source.glob("*.mp4"))
    if not files: print(f"no MP4 videos found under {source}", file=sys.stderr); return 1
    outdir = Path(args.outdir or "eval_results"); json_dir = outdir / "json"; json_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, 1):
        rc = evaluate_video(str(path), json_dir / f"{path.stem}.json", path.stem, args.max_frames, args.image_path, args.seed, args.model)
        print(f"[{index}/{len(files)}] {path.stem} {'OK' if rc == 0 else 'ERROR'}")
    with (outdir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "video_path", "image_path", "seed", "model", "M1_extract_success", "M1", "M2_extract_success", "M2", "M3", "failure_reason"]); writer.writeheader()
        for path in sorted(json_dir.glob("*.json")):
            payload = json.loads(path.read_text()); writer.writerow({"task_id": payload["task_id"], "video_path": payload["video_path"], "image_path": payload["image_path"], "seed": payload["seed"], "model": payload["model"], "M1_extract_success": payload["metrics"]["M1"]["extract_success"], "M1": payload["metrics"]["M1"]["metric"], "M2_extract_success": payload["metrics"]["M2"]["extract_success"], "M2": payload["metrics"]["M2"]["metric"], "M3": None, "failure_reason": payload["verbose"]["failure_reason"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
