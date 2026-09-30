#!/usr/bin/env python3
"""Self-contained P28 charged-ball evaluator."""
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

TASK_ID = "P28"
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
        candidates = ([TASK_ROOT / "first_frames" / "simulation" / "sim_first_frame.png", TASK_ROOT / "first_frames" / "simulation" / f"{base}.png", TASK_ROOT / "first_frames" / "provided" / "first_frame.png"]
                      if simulation else [TASK_ROOT / "first_frame.png", TASK_ROOT / "first_frames" / "provided" / "first_frame.png", TASK_ROOT / "first_frames" / "gpt" / f"{base}.png"])
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
                   m2: float | None, reason: str | None, debug: dict[str, Any], sample_id: str) -> dict[str, Any]:
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


def lines(frame: np.ndarray, min_length: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
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
    if not path.is_file():
        return {}, "auto"
    try:
        data = json.loads(path.read_text()); raw = data.get("rois", data); result = {}
        for key, value in raw.items():
            if key in {"task_id", "frame_size", "source_video", "source_frame"}:
                continue
            if isinstance(value, (list, tuple)) and len(value) == 4:
                box = tuple(float(v) for v in value)
                if all(0 <= v <= 1 for v in box) and box[2] > 0 and box[3] > 0 and box[0] + box[2] <= 1 and box[1] + box[3] <= 1:
                    result[key] = box
        return result, "annotated"
    except (OSError, ValueError, TypeError):
        return {}, "auto_invalid"


def roi_pixels(box: tuple[float, float, float, float] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x, y, w, h = box; x0, y0 = max(0, int(x * width)), max(0, int(y * height))
    x1, y1 = min(width, int((x + w) * width)), min(height, int((y + h) * height))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def default_rois(width: int, height: int) -> dict[str, tuple[float, float, float, float]]:
    return {"support": (.38, .02, .24, .16), "left_thread": (.42, .07, .10, .78),
            "right_thread": (.49, .07, .11, .78), "left_ball": (.38, .76, .13, .20),
            "right_ball": (.51, .76, .14, .20)}


def detect_ball(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[dict[str, float], np.ndarray] | None:
    x0, y0, x1, y1 = box; gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    # Dark balls are separated from the pale background; keeping the ROI below
    # the thread prevents the thread from becoming the selected component.
    threshold = min(145.0, float(np.percentile(crop, 22)))
    mask = (crop < threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < max(40, int(.002 * frame.shape[0] * frame.shape[1])) or w < 8 or h < 8:
            continue
        ratio = w / max(h, 1)
        if not .45 <= ratio <= 1.8:
            continue
        candidates.append((int(area), x, y, w, h, centers[label]))
    if not candidates:
        return None
    area, x, y, w, h, center = max(candidates, key=lambda item: item[0])
    global_center = np.array([x0 + center[0], y0 + center[1]], dtype=float)
    return ({"x": float(global_center[0]), "y": float(global_center[1]), "width": float(w),
             "height": float(h), "area": float(area), "threshold": threshold}, mask)


def fit_thread(frame: np.ndarray, box: tuple[int, int, int, int], ball: dict[str, float] | None) -> dict[str, Any] | None:
    x0, y0, x1, y1 = box; gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    mask = (crop < min(125, float(np.percentile(crop, 35)))).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return None
    points = np.column_stack([xs + x0, ys + y0]).astype(np.float32)
    # Keep the long near-vertical component and reject isolated text/noise.
    if ball is not None:
        points = points[points[:, 1] <= ball["y"] - max(4, .05 * frame.shape[0])]
    if len(points) < 25:
        return None
    vx, vy, px, py = cv2.fitLine(points.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, .01, .01).reshape(-1)
    direction = np.array([float(vx), float(vy)], dtype=float); direction /= max(np.linalg.norm(direction), 1e-9)
    if direction[1] < 0: direction = -direction
    center = np.array([float(px), float(py)])
    projection = (points - center) @ direction
    a = center + float(projection.min()) * direction; b = center + float(projection.max()) * direction
    residual = np.abs(direction[0] * (points[:, 1] - center[1]) - direction[1] * (points[:, 0] - center[0])).mean()
    return {"a": a, "b": b, "length": float(np.linalg.norm(b - a)),
            "residual_px": float(residual), "points": points}


def extract(frames: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    height, width = frames.shape[1:3]
    rois, roi_source = load_rois(); rois = {**default_rois(width, height), **rois}
    boxes = {key: roi_pixels(rois.get(key), width, height) for key in
             ("support", "left_thread", "right_thread", "left_ball", "right_ball")}
    if any(value is None for value in boxes.values()):
        return None, {"roi_source": roi_source, "roi": rois}, "P28_roi_missing"
    ball_tracks = {"left": [], "right": []}; thread_tracks = {"left": [], "right": []}
    for frame in frames:
        for side, key in (("left", "left_ball"), ("right", "right_ball")):
            found = detect_ball(frame, boxes[key]); ball_tracks[side].append(found[0] if found else None)
        for side, key, ball_side in (("left", "left_thread", "left"), ("right", "right_thread", "right")):
            thread_tracks[side].append(fit_thread(frame, boxes[key], ball_tracks[ball_side][-1]))
    support_y_candidates = []
    for side in ("left", "right"):
        for item in thread_tracks[side]:
            if item:
                support_y_candidates.append(float(min(item["a"][1], item["b"][1])))
    support_y = float(np.median(support_y_candidates)) if support_y_candidates else float(boxes["support"][1])
    centers = {side: np.array([[x["x"], x["y"]] if x else [np.nan, np.nan] for x in ball_tracks[side]], dtype=float)
               for side in ("left", "right")}
    pivots = {"left": np.full((len(frames), 2), np.nan), "right": np.full((len(frames), 2), np.nan)}
    angles = {"left": np.full(len(frames), np.nan), "right": np.full(len(frames), np.nan)}
    displacements = {"left": np.full(len(frames), np.nan), "right": np.full(len(frames), np.nan)}
    line_debug = {"left": [], "right": []}
    for index in range(len(frames)):
        for side in ("left", "right"):
            item = thread_tracks[side][index]; ball = centers[side][index]
            if item is None or not np.all(np.isfinite(ball)):
                line_debug[side].append(None); continue
            a, b = np.asarray(item["a"], float), np.asarray(item["b"], float)
            d = b - a
            if abs(d[1]) < 1e-6:
                line_debug[side].append(None); continue
            pivot_x = float(a[0] + (support_y - a[1]) * d[0] / d[1])
            pivot = np.array([pivot_x, support_y]); pivots[side][index] = pivot
            vector = ball - pivot
            angles[side][index] = np.degrees(np.arctan2(abs(vector[0]), max(abs(vector[1]), 1e-6)))
            displacements[side][index] = abs(vector[0])
            line_debug[side].append({**item, "pivot": pivot})
    valid = np.isfinite(angles["left"]) & np.isfinite(angles["right"])
    if valid.mean() < .55:
        debug = {"roi_source": roi_source, "roi": rois, "support_y_px": support_y,
                 "balls": ball_tracks, "threads": line_debug, "angles": angles,
                 "displacements": displacements, "valid_pair_coverage": float(valid.mean())}
        return None, debug, "P28_thread_ball_tracks_incomplete"
    tail = max(5, min(15, len(frames) // 6))
    stable_indices = np.flatnonzero(valid)[-tail:]
    if len(stable_indices) < 3:
        return None, {"roi_source": roi_source, "roi": rois, "balls": ball_tracks,
                      "threads": line_debug, "angles": angles, "displacements": displacements}, "P28_stable_tail_missing"
    left_angle = float(np.nanmedian(angles["left"][stable_indices])); right_angle = float(np.nanmedian(angles["right"][stable_indices]))
    axis_x = float(np.nanmedian(np.concatenate([pivots["left"][stable_indices, 0], pivots["right"][stable_indices, 0]])))
    left_disp = float(np.nanmedian(np.abs(centers["left"][stable_indices, 0] - axis_x)))
    right_disp = float(np.nanmedian(np.abs(centers["right"][stable_indices, 0] - axis_x)))
    ratio = left_disp / right_disp if right_disp > 1e-6 else float("nan")
    measurement = {"left_angle_deg": left_angle, "right_angle_deg": right_angle,
                   "left_displacement_px": left_disp, "right_displacement_px": right_disp,
                   "support_left_x_px": float(np.nanmedian(pivots["left"][stable_indices, 0])),
                   "support_right_x_px": float(np.nanmedian(pivots["right"][stable_indices, 0])),
                   "support_y_px": support_y, "vertical_axis_x_px": axis_x,
                   "stable_frame_count": int(len(stable_indices)), "angle_difference_deg": abs(left_angle - right_angle),
                   "horizontal_displacement_ratio": ratio,
                   "horizontal_displacement_symmetry_error": abs(math.log(ratio)) if np.isfinite(ratio) and ratio > 0 else None,
                   "track_pair_coverage": float(valid.mean())}
    debug = {"roi_source": roi_source, "roi": rois, "support_y_px": support_y,
             "balls": ball_tracks, "threads": line_debug, "angles": angles,
             "displacements": displacements, "stable_indices": stable_indices.tolist(),
             "vertical_axis_x_px": axis_x}
    return measurement, debug, None


def debug_artifacts(frames: np.ndarray, debug: dict[str, Any], root: Path,
                    sample_id: str, fps: float) -> dict[str, str | None]:
    folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "measurements.json"; raw.write_text(json.dumps(finite(debug), indent=2))
    plot = folder / "plot.png"; import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        axes[0].plot(debug.get("angles", {}).get(side, []), color=color, label=f"{side} angle")
        axes[1].plot(debug.get("displacements", {}).get(side, []), color=color, label=f"{side} horizontal displacement")
    axes[0].set_ylabel("angle (deg)"); axes[1].set_ylabel("|dx| (px)"); axes[1].set_xlabel("frame")
    axes[0].legend(loc="best"); axes[1].legend(loc="best"); fig.suptitle(f"{sample_id}: thread-ball geometry")
    fig.tight_layout(); fig.savefig(plot, dpi=120); plt.close(fig)
    overlay = folder / "overlay.mp4"; writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps or 24.0, (frames.shape[2], frames.shape[1]))
    if not writer.isOpened(): return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": None}
    rois = debug.get("roi", {})
    for index, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for key, color in (("left_thread", (255, 180, 0)), ("right_thread", (0, 180, 255)), ("support", (180, 180, 0)), ("left_ball", (255, 0, 255)), ("right_ball", (0, 0, 255))):
            box = roi_pixels(tuple(rois[key]), bgr.shape[1], bgr.shape[0]) if key in rois else None
            if box:
                cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), color, 1)
        for side, color in (("left", (255, 180, 0)), ("right", (0, 180, 255))):
            item = debug.get("threads", {}).get(side, [None] * len(frames))[index]
            if item:
                a, b = np.asarray(item["a"], int), np.asarray(item["b"], int)
                cv2.line(bgr, tuple(a), tuple(b), color, 2)
                p = np.asarray(item["pivot"], int); cv2.circle(bgr, tuple(p), 5, (0, 255, 255), -1)
            balls = debug.get("balls", {}).get(side, [None] * len(frames))
            ball = balls[index]
            if ball:
                cv2.circle(bgr, (int(ball["x"]), int(ball["y"])), max(6, int(.5 * ball["width"])), color, 2)
        axis = debug.get("vertical_axis_x_px")
        if axis is not None: cv2.line(bgr, (int(axis), 0), (int(axis), bgr.shape[0] - 1), (100, 255, 100), 1)
        if index < len(debug.get("stable_indices", [])) and index in debug.get("stable_indices", []):
            cv2.putText(bgr, "stable", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 0), 2)
        writer.write(bgr)
    writer.release(); return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": str(overlay)}


def evaluate_video(video: str, output: Path, sample_id: str, max_frames: int | None,
                   image_path: str | None, seed: int | None, model: str) -> int:
    try:
        frames, fps = read_video(video, max_frames); measurement, data, reason = extract(frames)
        root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"; debug = debug_artifacts(frames, data, root, sample_id, fps)
        if measurement is None:
            payload = result_payload(video, image_path, seed, model, False, None, None, None, reason, debug, sample_id)
        else:
            q1 = residual_physics_score(measurement["angle_difference_deg"], 90.0)
            q2 = residual_physics_score(measurement["horizontal_displacement_symmetry_error"])
            m1, m2 = recognized_score(q1), recognized_score(q2)
            measurement["score_normalization"] = {
                "score_range": [0.0, 1.0], "higher_is_better": True,
                "M1": {"raw_measurement": "angle_difference_deg",
                       "scale": 90.0, "physics_score": q1, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(angle_difference_deg) / 90)", "score": m1},
                "M2": {"raw_measurement": "horizontal_displacement_symmetry_error",
                       "scale": 1.0, "physics_score": q2, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(horizontal_displacement_symmetry_error))", "score": m2},
            }
            payload = result_payload(video, image_path, seed, model, True, measurement,
                                     m1, m2, None, debug, sample_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"; folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True); error_path = folder / "error.txt"; error_path.write_text(error + "\n")
        payload = result_payload(video, image_path, seed, model, False, None, None, None, error, {"directory": str(folder), "error": str(error_path)}, sample_id)
        print(error, file=sys.stderr); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2)); return 2
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(finite(payload), indent=2, allow_nan=False)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-contained P28 evaluator"); parser.add_argument("--video"); parser.add_argument("--task_id", default=TASK_ID); parser.add_argument("--output"); parser.add_argument("--videos"); parser.add_argument("--outdir"); parser.add_argument("--model", default="unknown"); parser.add_argument("--image-path", "--image_path", dest="image_path"); parser.add_argument("--seed", type=int); parser.add_argument("--sample-id"); parser.add_argument("--max-frames", type=int); args = parser.parse_args()
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
