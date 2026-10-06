#!/usr/bin/env python3
"""Self-contained P35 sand-pile angle-of-repose evaluator."""
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

TASK_ID = "P35"
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


def sand_mask(frame: np.ndarray) -> np.ndarray:
    # Identity is supplied by the two pile regions; sand need not be yellow.
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV); h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s >= 30) & (v >= 40)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def ground_y(frame: np.ndarray, box: tuple[int, int, int, int] | None,
             pile_points: list[np.ndarray] | None = None) -> float | None:
    h, w = frame.shape[:2]
    if pile_points:
        bottoms = [float(np.percentile(points[:, 1], 97)) for points in pile_points if len(points)]
        if bottoms: return float(np.median(bottoms))
    if box:
        x0, y0, x1, y1 = box; gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY); edge = np.mean(np.abs(np.diff(gray, axis=1)), axis=1)
        lo, hi = max(y0, int(.55 * h)), min(y1, int(.97 * h))
        return float(lo + int(np.argmax(edge[lo:hi]))) if hi > lo else None
    return float(.91 * h)


def pile_observation(frame: np.ndarray, box: tuple[int, int, int, int], floor: float | None,
                     mask: np.ndarray | None = None) -> dict[str, Any] | None:
    x0, y0, x1, y1 = box; mask = sand_mask(frame) if mask is None else mask; crop = mask[y0:y1, x0:x1].copy()
    if floor is not None:
        crop[int(max(0, floor - y0 + .03 * frame.shape[0])):] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < .002 * frame.shape[0] * frame.shape[1] or h < .05 * frame.shape[0]: continue
        candidates.append((int(area), label, x, y, w, h))
    if not candidates: return None
    _, label, x, y, w, h = max(candidates, key=lambda item: item[0])
    ys, xs = np.where(labels == label); points = np.column_stack([xs + x0, ys + y0]).astype(float)
    if len(points) < 30: return None
    peak_index = int(np.argmin(points[:, 1])); peak = points[peak_index]
    ground = float(np.percentile(points[:, 1], 97)) if floor is None else float(floor)
    # Compute the upper envelope with one sort/group operation.  Scanning all
    # points once per column is quadratic for large masks and stalls batching.
    columns = points[:, 0].astype(np.int32)
    order = np.argsort(columns, kind="stable")
    sorted_columns = columns[order]
    sorted_y = points[order, 1]
    unique_columns, starts = np.unique(sorted_columns, return_index=True)
    env = np.column_stack((unique_columns, np.minimum.reduceat(sorted_y, starts))).astype(float)
    fits = {}
    margin = max(3.0, .025 * frame.shape[0])
    for side, selector in (("left", env[:, 0] < peak[0]), ("right", env[:, 0] > peak[0])):
        chosen = env[selector & (env[:, 1] >= peak[1] + margin) & (env[:, 1] <= ground - margin)]
        if len(chosen) < 8: return None
        coeff = np.polyfit(chosen[:, 0], chosen[:, 1], 1); pred = np.polyval(coeff, chosen[:, 0]); rmse = float(np.sqrt(np.mean((chosen[:, 1] - pred) ** 2)))
        fits[side] = {"slope": float(coeff[0]), "intercept": float(coeff[1]), "angle_deg": float(np.degrees(np.arctan(abs(coeff[0])))), "rmse_px": rmse, "points": chosen}
    return {"points": points, "envelope": env, "peak": peak, "ground_y": ground, "fits": fits, "area": float(len(points))}


def fit_angle(contour: np.ndarray, ground_y: float) -> tuple[float, float] | None:
    if len(contour) < 8: return None
    left, right = contour[np.argmin(contour[:, 0])], contour[np.argmax(contour[:, 0])]
    values = []
    for foot in (left, right):
        score = np.linalg.norm(contour - foot, axis=1) + 0.5 * np.abs(contour[:, 1] - ground_y)
        points = contour[np.argsort(score)[:max(5, len(contour) // 5)]]
        if len(points) < 2: return None
        vx, vy, _, _ = cv2.fitLine(points.astype(np.float32).reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        values.append(float(np.degrees(np.arctan2(abs(float(vy)), abs(float(vx))))))
    return values[0], values[1]


def compact_observation(item: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep measurements.json inspectable without dumping every mask pixel."""
    if item is None:
        return None
    points = np.asarray(item.get("points", []), dtype=float)
    envelope = np.asarray(item.get("envelope", []), dtype=float)
    out = {key: value for key, value in item.items() if key not in {"points", "envelope", "fits"}}
    out["points_sample"] = points[::max(1, len(points) // 200)].tolist() if len(points) else []
    out["envelope_sample"] = envelope[::max(1, len(envelope) // 300)].tolist() if len(envelope) else []
    out["point_count"] = int(len(points))
    out["envelope_count"] = int(len(envelope))
    out["fits"] = {}
    for side, fit in item.get("fits", {}).items():
        fit_out = {key: value for key, value in fit.items() if key != "points"}
        fit_points = np.asarray(fit.get("points", []), dtype=float)
        fit_out["points_sample"] = fit_points[::max(1, len(fit_points) // 100)].tolist() if len(fit_points) else []
        fit_out["point_count"] = int(len(fit_points))
        out["fits"][side] = fit_out
    return out


def extract(frames: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    height, width = frames.shape[1:3]; rois, roi_source = load_rois()
    # The large left pile reaches close to both sides of the old box.  Keep a
    # generous first-frame search margin so the connected component is not
    # clipped before its envelope and repose angles are fitted.
    rois = {"ground": (.03, .68, .94, .28),
            "left_pile": (.04, .42, .53, .53),
            "right_pile": (.57, .42, .40, .53), **rois}
    pile_boxes = {side: roi_pixels(rois[side + "_pile"], width, height) for side in ("left", "right")}
    ground_box = roi_pixels(rois.get("ground"), width, height)
    observations = {"left": [], "right": []}; ground_tracks = []
    for frame in frames:
        frame_mask = sand_mask(frame)
        provisional = []
        for side in ("left", "right"):
            item = pile_observation(frame, pile_boxes[side], None, frame_mask)
            if item: provisional.append(item["points"])
        floor = ground_y(frame, ground_box, provisional)
        ground_tracks.append(floor)
        for side in ("left", "right"):
            observations[side].append(pile_observation(frame, pile_boxes[side], floor, frame_mask))
    valid_pair = np.array([observations["left"][i] is not None and observations["right"][i] is not None for i in range(len(frames))])
    debug: dict[str, Any] = {"roi_source": roi_source, "roi": rois, "ground_y_px": ground_tracks,
                             "piles": observations, "pair_coverage": float(valid_pair.mean())}
    if valid_pair.mean() < .45:
        return None, debug, "P35_pile_candidates_insufficient"
    tail = max(5, min(15, len(frames) // 6)); indices = np.flatnonzero(valid_pair)[-tail:]
    angle_series = {"left": np.full(len(frames), np.nan), "right": np.full(len(frames), np.nan)}
    for side in ("left", "right"):
        for i, item in enumerate(observations[side]):
            if item:
                angle_series[side][i] = np.mean([item["fits"]["left"]["angle_deg"], item["fits"]["right"]["angle_deg"]])
    if len(indices) < 3:
        return None, {**debug, "angle_series": angle_series}, "P35_stable_tail_missing"
    small_side, large_side = ("left", "right") if np.nanmedian([observations["left"][i]["area"] for i in indices]) < np.nanmedian([observations["right"][i]["area"] for i in indices]) else ("right", "left")
    small_left = float(np.nanmedian([observations[small_side][i]["fits"]["left"]["angle_deg"] for i in indices]))
    small_right = float(np.nanmedian([observations[small_side][i]["fits"]["right"]["angle_deg"] for i in indices]))
    large_left = float(np.nanmedian([observations[large_side][i]["fits"]["left"]["angle_deg"] for i in indices]))
    large_right = float(np.nanmedian([observations[large_side][i]["fits"]["right"]["angle_deg"] for i in indices]))
    small_angle = (small_left + small_right) / 2.0; large_angle = (large_left + large_right) / 2.0
    ratio = small_angle / large_angle if large_angle > 1e-6 else float("nan")
    measurement = {"small_pile_side": small_side, "large_pile_side": large_side,
                   "small_left_angle_deg": small_left, "small_right_angle_deg": small_right,
                   "large_left_angle_deg": large_left, "large_right_angle_deg": large_right,
                   "small_repose_angle_deg": small_angle, "large_repose_angle_deg": large_angle,
                   "repose_angle_ratio": ratio, "scale_error": abs(ratio - 1.0) if np.isfinite(ratio) else None,
                   "small_side_asymmetry_deg": abs(small_left - small_right), "large_side_asymmetry_deg": abs(large_left - large_right),
                   "stable_frame_count": int(len(indices)), "pile_pair_coverage": float(valid_pair.mean())}
    debug["angle_series"] = angle_series; debug["stable_indices"] = indices.tolist()
    return measurement, debug, None


def debug_artifacts(frames: np.ndarray, debug: dict[str, Any], root: Path, sample_id: str, fps: float) -> dict[str, str | None]:
    folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True); raw = folder / "measurements.json"
    compact = dict(debug)
    compact["piles"] = {side: [compact_observation(item) for item in items]
                         for side, items in debug.get("piles", {}).items()}
    raw.write_text(json.dumps(finite(compact), indent=2))
    plot = folder / "plot.png"; import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        axes[0].plot(debug.get("angle_series", {}).get(side, []), color=color, label=f"{side} repose angle")
    axes[0].set_ylabel("angle (deg)"); axes[0].legend(loc="best")
    axes[1].plot(debug.get("ground_y_px", []), color="black", label="ground y")
    axes[1].set_ylabel("ground (px)"); axes[1].set_xlabel("frame"); axes[1].legend(loc="best")
    fig.suptitle(f"{sample_id}: pile slope tracking"); fig.tight_layout(); fig.savefig(plot, dpi=120); plt.close(fig)
    overlay = folder / "overlay.mp4"; writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps or 24.0, (frames.shape[2], frames.shape[1]))
    if not writer.isOpened(): return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": None}
    piles = debug.get("piles", {})
    rois = debug.get("roi", {})
    for index, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for key, color in (("ground", (255, 220, 0)), ("left_pile", (0, 220, 0)), ("right_pile", (0, 0, 220))):
            box = roi_pixels(tuple(rois[key]), bgr.shape[1], bgr.shape[0]) if key in rois else None
            if box: cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), color, 1)
        gy = debug.get("ground_y_px", [None] * len(frames))[index]
        if gy is not None: cv2.line(bgr, (0, int(gy)), (bgr.shape[1] - 1, int(gy)), (255, 220, 0), 2)
        for side, color in (("left", (0, 220, 0)), ("right", (0, 0, 220))):
            item = piles.get(side, [None] * len(frames))[index]
            if not item: continue
            points = np.asarray(item["points"], int)
            if len(points) > 1200: points = points[::max(1, len(points) // 1200)]
            points = points.reshape(-1, 1, 2); cv2.polylines(bgr, [points], True, color, 1)
            env = np.asarray(item["envelope"])
            if len(env): cv2.polylines(bgr, [env.astype(int).reshape(-1, 1, 2)], False, (0, 220, 220), 1)
            for fit in item["fits"].values():
                xx = np.linspace(np.min(fit["points"][:, 0]), np.max(fit["points"][:, 0]), 20)
                yy = fit["slope"] * xx + fit["intercept"]
                cv2.polylines(bgr, [np.column_stack([xx, yy]).astype(int).reshape(-1, 1, 2)], False, (255, 0, 255), 2)
        writer.write(bgr)
    writer.release(); return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": str(overlay)}


def evaluate_video(video: str, output: Path, sample_id: str, max_frames: int | None,
                   image_path: str | None, seed: int | None, model: str) -> int:
    try:
        frames, fps = read_video(video, max_frames); measurement, data, reason = extract(frames); root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"; debug = debug_artifacts(frames, data, root, sample_id, fps)
        if measurement is None:
            payload = result_payload(video, image_path, seed, model, False, None, None, None, reason, debug, sample_id)
        else:
            asymmetry = max(measurement["small_side_asymmetry_deg"], measurement["large_side_asymmetry_deg"])
            q1 = residual_physics_score(measurement["scale_error"])
            q2 = residual_physics_score(asymmetry, 90.0)
            m1, m2 = recognized_score(q1), recognized_score(q2)
            measurement["score_normalization"] = {
                "score_range": [0.0, 1.0], "higher_is_better": True,
                "M1": {"raw_measurement": "scale_error",
                       "scale": 1.0, "physics_score": q1, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(scale_error))", "score": m1},
                "M2": {"raw_measurement": "max(small_side_asymmetry_deg, large_side_asymmetry_deg)",
                       "scale": 90.0, "physics_score": q2, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(max_side_asymmetry_deg) / 90)", "score": m2},
            }
            payload = result_payload(video, image_path, seed, model, True, measurement,
                                     m1, m2, None, debug, sample_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"; folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True); error_path = folder / "error.txt"; error_path.write_text(error + "\n"); payload = result_payload(video, image_path, seed, model, False, None, None, None, error, {"directory": str(folder), "error": str(error_path)}, sample_id); print(error, file=sys.stderr); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2)); return 2
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(finite(payload), indent=2, allow_nan=False)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-contained P35 evaluator"); parser.add_argument("--video"); parser.add_argument("--task_id", default=TASK_ID); parser.add_argument("--output"); parser.add_argument("--videos"); parser.add_argument("--outdir"); parser.add_argument("--model", default="unknown"); parser.add_argument("--image-path", "--image_path", dest="image_path"); parser.add_argument("--seed", type=int); parser.add_argument("--sample-id"); parser.add_argument("--max-frames", type=int); args = parser.parse_args()
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
