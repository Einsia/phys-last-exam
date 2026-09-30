#!/usr/bin/env python3
"""Self-contained P21b floating-ice-and-stone evaluator."""
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

TASK_ID = "P21b"
TASK_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vdmbench-matplotlib")


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def unit_score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("physics score must be finite")
    return float(np.clip(value, 0.0, 1.0))


def direction_score(value: float, expected_sign: int) -> float:
    """Score an inequality-only metric without inventing a target magnitude."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("direction measurement must be finite")
    return 1.0 if expected_sign * value > 0.0 else 0.0


def recognized_score(physics_score: float) -> float:
    """Add the V2 recognition share once to a measurable M metric."""
    return 0.15 + 0.85 * unit_score(physics_score)


def relative_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    try:
        return path.resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def metadata(video: str, image_path: str | None, seed: int | None) -> tuple[str, str | None, int | None]:
    stem = Path(video).stem
    match = re.search(r"_seed(\d+)$", stem, flags=re.IGNORECASE)
    actual_seed = seed if seed is not None else (int(match.group(1)) if match else None)
    if image_path is None:
        candidates = ["first_frames/simulation/sim_first_frame.png"] if stem.lower().startswith("sim") else ["first_frames/provided/first_frame.png"]
        base = re.sub(r"_seed\d+$", "", stem, flags=re.IGNORECASE)
        candidates += [f"first_frames/gpt/{base}.png", f"first_frames/simulation/{base}.png"]
        for candidate in candidates:
            if (TASK_ROOT / candidate).is_file():
                image_path = str(TASK_ROOT / candidate)
                break
    return relative_path(video) or str(video), relative_path(image_path), actual_seed


def infer_model(video: str, supplied: str) -> str:
    if supplied and supplied != "unknown":
        return supplied
    for part in Path(video).parts:
        if "minimax" in part.lower() and "h3" in part.lower():
            return part
    return supplied or "unknown"


def result_payload(
    video: str,
    image_path: str | None,
    seed: int | None,
    model: str,
    m1_success: bool,
    m1: float | None,
    m2_success: bool,
    m2: float | None,
    measurement: dict[str, Any] | None,
    reason: str | None,
    warnings: list[str],
    debug: dict[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    video_value, image_value, seed_value = metadata(video, image_path, seed)
    return finite({
        "task_id": TASK_ID,
        "video_path": video_value,
        "image_path": image_value,
        "seed": seed_value,
        "model": infer_model(video, model),
        "metrics": {
            "M1": {"extract_success": m1_success, "metric": m1 if m1_success else None},
            "M2": {"extract_success": m2_success, "metric": m2 if m2_success else None},
            "M3": None,
        },
        "verbose": {
            "sample_id": sample_id,
            "failure_reason": reason,
            "quality_warnings": warnings,
            "measurements": measurement or {},
            "debug": debug,
        },
    })


def read_video(path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
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


def edge_profile(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    profile = np.median(sobel[:, int(0.40 * width):int(0.60 * width)], axis=1)
    return np.convolve(profile, np.ones(3) / 3.0, mode="same")


def water_geometry(frames: np.ndarray) -> tuple[np.ndarray | None, int | None, dict[str, Any], str | None]:
    height = frames.shape[1]
    profiles = np.stack([edge_profile(frame) for frame in frames])
    reference = np.median(profiles[:max(3, min(8, len(frames) // 10))], axis=0)
    upper_lo, upper_hi = int(0.18 * height), int(0.55 * height)
    baseline = upper_lo + int(np.argmax(reference[upper_lo:upper_hi]))
    lower_lo, lower_hi = int(0.75 * height), int(0.93 * height)
    lower = reference[lower_lo:lower_hi]
    peak_threshold = 0.35 * float(np.max(lower)) if len(lower) else 0.0
    peaks = [
        y for y in range(lower_lo + 1, lower_hi - 1)
        if reference[y] >= reference[y - 1]
        and reference[y] >= reference[y + 1]
        and reference[y] >= peak_threshold
    ]
    floor_y = min(peaks) if peaks else (lower_lo + int(np.argmax(lower)) if len(lower) else None)
    levels = []
    search_lo = max(upper_lo, int(baseline - 0.12 * height))
    search_hi = min(upper_hi, int(baseline + 0.12 * height))
    for profile in profiles:
        index = search_lo + int(np.argmax(profile[search_lo:search_hi]))
        rows = np.arange(max(search_lo, index - 2), min(search_hi, index + 3))
        weights = profile[rows] + 1e-3
        levels.append(float(np.average(rows, weights=weights)))
    levels_array = np.asarray(levels)
    debug = {
        "waterline_y_px": levels_array,
        "vessel_floor_y_px": floor_y,
        "reference_surface_y_px": baseline,
        "surface_search_range_px": [search_lo, search_hi],
    }
    if floor_y is None or floor_y <= baseline + 0.12 * height:
        return None, floor_y, debug, "P21b_vessel_floor_missing"
    if float(np.std(levels_array)) > 0.18 * height:
        return None, floor_y, debug, "P21b_waterline_track_unstable"
    return levels_array, floor_y, debug, None


def dark_solid(frame: np.ndarray, surface_y: float, floor_y: int) -> dict[str, float] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    x0, x1 = int(0.36 * width), int(0.64 * width)
    y0, y1 = max(0, int(surface_y + 8)), min(height, int(floor_y + 2))
    if y1 - y0 < 20:
        return None
    roi = gray[y0:y1, x0:x1]
    threshold = min(170.0, float(np.percentile(roi, 28)))
    mask = (roi < threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if not 0.0002 * height * width <= area <= 0.08 * height * width:
            continue
        if w <= 2 or h <= 2 or w / h < 0.25 or w / h > 4.0:
            continue
        cx, cy = centers[label]
        global_cx, global_cy = float(x0 + cx), float(y0 + cy)
        centrality = abs(global_cx - width / 2) / width
        candidates.append((float(area) * (1.0 - centrality), x, y, w, h, global_cx, global_cy))
    if not candidates:
        return None
    _, x, y, w, h, cx, cy = max(candidates, key=lambda item: item[0])
    return {
        "x": float(x0 + x), "y": float(y0 + y), "width": float(w), "height": float(h),
        "center_x": cx, "center_y": cy, "bottom_y": float(y0 + y + h - 1),
        "threshold": threshold,
    }


def stone_observation(frame: np.ndarray, surface_y: float, floor_y: int,
                      vessel_box: tuple[int, int, int, int], seed_box: tuple[int, int, int, int] | None,
                      previous: dict[str, float] | None = None) -> dict[str, Any] | None:
    """Detect the dense stone independently from the lighter ice body."""
    height, width = frame.shape[:2]; vx0, vy0, vx1, vy1 = vessel_box
    x0, x1 = vx0 + int(.08 * (vx1 - vx0)), vx1 - int(.08 * (vx1 - vx0))
    y0, y1 = max(vy0, int(surface_y + 5)), min(vy1, int(floor_y + 2))
    if y1 - y0 < 16: return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV); hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    warm_dark = (hue >= 4) & (hue <= 45) & (sat >= 16) & (val <= 185)
    dark_textured = (val <= 105) & (sat >= 18)
    mask = (warm_dark | dark_textured).astype(np.uint8)[y0:y1, x0:x1]
    # A seed box narrows the first-frame search; subsequent frames are linked
    # to the previous centroid so the ice component cannot replace the stone.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8); candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < max(15, int(.00008 * width * height)) or area > .025 * width * height: continue
        if w < 3 or h < 3 or w > .18 * width or h > .18 * height or not .25 <= w / max(h, 1) <= 4.5: continue
        cx, cy = x0 + centers[label][0], y0 + centers[label][1]
        distance = 0.0
        if previous:
            distance = float(np.hypot(cx - previous["center_x"], cy - previous["center_y"]))
            if distance > .18 * height: continue
        seed_bonus = 0.0
        if seed_box:
            sx0, sy0, sx1, sy1 = seed_box
            if previous is None and not (sx0 - .08 * width <= cx <= sx1 + .08 * width and sy0 - .08 * height <= cy <= sy1 + .08 * height):
                continue
            seed_bonus = 1.0 if sx0 <= cx <= sx1 and sy0 <= cy <= sy1 else 0.0
        score = float(area) * (1.0 + 2.0 * seed_bonus) / (1.0 + distance / max(height, 1))
        candidates.append((score, label, x, y, w, h, cx, cy, area))
    if not candidates: return None
    _, label, x, y, w, h, cx, cy, area = max(candidates, key=lambda item: item[0])
    return {"x": float(x0 + x), "y": float(y0 + y), "width": float(w), "height": float(h),
            "center_x": float(cx), "center_y": float(cy), "bottom_y": float(y0 + y + h - 1),
            "area": float(area), "confidence": float(min(1.0, area / max(.004 * width * height, 1.0))),
            "threshold": "warm_dark_or_textured"}


def ice_observation(frame: np.ndarray, surface_y: float, floor_y: int,
                    vessel_box: tuple[int, int, int, int], search_box: tuple[int, int, int, int] | None = None) -> dict[str, float] | None:
    vx0, vy0, vx1, vy1 = search_box or vessel_box; y0, y1 = max(vy0, int(surface_y + 4)), min(vy1, int(floor_y + 1))
    if y1 <= y0: return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV); sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((sat < 65) & (val > 105)).astype(np.uint8)[y0:y1, vx0:vx1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [(int(stats[i, 4]), i) for i in range(1, count) if stats[i, 4] > .0005 * frame.shape[0] * frame.shape[1]]
    if not candidates: return None
    area, label = max(candidates)
    x, y, w, h, _ = stats[label]; return {"x": float(vx0 + x), "y": float(y0 + y), "width": float(w), "height": float(h), "area": float(area), "bottom_y": float(y0 + y + h - 1)}


def final_frame_ice_detection(frame: np.ndarray, surface_y: float, floor_y: int,
                              vessel_box: tuple[int, int, int, int]) -> dict[str, Any]:
    """Detect residual ice from the final frame only.

    This intentionally does not reuse the temporal ice track.  It searches for
    a compact, irregular, high-gradient solid inside the vessel and rejects the
    long horizontal/vertical glass and liquid boundaries that otherwise create
    false positives after the ice melts.
    """
    vx0, vy0, vx1, vy1 = vessel_box
    height, width = frame.shape[:2]
    x0 = max(vx0 + int(.04 * (vx1 - vx0)), 0)
    x1 = min(vx1 - int(.04 * (vx1 - vx0)), width)
    y0 = max(vy0, int(surface_y + 6))
    y1 = min(vy1, int(floor_y + .06 * height), height)
    result: dict[str, Any] = {"detected": False, "confidence": 0.0,
                              "box_px": None, "edge_density": 0.0,
                              "candidate_count": 0, "method": "final_frame_irregular_edge"}
    if x1 <= x0 + 20 or y1 <= y0 + 20:
        return result
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 70)
    edges[:8] = 0; edges[-8:] = 0; edges[:, :8] = 0; edges[:, -8:] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    crop_h, crop_w = gray.shape
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < .15 * crop_w or bw > .85 * crop_w or bh < .12 * crop_h or bh > .60 * crop_h:
            continue
        if bx < .02 * crop_w or bx + bw > .98 * crop_w:
            continue
        cx = (bx + .5 * bw) / crop_w
        if not .15 <= cx <= .85:
            continue
        perimeter = max(2.0 * (bw + bh), 1.0)
        arc = float(cv2.arcLength(contour, False))
        complexity = arc / perimeter
        if not .45 <= complexity <= 3.0:
            continue
        local_edges = edges[by:by + bh, bx:bx + bw]
        density = float(np.mean(local_edges > 0))
        if density < .035:
            continue
        brightness_p65 = float(np.percentile(gray[by:by + bh, bx:bx + bw], 65))
        # The dense stone is intentionally dark and must not be mistaken for
        # residual translucent ice after melting.
        if brightness_p65 < 165.0:
            continue
        score = min(1.0, density / .14) * min(1.0, arc / 700.0)
        candidates.append({"x": int(x0 + bx), "y": int(y0 + by), "width": int(bw),
                           "height": int(bh), "arc_px": arc,
                           "edge_density": density, "brightness_p65": brightness_p65,
                           "complexity": complexity,
                           "score": score})
    result["candidate_count"] = len(candidates)
    if candidates:
        best = max(candidates, key=lambda item: float(item["score"]))
        result.update({"detected": bool(best["score"] >= .20),
                       "confidence": float(best["score"]),
                       "box_px": {key: best[key] for key in ("x", "y", "width", "height")},
                       "edge_density": float(best["edge_density"]),
                       "candidate": best})
    return result


def extract(frames: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    levels, floor_y, debug, reason = water_geometry(frames)
    rois, roi_source = load_rois(); height, width = frames.shape[1:3]
    rois = {"vessel": (.30, .04, .40, .91), "ice": (.34, .20, .32, .42), "stone_seed": (.43, .30, .14, .18), "floor": (.30, .84, .40, .08), **rois}
    vessel_box = roi_pixels(rois["vessel"], width, height); seed_box = roi_pixels(rois.get("stone_seed"), width, height)
    stone_boxes: list[dict[str, float] | None] = []; ice_boxes: list[dict[str, float] | None] = []; previous = None
    if levels is not None and floor_y is not None and vessel_box is not None:
        for frame, level in zip(frames, levels):
            stone = stone_observation(frame, float(level), int(floor_y), vessel_box, seed_box, previous)
            stone_boxes.append(stone); previous = stone if stone is not None else previous
            ice_boxes.append(ice_observation(frame, float(level), int(floor_y), vessel_box, roi_pixels(rois.get("ice"), width, height)))
    else:
        stone_boxes = [None] * len(frames); ice_boxes = [None] * len(frames)
    bottoms = np.asarray([box["bottom_y"] if box else np.nan for box in stone_boxes], dtype=float)
    ice_bottoms = np.asarray([box["bottom_y"] if box else np.nan for box in ice_boxes], dtype=float)
    debug.update({"roi_source": roi_source, "roi": rois, "vessel_box_px": vessel_box,
                  "stone_boxes": stone_boxes, "stone_bottom_y_px": bottoms,
                  "ice_boxes": ice_boxes, "ice_bottom_y_px": ice_bottoms})
    measurement: dict[str, Any] = {}
    if levels is not None and floor_y is not None:
        tail = max(5, min(12, len(levels) // 8))
        initial_y = float(np.median(levels[:tail]))
        final_y = float(np.median(levels[-tail:]))
        initial_height = float(floor_y - initial_y)
        if initial_height > 0:
            measurement.update({
                "initial_waterline_y_px": initial_y,
                "final_waterline_y_px": final_y,
                "vessel_floor_y_px": int(floor_y),
                "initial_water_height_px": initial_height,
                "final_water_height_px": float(floor_y - final_y),
                "water_height_change_norm": float((initial_y - final_y) / initial_height),
            })
    valid_bottoms = bottoms[-max(5, len(bottoms) // 8):]
    valid_bottoms = valid_bottoms[np.isfinite(valid_bottoms)]
    if floor_y is not None and len(valid_bottoms):
        final_bottom = float(np.median(valid_bottoms))
        reference_height = max(float(floor_y - measurement.get("initial_waterline_y_px", 0.0)), 1.0)
        measurement.update({
            "final_stone_bottom_y_px": final_bottom,
            "stone_bottom_gap_norm": float(max(0.0, floor_y - final_bottom) / reference_height),
            "stone_track_coverage": float(np.isfinite(bottoms).mean()),
            "stone_independent_mask": True,
            "stone_contact_frame_count": int(np.sum(np.isfinite(bottoms) & (np.abs(bottoms - floor_y) <= max(8.0, .015 * height)))),
            "ice_track_coverage": float(np.isfinite(ice_bottoms).mean()),
        })
    if levels is not None and floor_y is not None and vessel_box is not None:
        final_ice = final_frame_ice_detection(frames[-1], float(levels[-1]), int(floor_y), vessel_box)
        measurement["final_frame_ice_detected"] = bool(final_ice["detected"])
        measurement["final_frame_ice_detection"] = final_ice
    return measurement, debug, reason


def debug_artifacts(
    frames: np.ndarray,
    debug: dict[str, Any],
    measurement: dict[str, Any],
    root: Path,
    sample_id: str,
    fps: float,
) -> dict[str, str | None]:
    folder = root / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "measurements.json"
    raw.write_text(json.dumps(finite({"measurements": measurement, "detection": debug}), indent=2))
    plot = folder / "plot.png"
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    levels = np.asarray(debug.get("waterline_y_px", []), dtype=float)
    floor = debug.get("vessel_floor_y_px")
    if len(levels) and floor is not None:
        axes[0].plot(np.asarray(floor) - levels, label="water height")
    axes[0].set_ylabel("height (px)"); axes[0].legend(loc="best")
    axes[1].plot(debug.get("stone_bottom_y_px", []), label="independent stone bottom")
    axes[1].plot(debug.get("ice_bottom_y_px", []), label="ice bottom", alpha=.65)
    if floor is not None:
        axes[1].axhline(float(floor), color="black", linestyle="--", label="vessel floor")
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("image y (px)"); axes[1].legend(loc="best")
    fig.suptitle(f"{sample_id}: waterline and stone evidence")
    fig.tight_layout(); fig.savefig(plot, dpi=120); plt.close(fig)
    overlay = folder / "overlay.mp4"
    writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps or 24.0,
                             (frames.shape[2], frames.shape[1]))
    overlay_value: str | None = None
    if writer.isOpened():
        boxes = debug.get("stone_boxes", []); ice_boxes = debug.get("ice_boxes", [])
        for index, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if index < len(levels) and math.isfinite(float(levels[index])):
                y = int(round(levels[index])); cv2.line(bgr, (int(0.34 * bgr.shape[1]), y), (int(0.66 * bgr.shape[1]), y), (0, 220, 0), 2)
            if floor is not None:
                cv2.line(bgr, (int(0.34 * bgr.shape[1]), int(floor)), (int(0.66 * bgr.shape[1]), int(floor)), (0, 220, 220), 2)
            box = boxes[index] if index < len(boxes) else None
            if box:
                p0 = (int(box["x"]), int(box["y"])); p1 = (int(box["x"] + box["width"]), int(box["y"] + box["height"]))
                cv2.rectangle(bgr, p0, p1, (0, 0, 230), 2); cv2.circle(bgr, (int(box["center_x"]), int(box["center_y"])), 4, (0, 0, 255), -1)
                cv2.putText(bgr, f"stone gap {max(0, float(floor) - box['bottom_y']):.1f}px", (p0[0], max(18, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 230), 1)
            ice = ice_boxes[index] if index < len(ice_boxes) else None
            if ice:
                cv2.rectangle(bgr, (int(ice["x"]), int(ice["y"])), (int(ice["x"] + ice["width"]), int(ice["y"] + ice["height"])), (255, 180, 0), 2)
            if index == len(frames) - 1:
                final_detection = measurement.get("final_frame_ice_detection", {})
                final_box = final_detection.get("box_px")
                if final_box:
                    cv2.rectangle(bgr, (int(final_box["x"]), int(final_box["y"])),
                                  (int(final_box["x"] + final_box["width"]), int(final_box["y"] + final_box["height"])),
                                  (0, 165, 255), 3)
                cv2.putText(bgr, f"final-frame ice: {bool(final_detection.get('detected', False))}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 165, 255), 2)
            writer.write(bgr)
        writer.release(); overlay_value = str(overlay)
    return {"directory": str(folder), "plot": str(plot), "measurements": str(raw), "overlay_video": overlay_value}


def evaluate_video(video: str, output: Path, sample_id: str, max_frames: int | None,
                   image_path: str | None, seed: int | None, model: str) -> int:
    try:
        frames, fps = read_video(video, max_frames)
        measurement, debug_data, reason = extract(frames)
        root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"
        debug = debug_artifacts(frames, debug_data, measurement, root, sample_id, fps)
        m1_success = "water_height_change_norm" in measurement
        m2_success = ("stone_bottom_gap_norm" in measurement
                       and measurement.get("stone_independent_mask", False)
                       and measurement.get("stone_track_coverage", 0.0) >= 0.50)
        # An independently detected residual ice body is a definitive physical
        # failure for the completion criterion, even if stone tracking itself
        # is incomplete. It therefore has a measurable zero physics score
        # (and, under V2, a final metric of 0.15) rather than a null score.
        if measurement.get("final_frame_ice_detected", False):
            m2_success = True
        reasons = [] if reason is None else [reason]
        if not m1_success and not reasons:
            reasons.append("P21b_waterline_measurement_missing")
        if not m2_success:
            reasons.append("P21b_stone_separation_missing")
        warnings = []
        if m1_success and measurement["water_height_change_norm"] >= 0:
            warnings.append("P21b_waterline_did_not_decrease")
        if m2_success and measurement.get("stone_bottom_gap_norm", 0.0) > 0.08:
            warnings.append("P21b_stone_not_at_bottom")
        if m2_success and measurement.get("final_frame_ice_detected", False):
            warnings.append("P21b_ice_detected_in_final_frame")
        q1 = direction_score(measurement["water_height_change_norm"], -1) if m1_success else None
        q2 = unit_score(1.0 - measurement["stone_bottom_gap_norm"]) if m2_success and "stone_bottom_gap_norm" in measurement else None
        if measurement.get("final_frame_ice_detected", False):
            q2 = 0.0
        m1_score = recognized_score(q1) if q1 is not None else None
        m2_score = recognized_score(q2) if q2 is not None else None
        measurement["score_normalization"] = {
            "score_range": [0.0, 1.0], "higher_is_better": True,
            "M1": {"raw_measurement": "water_height_change_norm",
                   "physics_score": q1, "recognition_score": 0.15 if m1_success else 0.0,
                   "formula": "0.15 + 0.85 * (1 if water_height_change_norm < 0 else 0)", "score": m1_score},
            "M2": {"raw_measurement": "stone_bottom_gap_norm",
                   "physics_score": q2, "recognition_score": 0.15 if m2_success else 0.0,
                   "formula": "0.15 + 0.85 * (0 if final_frame_ice_detected else clip(1 - stone_bottom_gap_norm, 0, 1))", "score": m2_score},
        }
        payload = result_payload(
            video, image_path, seed, model,
            m1_success, m1_score,
            m2_success, m2_score,
            measurement, "; ".join(reasons) if reasons else None, warnings, debug, sample_id,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"
        folder = root / sample_id; folder.mkdir(parents=True, exist_ok=True)
        error_path = folder / "error.txt"; error_path.write_text(error + "\n")
        payload = result_payload(video, image_path, seed, model, False, None, False, None, None,
                                 error, [], {"directory": str(folder), "error": str(error_path)}, sample_id)
        print(error, file=sys.stderr)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2)); return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(finite(payload), indent=2, allow_nan=False)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-contained P21b evaluator")
    parser.add_argument("--video"); parser.add_argument("--task_id", default=TASK_ID); parser.add_argument("--output")
    parser.add_argument("--videos"); parser.add_argument("--outdir"); parser.add_argument("--model", default="unknown")
    parser.add_argument("--image-path", "--image_path", dest="image_path"); parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-id"); parser.add_argument("--max-frames", type=int); args = parser.parse_args()
    if args.task_id.lower() != TASK_ID.lower():
        parser.error(f"this evaluator only supports {TASK_ID}")
    if args.video:
        if not args.output:
            parser.error("--video requires --output")
        return evaluate_video(args.video, Path(args.output), args.sample_id or Path(args.video).stem,
                              args.max_frames, args.image_path, args.seed, args.model)
    if not args.videos:
        parser.error("provide --video or --videos")
    source = Path(args.videos); files = [source] if source.is_file() else sorted(source.glob("*.mp4"))
    if not files:
        print(f"no MP4 videos found under {source}", file=sys.stderr); return 1
    outdir = Path(args.outdir or "eval_results"); json_dir = outdir / "json"; json_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, 1):
        rc = evaluate_video(str(path), json_dir / f"{path.stem}.json", path.stem,
                            args.max_frames, args.image_path, args.seed, args.model)
        print(f"[{index}/{len(files)}] {path.stem} {'OK' if rc == 0 else 'ERROR'}")
    with (outdir / "results.csv").open("w", newline="") as handle:
        fields = ["task_id", "video_path", "image_path", "seed", "model", "M1_extract_success", "M1", "M2_extract_success", "M2", "M3", "failure_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for path in sorted(json_dir.glob("*.json")):
            payload = json.loads(path.read_text()); metrics = payload["metrics"]
            writer.writerow({"task_id": payload["task_id"], "video_path": payload["video_path"], "image_path": payload["image_path"],
                             "seed": payload["seed"], "model": payload["model"],
                             "M1_extract_success": metrics["M1"]["extract_success"], "M1": metrics["M1"]["metric"],
                             "M2_extract_success": metrics["M2"]["extract_success"], "M2": metrics["M2"]["metric"],
                             "M3": None, "failure_reason": payload["verbose"]["failure_reason"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
